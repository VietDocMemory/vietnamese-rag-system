import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_generator, get_retriever
from src.core.cache import get_cached_response, set_cached_response
from src.generation.generator import RAGGenerator
from src.retrieval.search_engine import RAGRetriever

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)
STREAM_HEARTBEAT_SECONDS = 10


def stream_event(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


async def with_heartbeat(task, message):
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=STREAM_HEARTBEAT_SECONDS)
            if task in done:
                break
            yield stream_event("status", message=message)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@router.get("/ask")
async def ask_rag(
    query: str = Query(..., min_length=1, max_length=20000),
    session_id: str = Query(...),
    retriever: RAGRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator),
    mode: Literal["auto", "rag", "d2l"] = "auto",
):
    async def stream_result():
        task = None
        try:
            yield stream_event("status", message="Đang tìm kiếm tài liệu liên quan...")
            task = asyncio.create_task(retriever.search(query, collection_name=session_id, top_k=8))
            async for heartbeat in with_heartbeat(task, "Đang tìm kiếm tài liệu liên quan..."):
                yield heartbeat
            docs = task.result()
            if not docs:
                yield stream_event("error", message="Không tìm thấy nội dung tài liệu phù hợp.")
                return

            # Changed evidence, mode, parameters and model instance invalidate the cache.
            task = asyncio.create_task(generator.cache_key(query, docs, mode))
            async for heartbeat in with_heartbeat(task, "Đang chuẩn bị model D2L..."):
                yield heartbeat
            key = task.result()
            cached = await get_cached_response(session_id, key)
            if cached:
                result = {
                    "answer": cached["response"],
                    "sources": cached["sources"],
                    "routing": cached["routing"],
                }
            else:
                message = (
                    "Đang tạo hai câu trả lời và chọn bằng TrustMargin..."
                    if mode == "auto"
                    else f"Đang tạo câu trả lời bằng {mode.upper()}..."
                )
                yield stream_event("status", message=message)
                task = asyncio.create_task(generator.generate(query, docs, mode))
                async for heartbeat in with_heartbeat(task, message):
                    yield heartbeat
                result = task.result()
                await set_cached_response(
                    session_id, key, result["answer"], result["sources"], routing=result["routing"]
                )

            # Sources reflect the actual token budget; expose only the selected candidate.
            yield stream_event("sources", data=result["sources"])
            yield stream_event("routing", data=result["routing"], cached=bool(cached))
            yield stream_event("content", data=result["answer"])
        except (RuntimeError, ConnectionError) as exc:
            yield stream_event("error", message=str(exc))
        except Exception:
            logger.exception("Chat inference failed for session %s", session_id)
            yield stream_event(
                "error", message="Không thể xử lý câu hỏi. Kiểm tra tài liệu và dịch vụ model."
            )
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(stream_result(), media_type="application/x-ndjson")
