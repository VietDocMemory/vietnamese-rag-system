import json
import asyncio
import logging
import time
import uuid
from fastapi import APIRouter, Query, Depends
from fastapi.responses import StreamingResponse

from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.api.dependencies import get_retriever, get_generator
from src.core.cache import get_cached_response, set_cached_response  # add cache
from src.core.config import settings
from src.core.inference_tracking import inference_tracker

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)
STREAM_HEARTBEAT_SECONDS = 10


def stream_event(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


# query rag system
@router.get("/ask")
async def ask_rag(
    query: str = Query(...),
    session_id: str = Query(...),
    retriever: RAGRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator),
):
    # stream response generator
    async def stream_result():
        request_id = str(uuid.uuid4())
        request_started = time.perf_counter()
        retrieval_seconds = 0.0
        generation_seconds = 0.0

        async def track(
            *,
            answer: str = "",
            contexts: list | None = None,
            cache_hit: bool = False,
            success: bool = True,
            error: str = "",
            model_metrics: dict | None = None,
        ) -> None:
            await inference_tracker.log(
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "question": query,
                    "answer": answer,
                    "contexts": contexts or [],
                    "cache_hit": cache_hit,
                    "success": success,
                    "error": error,
                    "model_metrics": model_metrics or {},
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "end_to_end_seconds": time.perf_counter() - request_started,
                }
            )

        try:
            # cache check
            cached = await get_cached_response(session_id, query)
            if cached:
                await track(
                    answer=cached["response"],
                    contexts=cached["sources"],
                    cache_hit=True,
                )
                # return sources
                yield json.dumps({"type": "sources", "data": cached["sources"]}) + "\n"
                # fake streaming for smooth ux
                words = cached["response"].split(" ")
                for word in words:
                    yield json.dumps({"type": "content", "data": word + " "}) + "\n"
                    await asyncio.sleep(0.02)
                return

            # run pipeline if no cache
            yield stream_event("status", message="Đang tìm kiếm tài liệu liên quan...")
            retrieval_started = time.perf_counter()
            try:
                search_task = asyncio.create_task(
                    retriever.search(query, collection_name=session_id, top_k=settings.RAG_TOP_K)
                )
                while not search_task.done():
                    done, _ = await asyncio.wait({search_task}, timeout=STREAM_HEARTBEAT_SECONDS)
                    if search_task in done:
                        break
                    yield stream_event("status", message="Đang tìm kiếm tài liệu liên quan...")

                relevant_docs = search_task.result()
                retrieval_seconds = time.perf_counter() - retrieval_started
            except asyncio.CancelledError:
                search_task.cancel()
                raise
            except RuntimeError as e:
                retrieval_seconds = time.perf_counter() - retrieval_started
                await track(success=False, error=str(e))
                yield stream_event("error", message=str(e))
                return
            except Exception:
                retrieval_seconds = time.perf_counter() - retrieval_started
                logger.error(
                    "Error while retrieving documents for session %s.", session_id, exc_info=True
                )
                await track(success=False, error="retrieval_failed")
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Phiên làm việc không tồn tại hoặc dữ liệu chưa sẵn sàng. Vui lòng tải file lại.",
                        }
                    )
                    + "\n"
                )
                return

            # handle empty results
            if not relevant_docs:
                await track(success=False, error="no_relevant_documents")
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Dựa trên tài liệu bạn tải lên, tôi không tìm thấy thông tin phù hợp.",
                        }
                    )
                    + "\n"
                )
                return

            sources = [
                {
                    "page": d.get("page"),
                    "chunk_index": d.get("chunk_index"),
                    "content": d.get("content"),
                }
                for d in relevant_docs
            ]
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # cache response while streaming
            full_response_text = ""
            model_metrics = {}
            try:
                generation_started = time.perf_counter()
                if hasattr(generator, "generate_with_metrics"):
                    generation_task = asyncio.create_task(
                        generator.generate_with_metrics(query, relevant_docs)
                    )
                    try:
                        while not generation_task.done():
                            done, _ = await asyncio.wait(
                                {generation_task}, timeout=STREAM_HEARTBEAT_SECONDS
                            )
                            if generation_task in done:
                                break
                            yield stream_event("status", message="Đang sinh câu trả lời...")
                    except asyncio.CancelledError:
                        generation_task.cancel()
                        raise
                    full_response_text, model_metrics = generation_task.result()
                    yield stream_event("content", data=full_response_text)
                else:
                    answer_stream = generator.generate_stream(query, relevant_docs)
                    answer_iter = answer_stream.__aiter__()

                    while True:
                        chunk_task = asyncio.create_task(answer_iter.__anext__())
                        try:
                            while not chunk_task.done():
                                done, _ = await asyncio.wait(
                                    {chunk_task}, timeout=STREAM_HEARTBEAT_SECONDS
                                )
                                if chunk_task in done:
                                    break
                                yield stream_event("status", message="Đang sinh câu trả lời...")

                            chunk = chunk_task.result()
                        except StopAsyncIteration:
                            break
                        except asyncio.CancelledError:
                            chunk_task.cancel()
                            raise

                        full_response_text += chunk
                        yield stream_event("content", data=chunk)

                # save to redis
                generation_seconds = time.perf_counter() - generation_started
                await set_cached_response(session_id, query, full_response_text, sources)
                await track(
                    answer=full_response_text,
                    contexts=sources,
                    model_metrics=model_metrics,
                )

            except (RuntimeError, ConnectionError) as e:
                generation_seconds = time.perf_counter() - generation_started
                await track(
                    answer=full_response_text,
                    contexts=sources,
                    success=False,
                    error=str(e),
                    model_metrics=model_metrics,
                )
                yield json.dumps({"type": "error", "message": f"\n\n*({str(e)})*"}) + "\n"
            except Exception:
                generation_seconds = time.perf_counter() - generation_started
                logger.error("Error while streaming LLM response.", exc_info=True)
                await track(
                    answer=full_response_text,
                    contexts=sources,
                    success=False,
                    error="generation_failed",
                    model_metrics=model_metrics,
                )
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "\n\n*(Lỗi: Mất kết nối tới mô hình ngôn ngữ AI)*",
                        }
                    )
                    + "\n"
                )

        except Exception:
            logger.error("Unhandled chat stream error.", exc_info=True)
            await track(success=False, error="unhandled_stream_error")
            yield json.dumps({"type": "error", "message": "Lỗi luồng hệ thống."}) + "\n"

    return StreamingResponse(stream_result(), media_type="application/x-ndjson")
