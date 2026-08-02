import json
import asyncio
import logging
from fastapi import APIRouter, Query, Depends
from fastapi.responses import StreamingResponse

from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.api.dependencies import get_retriever, get_generator
from src.core.cache import get_cached_response, set_cached_response  # add cache

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
        try:
            # cache check
            cached = await get_cached_response(session_id, query)
            if cached:
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
            try:
                search_task = asyncio.create_task(
                    retriever.search(query, collection_name=session_id, top_k=8)
                )
                while not search_task.done():
                    done, _ = await asyncio.wait({search_task}, timeout=STREAM_HEARTBEAT_SECONDS)
                    if search_task in done:
                        break
                    yield stream_event("status", message="Đang tìm kiếm tài liệu liên quan...")

                relevant_docs = search_task.result()
            except asyncio.CancelledError:
                search_task.cancel()
                raise
            except RuntimeError as e:
                yield stream_event("error", message=str(e))
                return
            except Exception:
                logger.error(
                    "Error while retrieving documents for session %s.", session_id, exc_info=True
                )
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
            try:
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
                await set_cached_response(session_id, query, full_response_text, sources)

            except (RuntimeError, ConnectionError) as e:
                yield json.dumps({"type": "error", "message": f"\n\n*({str(e)})*"}) + "\n"
            except Exception:
                logger.error("Error while streaming LLM response.", exc_info=True)
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
            yield json.dumps({"type": "error", "message": "Lỗi luồng hệ thống."}) + "\n"

    return StreamingResponse(stream_result(), media_type="application/x-ndjson")
