import asyncio
import json

import pytest

import src.api.routers.chat as chat


class SlowRetriever:
    async def search(self, query, collection_name, top_k):
        await asyncio.sleep(0.03)
        return [{"page": 1, "chunk_index": 0, "content": "demo context"}]


class SlowGenerator:
    async def cache_key(self, query, contexts, mode):
        return "model-and-document-key"

    async def generate(self, query, contexts, mode):
        await asyncio.sleep(0.03)
        return {
            "answer": "demo answer",
            "sources": contexts,
            "routing": {"selected": "d2l", "reason": "trustmargin"},
        }


async def events(response):
    return [json.loads(raw) async for raw in response.body_iterator]


@pytest.fixture
def cache(monkeypatch):
    saved = {}

    async def get(session_id, query):
        return saved.get((session_id, query))

    async def put(session_id, query, response, sources, routing):
        saved[session_id, query] = dict(response=response, sources=sources, routing=routing)

    monkeypatch.setattr(chat, "STREAM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(chat, "get_cached_response", get)
    monkeypatch.setattr(chat, "set_cached_response", put)
    return saved


async def test_heartbeat_routing_and_selected_content(cache):
    result = await events(
        await chat.ask_rag("question", "session", SlowRetriever(), SlowGenerator())
    )
    kinds = [event["type"] for event in result]
    assert kinds[0] == "status"
    assert kinds.count("status") >= 3
    assert kinds[-3:] == ["sources", "routing", "content"]
    assert result[-2]["data"]["selected"] == "d2l"
    assert result[-1]["data"] == "demo answer"
    assert cache["session", "model-and-document-key"]["routing"]["selected"] == "d2l"


async def test_cached_answer_retains_routing(cache):
    cache["session", "model-and-document-key"] = {
        "response": "cached",
        "sources": [],
        "routing": {"selected": "rag"},
    }

    class CachedGenerator(SlowGenerator):
        async def generate(self, *args):
            raise AssertionError("must not regenerate cached answer")

    result = await events(await chat.ask_rag("q", "session", SlowRetriever(), CachedGenerator()))
    assert result[-2]["cached"] is True
    assert result[-2]["data"]["selected"] == "rag"
    assert result[-1]["data"] == "cached"


async def test_failed_inference_not_cached(cache):
    class FailingGenerator(SlowGenerator):
        async def generate(self, *args):
            raise RuntimeError("GPU failure")

    result = await events(await chat.ask_rag("q", "s", SlowRetriever(), FailingGenerator()))
    assert result[-1] == {"type": "error", "message": "GPU failure"}
    assert not cache
    assert not any(event["type"] == "content" for event in result)


async def test_client_cancellation_cancels_pending_work(cache):
    cancelled = asyncio.Event()

    class WaitingGenerator(SlowGenerator):
        async def generate(self, *args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    response = await chat.ask_rag("q", "s", SlowRetriever(), WaitingGenerator())
    stream = response.body_iterator
    # Wait until the generation task is actually running, then close the client stream.
    async for raw in stream:
        if "TrustMargin" in raw:
            await stream.__anext__()
            await stream.aclose()
            break
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert not cache
