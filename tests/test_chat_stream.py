import asyncio
import json

import src.api.routers.chat as chat


class SlowRetriever:
    async def search(self, query, collection_name, top_k):
        await asyncio.sleep(0.03)
        return [{"page": 1, "chunk_index": 0, "content": "demo context"}]


class SlowGenerator:
    async def generate_stream(self, query, contexts):
        await asyncio.sleep(0.03)
        yield "demo answer"


async def test_ask_stream_sends_heartbeat_while_waiting(monkeypatch):
    async def fake_get_cached_response(session_id, query):
        return None

    async def fake_set_cached_response(session_id, query, response, sources):
        return None

    monkeypatch.setattr(chat, "STREAM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(chat, "get_cached_response", fake_get_cached_response)
    monkeypatch.setattr(chat, "set_cached_response", fake_set_cached_response)

    response = await chat.ask_rag("question", "session", SlowRetriever(), SlowGenerator())

    events = []
    async for raw in response.body_iterator:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        events.append(json.loads(text))

    event_types = [event["type"] for event in events]
    assert event_types[0] == "status"
    assert "sources" in event_types
    assert "content" in event_types
    assert event_types.count("status") >= 2
