import pytest
import os
from unittest.mock import MagicMock
from redis.exceptions import ConnectionError as RedisConnectionError
from src.core import cache
from src.core.cache import _hash_query


class FailingRedis:
    async def set(self, *args, **kwargs):
        raise RedisConnectionError("redis down")

    async def get(self, *args, **kwargs):
        raise RedisConnectionError("redis down")

    async def delete(self, *args, **kwargs):
        raise RedisConnectionError("redis down")


# MOCK testing for fast CI
def test_cache_hashing():
    """Test Redis hashing logic doesn't throw errors"""
    h = _hash_query("session-123", "Câu hỏi test!")
    assert "cache:session-123" in h


async def test_cache_falls_back_to_memory_when_redis_is_unavailable(monkeypatch):
    cache._memory_store.clear()
    monkeypatch.setattr(cache, "_redis_failure_logged", False)
    monkeypatch.setattr(cache, "redis_db", FailingRedis())

    await cache.set_upload_status("session-redis-down", "Đang tiếp nhận file...", 60)
    assert await cache.get_upload_status("session-redis-down") == "Đang tiếp nhận file..."

    sources = [{"page": 1, "content": "demo"}]
    await cache.set_cached_response("session-redis-down", "Câu hỏi?", "Câu trả lời", sources, 60)
    assert await cache.get_cached_response("session-redis-down", "Câu hỏi?") == {
        "response": "Câu trả lời",
        "sources": sources,
    }

    await cache.clear_session_data("session-redis-down")
    assert await cache.get_upload_status("session-redis-down") == "Không tìm thấy phiên xử lý."


@pytest.fixture
def mock_heavy_models(mocker):
    """
    Mock the heavy BGE-M3 model so it doesn't download 2.2GB on CI runs.
    Return a dummy embedding tensor matching the shape.
    """
    if os.environ.get("DISABLE_HEAVY_MODELS") == "true":
        # Mock BGEM3FlagModel
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": MagicMock(),  # dummy arrays
            "lexical_weights": [{"123": 0.5}],
        }

        mocker.patch("src.core.model_manager.ModelManager._embed_model", mock_model)
        mocker.patch("src.core.model_manager.ModelManager.get_embed_model", return_value=mock_model)

        # Mock CrossEncoder
        mocker.patch("src.retrieval.search_engine.CrossEncoder", return_value=MagicMock())
        yield mock_model
    else:
        yield None


def test_heavy_models_mocked(mock_heavy_models):
    """
    Ensure the heavy models mockup works if CI environment variable is set.
    """
    if os.environ.get("DISABLE_HEAVY_MODELS") == "true":
        from src.core.model_manager import ModelManager

        model = ModelManager.get_embed_model()
        assert model is not None

        # Check that we bypass the real logic
        res = model.encode(["Test"])
        assert "dense_vecs" in res
