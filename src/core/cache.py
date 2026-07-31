import json
import hashlib
import logging
import time
import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from src.core.config import settings

logger = logging.getLogger(__name__)

# init Redis client (async)
redis_db = redis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
    retry=Retry(NoBackoff(), 0),
)

_memory_store: dict[str, tuple[str, float | None]] = {}
_redis_failure_logged = False


def _expiry_time(expire_seconds: int | None) -> float | None:
    if expire_seconds is None:
        return None
    return time.monotonic() + expire_seconds


def _set_memory(key: str, value: str, expire_seconds: int | None = None):
    _memory_store[key] = (value, _expiry_time(expire_seconds))


def _get_memory(key: str):
    item = _memory_store.get(key)
    if item is None:
        return None

    value, expires_at = item
    if expires_at is not None and expires_at <= time.monotonic():
        _memory_store.pop(key, None)
        return None

    return value


def _delete_memory(key: str):
    _memory_store.pop(key, None)


def _log_redis_fallback(exc: Exception):
    global _redis_failure_logged
    if _redis_failure_logged:
        return

    _redis_failure_logged = True
    logger.warning(
        "Redis is unavailable, using in-memory cache fallback. "
        "Cached data will be lost when the API process stops. Error: %s",
        exc,
    )


async def _set_value(key: str, value: str, expire_seconds: int | None = None):
    _set_memory(key, value, expire_seconds)
    try:
        await redis_db.set(key, value, ex=expire_seconds)
    except (RedisError, OSError) as exc:
        _set_memory(key, value, expire_seconds)
        _log_redis_fallback(exc)


async def _get_value(key: str):
    try:
        redis_value = await redis_db.get(key)
        if redis_value is not None:
            return redis_value
    except (RedisError, OSError) as exc:
        _log_redis_fallback(exc)

    return _get_memory(key)


async def _delete_value(key: str):
    _delete_memory(key)
    try:
        await redis_db.delete(key)
    except (RedisError, OSError) as exc:
        _log_redis_fallback(exc)


# set processing status
async def set_upload_status(session_id: str, status: str, expire_seconds: int = 3600):
    """save file processing status, auto delete after 1 hour"""
    await _set_value(f"status:{session_id}", status, expire_seconds)


# get processing status
async def get_upload_status(session_id: str) -> str:
    status = await _get_value(f"status:{session_id}")
    return status or "Không tìm thấy phiên xử lý."


# clear session data
async def clear_session_data(session_id: str):
    await _delete_value(f"status:{session_id}")


def _hash_query(session_id: str, query: str) -> str:
    # hash query key
    clean_query = " ".join(query.lower().split())
    query_hash = hashlib.md5(clean_query.encode()).hexdigest()
    return f"cache:{session_id}:{query_hash}"


# get response from cache
async def get_cached_response(session_id: str, query: str):
    """check if query has been answered in session"""
    key = _hash_query(session_id, query)
    cached_data = await _get_value(key)
    if cached_data:
        return json.loads(cached_data)
    return None


# save response to cache
async def set_cached_response(
    session_id: str, query: str, response: str, sources: list, expire_seconds: int = 86400
):
    """save answer to cache (24h default)"""
    key = _hash_query(session_id, query)
    data = {"response": response, "sources": sources}
    await _set_value(key, json.dumps(data), expire_seconds)
