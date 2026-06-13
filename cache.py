import os
import json
import hashlib
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _make_key(prefix: str, *args) -> str:
    key_data = ":".join(str(arg) for arg in args)
    hash_suffix = hashlib.md5(key_data.encode()).hexdigest()[:12]
    return f"{prefix}:{hash_suffix}"


def cache_get(prefix: str, *args) -> Optional[Any]:
    try:
        client = get_redis_client()
        key = _make_key(prefix, *args)
        data = client.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


def cache_set(prefix: str, value: Any, *args, ttl: int = CACHE_TTL) -> bool:
    try:
        client = get_redis_client()
        key = _make_key(prefix, *args)
        client.setex(key, ttl, json.dumps(value))
        return True
    except Exception:
        return False


def cache_delete(prefix: str, *args) -> bool:
    try:
        client = get_redis_client()
        key = _make_key(prefix, *args)
        client.delete(key)
        return True
    except Exception:
        return False


def cache_clear_pattern(pattern: str) -> int:
    try:
        client = get_redis_client()
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
    except Exception:
        pass
    return 0


def get_cached_embedding(text: str) -> Optional[list]:
    return cache_get("embedding", text)


def set_cached_embedding(text: str, embedding: list, ttl: int = CACHE_TTL) -> bool:
    return cache_set("embedding", embedding, text, ttl=ttl)


def get_cached_query_result(query: str, doc_id: str, top_k: int) -> Optional[list]:
    return cache_get("query", query, doc_id, top_k)


def set_cached_query_result(query: str, doc_id: str, top_k: int, results: list, ttl: int = CACHE_TTL) -> bool:
    return cache_set("query", results, query, doc_id, top_k, ttl=ttl)


def invalidate_doc_cache(doc_id: str) -> int:
    return cache_clear_pattern(f"query:*:{doc_id}:*")