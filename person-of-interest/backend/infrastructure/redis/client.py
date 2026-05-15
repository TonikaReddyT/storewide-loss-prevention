"""Redis client — Singleton pattern."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import redis

from backend.core.config import get_config

log = logging.getLogger("poi.redis")


class RedisClient:
    """Thread-safe singleton Redis client."""

    _instance: Optional[RedisClient] = None
    _lock = threading.Lock()

    def __new__(cls) -> RedisClient:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cfg = get_config()
                    instance._pool = redis.ConnectionPool(
                        host=cfg.redis_host,
                        port=cfg.redis_port,
                        db=cfg.redis_db,
                        decode_responses=True,
                    )
                    instance._client = redis.Redis(connection_pool=instance._pool)
                    cls._instance = instance
        return cls._instance

    @property
    def client(self) -> redis.Redis:
        return self._client

    def ping(self) -> bool:
        return self._client.ping()

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
