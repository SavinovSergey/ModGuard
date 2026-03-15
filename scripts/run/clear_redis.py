#!/usr/bin/env python3
"""Очистка Redis (кэш модерации). Запуск: python scripts/run/clear_redis.py"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)


def clear_rs():
    from app.core.config import settings
    if not settings.redis_url:
        print("REDIS_URL не задан, очистка кэша пропущена.")
        return
    try:
        import redis
        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.flushdb()
        print("Кэш Redis очищен.")
    except Exception as e:
        print(f"Ошибка очистки кэша: {e}")


if __name__ == "__main__":
    clear_rs()
