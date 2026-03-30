#!/usr/bin/env python3
"""Создаёт базу из DATABASE_URL (если её ещё нет) и таблицы tasks / task_items."""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)


def main() -> None:
    from app.core.config import settings
    from app.core.db import init_db

    if not settings.database_url:
        print(
            "DATABASE_URL не задан. Скопируйте .env.example в .env и укажите подключение.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not init_db():
        print(
            "init_db не удался. Проверьте, что Postgres запущен, корректен DATABASE_URL, "
            "и у пользователя есть права на CREATE DATABASE (если базы ещё нет) и на DDL.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Postgres готов: база и таблицы tasks / task_items.")


if __name__ == "__main__":
    main()
