import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings
import psycopg2

conn = psycopg2.connect(settings.database_url)
cur = conn.cursor()

# Таблицы
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    ORDER BY table_name
""")
print("Таблицы:", [r[0] for r in cur.fetchall()])

# Столбцы task_items
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'task_items'
    ORDER BY ordinal_position
""")
print("task_items:", cur.fetchall())

cur.close()
conn.close()