import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "")

DDL = """
CREATE TABLE IF NOT EXISTS depth_items (
  request_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status TEXT NOT NULL,
  image_bytes BYTEA,
  depth_png BYTEA,
  depth_npy BYTEA,
  error TEXT,
  processing_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_depth_items_created_at ON depth_items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_depth_items_status ON depth_items(status);
CREATE INDEX IF NOT EXISTS idx_depth_items_name ON depth_items(name);
CREATE INDEX IF NOT EXISTS idx_depth_items_name_created ON depth_items(name, created_at DESC);
"""


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db(retries=30, sleep_s=1.0):
    last_err = None
    for _ in range(retries):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(DDL)
                conn.commit()
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"DB init failed: {last_err}")
