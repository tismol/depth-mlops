import os
import io
import uuid
import zipfile
import os.path as op

import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, StreamingResponse
from typing import Optional

from .db import init_db, get_conn
from .kafka_io import make_producer

app = FastAPI(title="Depth API")

producer = None
TOPIC_REQUESTS = os.getenv("KAFKA_TOPIC_REQUESTS", "depth_requests")
TOPIC_EVENTS = os.getenv("KAFKA_TOPIC_EVENTS", "depth_events")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def is_allowed_image(name):
    return op.splitext(name.lower())[1] in ALLOWED_EXT


def is_zip_junk(path):
    if not path:
        return True
    if path.startswith("__MACOSX/") or "/__MACOSX/" in path:
        return True
    base = op.basename(path).strip()
    if not base or base.startswith("._") or base == ".DS_Store" or base.startswith("."):
        return True
    return False


def insert_item(request_id, name, image_bytes):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO depth_items(request_id, name, status, image_bytes)
                VALUES (%s, %s, %s, %s)
                """,
                (request_id, name, "QUEUED", psycopg2.Binary(image_bytes))
            )
        conn.commit()


def get_latest_row_by_name(name, field):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {field}
                FROM depth_items
                WHERE name=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,)
            )
            return cur.fetchone()


def get_media_type_by_name(name):
    ext = op.splitext(name.lower())[1]
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }.get(ext)


@app.on_event("startup")
def startup():
    global producer
    init_db()
    producer = make_producer()


@app.post("/queue")
async def queue(image: UploadFile = File(...)):
    name = op.basename(image.filename).strip()
    if not is_allowed_image(name):
        raise HTTPException(status_code=400, detail="Недопустимое расширение файла")

    b = await image.read()
    request_id = str(uuid.uuid4())

    insert_item(request_id, name, b)

    producer.send(TOPIC_REQUESTS, {"request_id": request_id, "name": name})
    producer.send(TOPIC_EVENTS, {
        "request_id": request_id,
        "name": name,
        "status": "QUEUED",
        "error": None
    })
    producer.flush(timeout=5)


@app.post("/queue_zip")
async def queue_zip(zip_file: UploadFile = File(...)):
    raw = await zip_file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Файл не является корректным ZIP")

    for info in zf.infolist():
        if info.is_dir():
            continue

        if is_zip_junk(info.filename):
            continue

        name = op.basename(info.filename).strip()
        if not is_allowed_image(name):
            continue

        with zf.open(info) as f:
            img_bytes = f.read()

        request_id = str(uuid.uuid4())
        insert_item(request_id, name, img_bytes)
        producer.send(TOPIC_REQUESTS, {"request_id": request_id, "name": name})

        producer.send(TOPIC_EVENTS, {
            "request_id": request_id,
            "name": name,
            "status": "QUEUED",
            "error": None
        })

    producer.flush(timeout=10)


@app.get("/items")
def list_items(limit: Optional[int] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            if limit is None:
                cur.execute(
                    """
                    SELECT request_id, name, created_at, status, processing_ms, error
                    FROM depth_items
                    ORDER BY created_at DESC
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT request_id, name, created_at, status, processing_ms, error
                    FROM depth_items
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
            rows = cur.fetchall()
    return {"items": rows}


@app.get("/image")
def get_image(name=None, request_id=None):
    if not name and not request_id:
        raise HTTPException(status_code=400, detail="Нужен параметр name или request_id")

    with get_conn() as conn:
        with conn.cursor() as cur:
            if request_id:
                cur.execute("SELECT image_bytes FROM depth_items WHERE request_id=%s", (request_id,))
                row = cur.fetchone()
            else:
                name = op.basename(name).strip()
                row = get_latest_row_by_name(name, "image_bytes")

    if not row or not row["image_bytes"]:
        raise HTTPException(status_code=404, detail="Не найдено")

    media_type = get_media_type_by_name(name)
    return Response(content=bytes(row["image_bytes"]), media_type=media_type)


@app.get("/depth_png")
def get_depth(name=None, request_id=None):
    if not name and not request_id:
        raise HTTPException(status_code=400, detail="Нужен параметр name или request_id")

    with get_conn() as conn:
        with conn.cursor() as cur:
            if request_id:
                cur.execute("SELECT depth_png FROM depth_items WHERE request_id=%s", (request_id,))
                row = cur.fetchone()
            else:
                name = op.basename(name).strip()
                row = get_latest_row_by_name(name, "depth_png")

    if not row or not row["depth_png"]:
        raise HTTPException(status_code=404, detail="Результат ещё не готов")

    return Response(content=bytes(row["depth_png"]), media_type="image/png")


@app.get("/depth_npy")
def get_depth_npy(name=None, request_id=None):
    if not name and not request_id:
        raise HTTPException(status_code=400, detail="Нужен параметр name или request_id")

    with get_conn() as conn:
        with conn.cursor() as cur:
            if request_id:
                cur.execute("SELECT name, depth_npy FROM depth_items WHERE request_id=%s", (request_id,))
                row = cur.fetchone()
            else:
                name = op.basename(name).strip()
                row = get_latest_row_by_name(name, "depth_npy")

    if not row or not row["depth_npy"]:
        raise HTTPException(status_code=404, detail="Результат ещё не готов")

    base = op.splitext(op.basename(name))[0]
    headers = {"Content-Disposition": f'attachment; filename="{base}_depth.npy"'}
    return Response(content=bytes(row["depth_npy"]), media_type="application/octet-stream", headers=headers)


@app.get("/download_processed_zip")
def download_processed_zip():
    buf = io.BytesIO()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, name, depth_png, depth_npy
                FROM depth_items
                WHERE status='DONE' AND (depth_png IS NOT NULL OR depth_npy IS NOT NULL)
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for r in rows:
            base = op.splitext(op.basename(r["name"]))[0]
            if r.get("depth_png"):
                z.writestr(f"{base}_depth.png", bytes(r["depth_png"]))
            if r.get("depth_npy"):
                z.writestr(f"{base}_depth.npy", bytes(r["depth_npy"]))

    buf.seek(0)
    headers = {"Content-Disposition": "attachment; filename=processed_depth.zip"}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.get("/stats")
def stats():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  COUNT(*) FILTER (WHERE status='DONE') AS done,
                  COUNT(*) FILTER (WHERE status='FAILED') AS failed,
                  COUNT(*) FILTER (WHERE status='PROCESSING') AS processing,
                  COUNT(*) FILTER (WHERE status='QUEUED') AS queued
                FROM depth_items
                """
            )
            row = cur.fetchone()

    total = int(row["total"])
    done = int(row["done"])
    failed = int(row["failed"])
    processing = int(row["processing"])
    queued = int(row["queued"])

    remaining = queued + processing

    return {
        "total": total,
        "done": done,
        "remaining": remaining,
        "queued": queued,
        "processing": processing,
        "failed": failed,
    }
