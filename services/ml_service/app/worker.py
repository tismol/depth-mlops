import os
import time
import psycopg2
from datetime import datetime

from .db import init_db, get_conn
from .kafka_io import make_consumer, make_producer
from .model import load_model, infer_depth_png_and_npy

DEVICE = os.getenv("DEVICE", "cpu")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "/models/model.pt")

TOPIC_REQUESTS = os.getenv("KAFKA_TOPIC_REQUESTS", "depth_requests")
TOPIC_EVENTS = os.getenv("KAFKA_TOPIC_EVENTS", "depth_events")


def publish_event(producer, request_id, name, status, error=None, processing_ms=None):
    payload = {
        "request_id": request_id,
        "name": name,
        "status": status,
        "error": error,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "service": "worker"
    }
    if processing_ms is not None:
        payload["processing_ms"] = processing_ms

    producer.send(TOPIC_EVENTS, payload)
    producer.flush(timeout=5)


def main():
    init_db()
    model = load_model(device=DEVICE, weights_path=MODEL_WEIGHTS_PATH, encoder_ckpt_path=None)

    consumer = make_consumer(group_id="depth-worker")
    producer = make_producer()

    for msg in consumer:
        request_id = msg.value.get("request_id")
        msg_name = msg.value.get("name")

        if not request_id:
            continue

        row = None
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE depth_items SET status=%s WHERE request_id=%s",
                        ("PROCESSING", request_id),
                    )
                    cur.execute(
                        "SELECT name, image_bytes FROM depth_items WHERE request_id=%s",
                        (request_id,),
                    )
                    row = cur.fetchone()
                conn.commit()

            if row and row.get("name"):
                name = row["name"]
            else:
                name = msg_name

            publish_event(producer, request_id=request_id, name=name, status="PROCESSING", error=None)

            if not row or not row.get("image_bytes"):
                err = "No image_bytes for request_id"
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE depth_items SET status=%s, error=%s WHERE request_id=%s",
                            ("FAILED", err, request_id)
                        )
                    conn.commit()
                publish_event(producer, request_id=request_id, name=name, status="FAILED", error=err)
                continue

            t0 = time.time()
            depth_png, depth_npy = infer_depth_png_and_npy(model, bytes(row["image_bytes"]), DEVICE)
            ms = int((time.time() - t0) * 1000)

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE depth_items
                        SET status=%s, depth_png=%s, depth_npy=%s, processing_ms=%s, error=NULL
                        WHERE request_id=%s
                        """,
                        ("DONE",
                         psycopg2.Binary(depth_png),
                         psycopg2.Binary(depth_npy),
                         ms,
                         request_id)
                    )

                conn.commit()

            publish_event(producer, request_id=request_id, name=name, status="DONE", error=None, processing_ms=ms)

        except Exception as e:
            err = str(e)
            if row and row.get("name"):
                name = row["name"]
            else:
                name = msg_name

            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE depth_items SET status=%s, error=%s WHERE request_id=%s",
                            ("FAILED", err, request_id)
                        )
                    conn.commit()
            except Exception:
                pass

            try:
                publish_event(producer, request_id=request_id, name=name, status="FAILED", error=err)
            except Exception:
                pass


if __name__ == "__main__":
    main()
