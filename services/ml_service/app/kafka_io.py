import os
import json
from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC_REQUESTS", "depth_requests")


def make_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def make_consumer(group_id):
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
