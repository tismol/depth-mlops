# Depth Estimation - сервис предсказания относительной карты глубины по изображению.

## Ценность для бизнеса
Многие продукты и команды сталкиваются с потребностью понимать геометрию сцены по обычной RGB-картинке: для AR, выделения планов, улучшения сегментаирования и пост-обработки, но LiDAR/3D-пайплайн - дорого и долго.
Подробнее в [one-pager](docs/one-pager.md).

## Архитектура
```text
┌───────────────────────────────────┐
│              USER                 │
│ загрузка / просмотр / скачивание  │
└───────────────┬───────────────────┘
                │ HTTP
                ▼
┌───────────────────────────────────┐
│      UI: Streamlit (:8501)        │
│  - загрузка image / zip           │
│  - история + превью               │
│  - скачивание результатов         │
└───────────────┬───────────────────┘
                │ HTTP
                ▼
┌──────────────────────────────────────────────────────────┐
│                        FastAPI                           │
│  - принимает файлы                                       │
│  - создает запись в Postgres                             │
│  - публикует задачи в Kafka: depth_requests              │
│  - публикует события в Kafka: depth_events               │
└───────────────┬─────────────────────────────────────┬────┘
                │ SQL                                 │ Kafka (producer)
                ▼                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────┐
│        Postgres 16            │     │     Kafka / Kafka UI(:8080)   │
│  table: depth_items           │     │  topic: depth_requests        │
│  - image_bytes                │     │    request_id, name           │
│  - depth_png / depth_npy      │     │  topic: depth_events          │
│  - status / error / ms        │     │    status / time / error / ms │
└───────────────────────────────┘     └───────────────┬───────────────┘
                ▲                                     │ Kafka (consumer)
                │ SQL                                 ▼
                │                     ┌───────────────────────────────┐
                │                     │            WORKER             │
                │                     │ - inference                   │
                │                     │ - пишет depth_png/depth_npy   │
                │                     │ - events (DONE/FAILED)        │
                │                     │ - веса: /models/model.pt      │
                │                     └───────────────┬───────────────┘
                └─────────────────────────────────────┘
```

## Структура и репозиторий

```text
depth-mlops/
  docker-compose.yml
  README.md
  docs/
    one-pager.md
  examples/
    example1.jpg ... example5.jpg           # тестовые изображения для ручной проверки
    zip_example.zip                         # пример архива
  models/
    model.pt                                # веса модели глубины
  services/
    ml_service/
      Dockerfile
      requirements.txt
      app/
        api.py                              # FastAPI endpoints
        worker.py                           # Kafka consumer + inference + db update
        db.py                               # init_db
        kafka_io.py                         # producer/consumer
        model.py                            # PyTorch model + inference
    ui/
      Dockerfile
      requirements.txt
      app.py           # Streamlit UI
```

## Быстрый старт

```bash
git clone https://github.com/tismol/depth-mlops.git
cd depth-mlops
docker compose up --build
```

### Доступные интерфейсы

* UI (Streamlit): [http://localhost:8501](http://localhost:8501)
* Kafka UI: [http://localhost:8080](http://localhost:8080)

Топики:
* `depth_requests` - задания
* `depth_events` - статусы, ошибки и время обработки