# Dùng lại cấu trúc của Backend Dockerfile cho Worker
FROM python:3.11-slim AS builder

WORKDIR /app
COPY backend/requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install celery redis && \
    apt-get purge -y gcc python3-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY backend/app/ ./app/
COPY backend/weights/ ./weights/

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# Chạy Celery worker thay vì uvicorn
CMD ["celery", "-A", "app.worker.celery_app", "worker", "-Q", "video,default", "-c", "4", "--loglevel=info"]
