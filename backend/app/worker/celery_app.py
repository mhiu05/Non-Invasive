from celery import Celery
import os

# Lấy Redis URL từ môi trường (do docker-compose truyền vào)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rppg_worker",
    broker=redis_url,
    backend=redis_url,
    include=["app.worker.tasks.video"],
)

celery_app.conf.update(
    task_serializer="json",
    result_expires=3600,
    task_acks_late=True,          
    worker_prefetch_multiplier=1, 
    task_routes={
        "video.*": {"queue": "video"},
        "default.*": {"queue": "default"},
    },
)
