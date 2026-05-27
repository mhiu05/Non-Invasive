from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "rppg_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
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
