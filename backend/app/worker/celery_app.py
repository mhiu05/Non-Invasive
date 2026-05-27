"""
celery_app.py — Cấu hình ứng dụng Celery (Background Worker).

Nhiệm vụ:
- Khởi tạo hệ thống Celery để tiếp nhận các tác vụ nặng (xử lý video) chạy ngầm.
- Kết nối tới Message Broker và Backend (thường là Redis) để quản lý hàng đợi công việc.
- Định tuyến các tác vụ (ví dụ: 'video.*' sẽ vào hàng đợi riêng biệt để tránh kẹt mạng).
"""
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
