from app.worker.celery_app import celery_app
import time

@celery_app.task(
    name="video.process",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def process_video(self, job_id: str, file_key: str, user_id: str):
    """
    Task xử lý video mẫu.
    Thực tế sẽ gọi rppg_engine.
    """
    print(f"Bắt đầu xử lý video cho job: {job_id}, user: {user_id}, file: {file_key}")
    
    # Giả lập xử lý tốn thời gian
    time.sleep(5)
    
    print(f"Đã xử lý xong video cho job: {job_id}")
    return {"status": "success", "job_id": job_id, "heart_rate": 75}
