"""
video.py — Định nghĩa các tác vụ (Tasks) xử lý video chạy ngầm bằng Celery.

Nhiệm vụ:
- init_worker: Tự động khởi tạo mô hình AI (RPPGEngine, FaceDetector) MỘT LẦN duy nhất khi Worker khởi động, giúp tiết kiệm RAM và thời gian chạy.
- _process_video_local: Dùng OpenCV đọc từng frame của video, bóc tách khuôn mặt và tính toán tín hiệu BVP.
- process_video: Tác vụ (Task) chính thực hiện quy trình khép kín:
    1. Đánh dấu Job đang xử lý ("processing").
    2. Tải file video nặng từ kho lưu trữ đám mây (Storage) về máy cục bộ.
    3. Phân tích video trích xuất nhịp tim (Heart Rate) và chỉ số biến thiên (HRV).
    4. Lưu kết quả vào CSDL và xóa file video để giải phóng dung lượng.
"""
import json
import logging
import os
import tempfile
import numpy as np

from celery import shared_task
from celery.signals import worker_process_init

from app.worker.celery_app import celery_app
from app.core.config import settings
from app.services.rppg_engine import RPPGEngine, SessionState
from app.services.face_detector import FaceDetector
from app.services.history_store import save_history_record
from app.services.signal_processor import (
    compute_heart_rate,
    compute_hrv,
    process_bvp,
    get_age_group,
    get_bandpass_by_age,
)
from app.services.storage import download_file, delete_file
from app.api.routes.video import _get_conn, _update_job_status, _build_history_payload

logger = logging.getLogger(__name__)

engine: RPPGEngine | None = None
face_detector: FaceDetector | None = None

@worker_process_init.connect
def init_worker(**kwargs):
    global engine, face_detector
    logger.info("Initializing models for Celery worker...")
    engine = RPPGEngine(settings.model_path, settings.model_config_path, settings.device)
    face_detector = FaceDetector()
    logger.info("Models initialized successfully.")

def _process_video_local(path: str) -> tuple[list[float], int]:
    import cv2
    from app.services.preprocessor import build_chunk_input
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được file video")

    bvp_values: list[float] = []
    total_frames = 0
    frame_buf = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        crop, bbox = face_detector.crop_resize(frame, engine.img_size)
        frame_buf.append(crop)

        if len(frame_buf) == engine.input_frames:
            try:
                inp = build_chunk_input(frame_buf)
                bvp_arr = engine.infer_chunk(inp)
                bvp_values.extend(bvp_arr.tolist())
            except Exception as e:
                logger.error(f"Inference error chunk: {e}")
            frame_buf.clear()

    cap.release()
    return bvp_values, total_frames

@celery_app.task(
    name="video.process",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def process_video(self, job_id: str, file_key: str, filename: str, age: int | None, user_id: str | None = None):
    logger.info(f"Bắt đầu xử lý video cho job: {job_id}, user: {user_id}, file: {file_key}")
    conn = _get_conn()
    tmp_path = None
    try:
        _update_job_status(job_id, "processing", conn)
        conn.commit()

        # Download from Object Storage
        file_bytes = download_file(file_key)
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        bvp_values, total_frames = _process_video_local(tmp_path)
        if len(bvp_values) < 30:
            raise RuntimeError("Video quá ngắn hoặc không phát hiện được mặt")

        duration_sec = total_frames / max(settings.fps, 1)
        age_group = get_age_group(age)
        low_hz, high_hz = get_bandpass_by_age(age)
        bvp_array = np.array(bvp_values)
        signal = process_bvp(bvp_array, fs=settings.fps, low_hz=low_hz, high_hz=high_hz)
        heart_rate, snr_db = compute_heart_rate(bvp_array, fs=settings.fps, low_hz=low_hz, high_hz=high_hz)
        hrv = compute_hrv(signal, settings.fps)

        import math
        def sanitize_float(v):
            if v is None: return None
            if math.isnan(v) or math.isinf(v): return 0.0
            return round(float(v), 2)

        result = {
            "filename": filename,
            "total_frames": total_frames,
            "duration_sec": sanitize_float(duration_sec),
            "heart_rate": sanitize_float(heart_rate),
            "snr_db": sanitize_float(snr_db),
            "bvp_signal": [round(float(v), 4) for v in bvp_values],
            "age": age,
            "age_group": age_group,
            "bandpass_low_hz": low_hz,
            "bandpass_high_hz": high_hz,
            "hrv_ms": sanitize_float(hrv.get("hrv_ms")),
            "sdnn_ms": sanitize_float(hrv.get("sdnn_ms")),
            "rmssd_ms": sanitize_float(hrv.get("rmssd_ms")),
            "pnn50": sanitize_float(hrv.get("pnn50")),
            "peak_count": int(hrv.get("peak_count", 0)),
        }

        save_history_record(
            _build_history_payload(
                filename=filename,
                duration_sec=duration_sec,
                heart_rate=sanitize_float(heart_rate),
                snr_db=sanitize_float(snr_db),
                age=age,
                age_group=age_group,
                low_hz=low_hz,
                high_hz=high_hz,
                hrv={k: sanitize_float(v) for k, v in hrv.items()},
                peak_count=int(hrv.get("peak_count", 0)),
                user_id=user_id,
                extra_result={"total_frames": total_frames, "bvp_length": len(bvp_values)},
            )
        )

        _update_job_status(job_id, "done", conn, result=json.dumps(result), error=None)
        conn.commit()

        # Delete file from Storage after success
        delete_file(file_key)

    except Exception as exc:
        logger.exception("Video async job failed: %s", exc)
        _update_job_status(job_id, "failed", conn, result=None, error=str(exc))
        conn.commit()
        raise self.retry(exc=exc)
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
