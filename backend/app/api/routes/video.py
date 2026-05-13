import logging
import tempfile
import os

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.config import settings
from app.schemas.vitals import VideoResultResponse
from app.services.blink_detector import BlinkDetector
from app.services.rppg_engine import SessionState
from app.services.signal_processor import compute_heart_rate
import app.core.lifespan as state

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


@router.post("/video/upload", response_model=VideoResultResponse)
async def upload_video(file: UploadFile = File(...)):
    if state.engine is None or state.face_detector is None:
        raise HTTPException(503, "Model not loaded yet")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")

    # Lưu vào temp file
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        bvp_values, blink_rate, total_frames = _process_video(tmp_path)
    finally:
        os.unlink(tmp_path)

    if len(bvp_values) < 30:
        raise HTTPException(422, "Video quá ngắn hoặc không phát hiện được mặt")

    heart_rate, snr_db = compute_heart_rate(
        np.array(bvp_values),
        fs=settings.fps,
        low_hz=settings.hr_low_hz,
        high_hz=settings.hr_high_hz,
    )
    duration_sec = total_frames / max(settings.fps, 1)

    return VideoResultResponse(
        filename=file.filename or "unknown",
        total_frames=total_frames,
        duration_sec=round(duration_sec, 2),
        heart_rate=round(heart_rate, 2),
        blink_rate=round(blink_rate, 2),
        snr_db=round(snr_db, 2),
        bvp_signal=[round(v, 4) for v in bvp_values],
    )


def _process_video(path: str) -> tuple[list[float], float, int]:
    """Đọc video, chạy inference từng frame. Trả (bvp_values, blink_rate, n_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(422, "Không mở được file video")

    sess = SessionState(state.engine, fps=settings.fps)
    blink = BlinkDetector(fps=settings.fps)
    bvp_values: list[float] = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        crop, bbox = state.face_detector.crop_resize(frame, state.engine.img_size)
        blink.push(frame, bbox)

        result = sess.push_frame(crop)
        if result is not None:
            bvp_values.extend(result.tolist())
            sess.reset()  # reset buffer, tiếp tục tích lũy

    cap.release()
    return bvp_values, blink.get_rate(), total_frames
