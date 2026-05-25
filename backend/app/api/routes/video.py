import json
import logging
import sqlite3
import tempfile
import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, status, Depends
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.vitals import VideoResultResponse

from app.services.history_store import save_history_record
from app.services.rppg_engine import SessionState
from app.services.signal_processor import (
    compute_heart_rate,
    compute_hrv,
    process_bvp,
    get_age_group,
    get_bandpass_by_age,
)
from app.core.security import get_current_user
import app.core.lifespan as state

logger = logging.getLogger(__name__)
router = APIRouter()

DB_PATH = os.path.join(os.path.dirname(__file__), "../../..", "video_jobs.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_jobs_db() -> None:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            result TEXT,
            error TEXT,
            file_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _update_job_status(job_id: str, status_value: str, conn: sqlite3.Connection, result: str | None = None, error: str | None = None) -> None:
    updated_at = datetime.utcnow().isoformat()
    if result is not None or error is not None:
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE id = ?",
            (status_value, result, error, updated_at, job_id),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status_value, updated_at, job_id),
        )


init_jobs_db()

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _build_history_payload(
    filename: str,
    duration_sec: float,
    heart_rate: float,
    snr_db: float,
    age: int | None,
    age_group: str,
    low_hz: float,
    high_hz: float,
    hrv: dict[str, float | int],
    peak_count: int,
    user_id: str | None = None,
    extra_result: dict | None = None,
) -> dict:
    return {
        "type": "video",
        "filename": filename,
        "session_id": None,
        "duration_sec": round(duration_sec, 2),
        "heart_rate": round(heart_rate, 2),
        "snr_db": round(snr_db, 2),
        "age": age,
        "age_group": age_group,
        "bandpass_low_hz": low_hz,
        "bandpass_high_hz": high_hz,
        "hrv_ms": round(hrv["hrv_ms"], 2),
        "sdnn_ms": round(hrv["sdnn_ms"], 2),
        "rmssd_ms": round(hrv["rmssd_ms"], 2),
        "pnn50": round(hrv["pnn50"], 2),
        "peak_count": peak_count,
        "user_id": user_id,
        "result": extra_result,
    }


def _create_job_record(job_id: str, file_path: str) -> None:
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO jobs (id, status, created_at, updated_at, file_path) VALUES (?, ?, ?, ?, ?)",
        (job_id, "pending", now, now, file_path),
    )
    conn.commit()
    conn.close()


@router.post("/video/upload-async")
async def upload_video_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    age: int | None = Form(None),
    current_user: dict | None = Depends(get_current_user),
):
    if state.engine is None or state.face_detector is None:
        raise HTTPException(503, "Model not loaded yet")

    if age is not None and age < 0:
        raise HTTPException(400, "Tuổi phải là số nguyên không âm")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")

    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    _create_job_record(job_id, tmp_path)
    user_id = current_user["id"] if current_user else None
    background_tasks.add_task(process_video_job, job_id, tmp_path, file.filename or "unknown", age, user_id)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "status": "pending"},
    )


def process_video_job(job_id: str, path: str, filename: str, age: int | None, user_id: str | None = None) -> None:
    conn = _get_conn()
    try:
        _update_job_status(job_id, "running", conn)
        conn.commit()

        bvp_values, total_frames = _process_video(path)
        if len(bvp_values) < 30:
            raise RuntimeError("Video quá ngắn hoặc không phát hiện được mặt")

        duration_sec = total_frames / max(settings.fps, 1)
        age_group = get_age_group(age)
        low_hz, high_hz = get_bandpass_by_age(age)
        bvp_array = np.array(bvp_values)
        signal = process_bvp(bvp_array, fs=settings.fps, low_hz=low_hz, high_hz=high_hz)
        heart_rate, snr_db = compute_heart_rate(
            bvp_array,
            fs=settings.fps,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        hrv = compute_hrv(signal, settings.fps)

        result = {
            "filename": filename,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "heart_rate": round(heart_rate, 2),
            "snr_db": round(snr_db, 2),
            "bvp_signal": [round(float(v), 4) for v in bvp_values],
            "age": age,
            "age_group": age_group,
            "bandpass_low_hz": low_hz,
            "bandpass_high_hz": high_hz,
            "hrv_ms": round(hrv["hrv_ms"], 2),
            "sdnn_ms": round(hrv["sdnn_ms"], 2),
            "rmssd_ms": round(hrv["rmssd_ms"], 2),
            "pnn50": round(hrv["pnn50"], 2),
            "peak_count": int(hrv["peak_count"]),
        }

        save_history_record(
            _build_history_payload(
                filename=filename,
                duration_sec=duration_sec,
                heart_rate=heart_rate,
                snr_db=snr_db,
                age=age,
                age_group=age_group,
                low_hz=low_hz,
                high_hz=high_hz,
                hrv=hrv,
                peak_count=int(hrv["peak_count"]),
                user_id=user_id,
                extra_result={
                    "total_frames": total_frames,
                    "bvp_length": len(bvp_values),
                },
            )
        )

        _update_job_status(job_id, "done", conn, result=json.dumps(result), error=None)
        conn.commit()
    except Exception as exc:
        logger.exception("Video async job failed: %s", exc)
        _update_job_status(job_id, "failed", conn, result=None, error=str(exc))
        conn.commit()
    finally:
        conn.close()
        try:
            os.unlink(path)
        except OSError:
            pass


@router.get("/video/jobs/{job_id}")
def get_video_job(job_id: str):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": row["id"],
        "status": row["status"],
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "updated_at": row["updated_at"],
    }


@router.post("/video/upload", response_model=VideoResultResponse)
async def upload_video(
    file: UploadFile = File(...),
    age: int | None = Form(None),
    current_user: dict | None = Depends(get_current_user),
):
    if state.engine is None or state.face_detector is None:
        raise HTTPException(503, "Model not loaded yet")

    if age is not None and age < 0:
        raise HTTPException(400, "Tuổi phải là số nguyên không âm")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")

    # Lưu vào temp file
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        bvp_values, total_frames = _process_video(tmp_path)
    finally:
        os.unlink(tmp_path)

    if len(bvp_values) < 30:
        raise HTTPException(422, "Video quá ngắn hoặc không phát hiện được mặt")

    duration_sec = total_frames / max(settings.fps, 1)

    age_group = get_age_group(age)
    low_hz, high_hz = get_bandpass_by_age(age)
    bvp_array = np.array(bvp_values)
    signal = process_bvp(bvp_array, fs=settings.fps, low_hz=low_hz, high_hz=high_hz)
    heart_rate, snr_db = compute_heart_rate(
        bvp_array,
        fs=settings.fps,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    hrv = compute_hrv(signal, settings.fps)

    save_history_record(
        _build_history_payload(
            filename=file.filename or "unknown",
            duration_sec=duration_sec,
            heart_rate=heart_rate,
            snr_db=snr_db,
            age=age,
            age_group=age_group,
            low_hz=low_hz,
            high_hz=high_hz,
            hrv=hrv,
            peak_count=hrv["peak_count"],
            user_id=current_user["id"] if current_user else None,
            extra_result={
                "total_frames": total_frames,
                "bvp_length": len(bvp_values),
            },
        )
    )

    return VideoResultResponse(
        filename=file.filename or "unknown",
        total_frames=total_frames,
        duration_sec=round(duration_sec, 2),
        heart_rate=round(heart_rate, 2),
        snr_db=round(snr_db, 2),
        bvp_signal=[round(v, 4) for v in bvp_values],
        age=age,
        age_group=age_group,
        bandpass_low_hz=low_hz,
        bandpass_high_hz=high_hz,
        hrv_ms=round(hrv["hrv_ms"], 2),
        sdnn_ms=round(hrv["sdnn_ms"], 2),
        rmssd_ms=round(hrv["rmssd_ms"], 2),
        pnn50=round(hrv["pnn50"], 2),
        peak_count=hrv["peak_count"],
    )


def _process_video(path: str) -> tuple[list[float], int]:
    """Đọc video, chạy inference từng frame. Trả (bvp_values, n_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(422, "Không mở được file video")

    sess = SessionState(state.engine, fps=settings.fps)
    bvp_values: list[float] = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        crop, bbox = state.face_detector.crop_resize(frame, state.engine.img_size)

        result = sess.push_frame(crop)
        if result is not None:
            bvp_values.extend(result.tolist())
            sess.reset()  # reset buffer, tiếp tục tích lũy

    cap.release()
    return bvp_values, total_frames
