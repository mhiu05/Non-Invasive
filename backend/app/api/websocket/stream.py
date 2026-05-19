"""
WebSocket endpoint: nhận JPEG frames từ webcam, trả vitals real-time.

Protocol:
  Client → Server: {"type": "frame", "data": "<base64 JPEG>"}
  Client → Server: {"type": "reset"}
  Server → Client: {"type": "face", "detected": true, "bbox": [x,y,w,h]}
  Server → Client: {"type": "vitals", "heart_rate": 72.5, "blink_rate": 14.1,
                     "snr_db": 8.3, "bvp_window": [...]}
  Server → Client: {"type": "error", "message": "..."}
"""

import asyncio
import base64
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.blink_detector import BlinkDetector
from app.services.rppg_engine import SessionState
from app.services.signal_processor import (
    compute_heart_rate,
    compute_hrv,
    process_bvp,
    get_age_group,
    get_bandpass_by_age,
)
import app.core.lifespan as state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected: %s", ws.client)

    if state.engine is None or state.face_detector is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Model not loaded"}))
        await ws.close()
        return

    import uuid
    session_id = str(uuid.uuid4())[:8]
    frame_count = 0
    last_vitals = None

    sess = SessionState(state.engine, fps=settings.fps)
    blink = BlinkDetector(
        fps=settings.fps,
        low_hz=settings.blink_low_hz,
        high_hz=settings.blink_high_hz,
    )

    while True:
        # ── nhận message ──────────────────────────────────────────
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s", ws.client)
            break

        if not raw:
            continue

        # ── parse JSON ────────────────────────────────────────────
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "reset":
            sess.reset()
            blink.reset()
            continue

        if msg.get("type") != "frame":
            continue

        # ── xử lý frame (lỗi per-frame không đóng connection) ────
        try:
            img_bytes = base64.b64decode(msg["data"])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Face detection — chạy trong thread pool để không block event loop
            loop = asyncio.get_event_loop()
            crop, bbox = await loop.run_in_executor(
                None, state.face_detector.crop_resize, frame, state.engine.img_size
            )

            if bbox is None:
                # Nếu mất mặt -> reset session ngay lập tức để không tính HR từ background tường nhà
                sess.reset()
                blink.reset()
                await ws.send_text(json.dumps({
                    "type": "face",
                    "detected": False,
                    "bbox": None,
                }))
                await ws.send_text(json.dumps({
                    "type": "vitals",
                    "heart_rate": None,
                    "blink_rate": None,
                    "snr_db": None,
                    "bvp_window": [],
                    "buffer_frames": 0,
                    "buffer_needed": sess.engine.input_frames if sess._is_chunk else sess.engine.buffer_size,
                }))
                continue

            # Nếu có mặt thì mới push vào rPPG
            blink.push(frame, bbox)

            if isinstance(msg.get('age'), int) and msg['age'] >= 0:
                sess.age = msg['age']

            frame_count += 1

            await ws.send_text(json.dumps({
                "type": "face",
                "detected": True,
                "bbox": list(bbox),
            }))

            # rPPG inference - chạy non-blocking
            bvp_buf = await loop.run_in_executor(None, sess.push_frame, crop)
            if bvp_buf is not None:
                low_hz, high_hz = get_bandpass_by_age(sess.age)
                sig = await loop.run_in_executor(
                    None,
                    process_bvp,
                    bvp_buf,
                    settings.fps,
                    low_hz,
                    high_hz,
                )
                hr, snr = await loop.run_in_executor(
                    None,
                    compute_heart_rate,
                    bvp_buf,
                    settings.fps,
                    low_hz,
                    high_hz,
                )
                hrv = await loop.run_in_executor(None, compute_hrv, sig, settings.fps)
                
                last_vitals = {
                    "heart_rate": round(hr, 1),
                    "blink_rate": round(blink.get_rate(), 1),
                    "snr_db": round(snr, 2),
                    "age": sess.age,
                    "age_group": get_age_group(sess.age),
                    "bandpass_low_hz": low_hz,
                    "bandpass_high_hz": high_hz,
                    "hrv_ms": round(hrv["hrv_ms"], 2),
                    "sdnn_ms": round(hrv["sdnn_ms"], 2),
                    "rmssd_ms": round(hrv["rmssd_ms"], 2),
                    "pnn50": round(hrv["pnn50"], 2),
                    "peak_count": hrv["peak_count"],
                }

                await ws.send_text(json.dumps({
                    "type": "vitals",
                    **last_vitals,
                    "bvp_window": [round(float(v), 4) for v in bvp_buf.tolist()],
                }))
            else:
                # Send progress while buffer is filling up
                partial = (
                    list(sess._frame_buf) if sess._is_chunk else list(sess._bvp)
                )
                await ws.send_text(json.dumps({
                    "type": "vitals",
                    "heart_rate": None,
                    "blink_rate": round(blink.get_rate(), 1),
                    "snr_db": None,
                    "bvp_window": [],
                    "buffer_frames": len(partial),
                    "buffer_needed": sess.engine.input_frames if sess._is_chunk else sess.engine.buffer_size,
                    "age": sess.age,
                    "age_group": get_age_group(sess.age),
                }))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during processing: %s", ws.client)
            break
        except Exception as exc:
            # Lỗi xử lý frame — log nhưng KHÔNG đóng connection
            logger.exception("Frame processing error (skipping): %s", exc)

    # Save to history when session ends
    if last_vitals and last_vitals.get("heart_rate") is not None:
        duration = frame_count / settings.fps
        if duration >= 5.0:  # Only save meaningful sessions
            try:
                from app.services.history_store import save_history_record
                record = {
                    "type": "realtime",
                    "session_id": f"ws-{session_id}",
                    "duration_sec": round(duration, 1),
                    **last_vitals
                }
                save_history_record(record)
                logger.info("Saved realtime session %s to history (duration: %.1fs)", session_id, duration)
            except Exception as e:
                logger.error("Failed to save realtime history: %s", e)
