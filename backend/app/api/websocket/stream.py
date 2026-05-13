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

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.blink_detector import BlinkDetector
from app.services.rppg_engine import SessionState
from app.services.signal_processor import compute_heart_rate
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

            # Face detection
            crop, bbox = state.face_detector.crop_resize(frame, state.engine.img_size)
            blink.push(frame, bbox)

            await ws.send_text(json.dumps({
                "type": "face",
                "detected": bbox is not None,
                "bbox": list(bbox) if bbox else None,
            }))

            # rPPG inference
            bvp_buf = sess.push_frame(crop)
            if bvp_buf is not None:
                hr, snr = compute_heart_rate(
                    bvp_buf,
                    fs=settings.fps,
                    low_hz=settings.hr_low_hz,
                    high_hz=settings.hr_high_hz,
                )
                await ws.send_text(json.dumps({
                    "type": "vitals",
                    "heart_rate": round(hr, 1),
                    "blink_rate": round(blink.get_rate(), 1),
                    "snr_db": round(snr, 2),
                    "bvp_window": [round(v, 4) for v in bvp_buf[-60:].tolist()],
                }))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during processing: %s", ws.client)
            break
        except Exception as exc:
            # Lỗi xử lý frame — log nhưng KHÔNG đóng connection
            logger.warning("Frame processing error (skipping): %s", exc)
