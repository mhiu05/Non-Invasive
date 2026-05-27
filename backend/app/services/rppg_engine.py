"""
rPPGEngine  — ONNX inference session (global, thread-safe)
SessionState — per-WebSocket state (buffer, background inference)

Hardcoded to use FactorizePhys ONNX model.
"""

import json
import logging
import os
import threading
from collections import deque

import numpy as np
import onnxruntime as ort

from app.services.preprocessor import build_chunk_input

logger = logging.getLogger(__name__)

# Ensure backend/weights/ directory exists
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WEIGHTS_DIR = os.path.join(_BACKEND_ROOT, "weights")
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

class RPPGEngine:
    """Wraps ONNX inference session. Stateless — shared across all connections."""

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        # Resolve path relative to backend root if it's not absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(_BACKEND_ROOT, model_path)
            
        if not os.path.exists(model_path):
            logger.error("ONNX model not found at: %s", model_path)
            raise FileNotFoundError(f"Missing ONNX model at {model_path}. Please place the file in the weights/ folder.")

        if not os.path.isabs(config_path):
            config_path = os.path.join(_BACKEND_ROOT, config_path)

        with open(config_path) as f:
            cfg = json.load(f)

        self.img_size: int = cfg.get("img_size", 72)
        self.buffer_size: int = cfg.get("chunk", 180)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        logger.info(
            "ONNX backend | %s | img_size=%d | buffer=%d | device=%s",
            os.path.basename(model_path), self.img_size, self.buffer_size, device,
        )

        # Retrieve chunk size from ONNX input shape
        try:
            in_shape = self.session.get_inputs()[0].shape  # ['batch', 3, T, H, W]
            self.input_frames = int(in_shape[2])
        except Exception:
            self.input_frames = self.buffer_size + 1
        logger.info("Chunk model: input_frames=%d", self.input_frames)

        self._input_name: str = self.session.get_inputs()[0].name

    def infer_chunk(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass chunk-wise. inp: (1, 3, T, H, W) float32. Returns (T,) BVP array."""
        out = self.session.run(None, {self._input_name: inp})[0]
        return out.flatten().astype(np.float64)

# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class SessionState:
    """Per-WebSocket connection state. Not thread-safe — one per connection."""

    def __init__(self, engine: RPPGEngine, fps: int = 30):
        self.engine = engine
        self.fps = fps
        self.age: int | None = None
        
        # Lưu raw frame crops (deque giữ tối đa input_frames)
        self._frame_buf: deque[np.ndarray] = deque(maxlen=engine.input_frames)
        self._bvp: deque[float] = deque(maxlen=engine.buffer_size)
        self._step_counter: int = 0
        self._chunk_stride: int = 15  # Chỉ infer 1 lần mỗi 15 frames (~0.5s) để tránh lag
        
        # Threading cho background inference
        self._is_inferring: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def push_frame(self, face_crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Push một (img_size × img_size) BGR face crop.
        Trả về numpy BVP array khi đủ buffer_size, ngược lại trả None.
        """
        self._frame_buf.append(face_crop_bgr)

        if len(self._frame_buf) < self.engine.input_frames:
            return None  # chưa đủ frames

        self._step_counter += 1

        # Lấy copy của BVP buffer hiện tại một cách an toàn
        with self._lock:
            bvp_copy = np.array(self._bvp, dtype=np.float64) if len(self._bvp) > 0 else None

        # Tránh lag: chỉ infer 1 lần mỗi 15 frames và đảm bảo thread cũ đã xong
        if self._step_counter % self._chunk_stride == 1:
            if not self._is_inferring:
                self._is_inferring = True
                frames_copy = list(self._frame_buf)
                threading.Thread(
                    target=self._do_infer_bg, 
                    args=(frames_copy,), 
                    daemon=True
                ).start()

        return bvp_copy

    # ------------------------------------------------------------------

    def _do_infer_bg(self, frames_copy: list[np.ndarray]) -> None:
        """Chạy inference trong background thread để không block WebSocket loop."""
        try:
            inp = build_chunk_input(frames_copy)
            bvp_arr = self.engine.infer_chunk(inp)
            with self._lock:
                self._bvp.clear()
                for v in bvp_arr:
                    self._bvp.append(float(v))
        except Exception as e:
            logger.error("Background infer error: %s", e)
        finally:
            self._is_inferring = False

    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._frame_buf.clear()
            self._step_counter = 0
            self._bvp.clear()
