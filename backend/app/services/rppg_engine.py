"""
rPPGEngine  — ONNX session (global, thread-safe)
SessionState — per-WebSocket state (buffer, prev_frame)
"""

import json
import logging
from collections import deque

import numpy as np
import onnxruntime as ort

from app.services.preprocessor import build_deepphys_input

logger = logging.getLogger(__name__)


class RPPGEngine:
    """Wraps the ONNX session. Stateless — shared across all connections."""

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        with open(config_path) as f:
            cfg = json.load(f)

        self.img_size: int = cfg.get("img_size", 72)
        self.buffer_size: int = cfg.get("chunk", 180)
        self.norm_type: str = cfg.get("norm_type", "DiffNorm")

        logger.info(
            "ONNX model loaded | img_size=%d | buffer=%d | norm=%s | device=%s",
            self.img_size, self.buffer_size, self.norm_type, device,
        )

    def infer(self, inp: np.ndarray) -> float:
        """Run one forward pass. inp shape: (1, 6, H, W) float32."""
        out = self.session.run(None, {self.input_name: inp})[0]
        return float(out.flatten()[0])


class SessionState:
    """Per-WebSocket connection state. Not thread-safe — one per connection."""

    def __init__(self, engine: RPPGEngine, fps: int = 30):
        self.engine = engine
        self.fps = fps
        self._prev_frame: np.ndarray | None = None
        self._bvp: deque[float] = deque(maxlen=engine.buffer_size)

    def push_frame(self, face_crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Push one (img_size x img_size) BGR face crop.
        Returns numpy BVP buffer when it reaches buffer_size, else None.
        """
        if self._prev_frame is None:
            self._prev_frame = face_crop_bgr
            return None

        inp = build_deepphys_input(face_crop_bgr, self._prev_frame)
        bvp_val = self.engine.infer(inp)
        self._bvp.append(bvp_val)
        self._prev_frame = face_crop_bgr

        if len(self._bvp) >= self.engine.buffer_size:
            return np.array(self._bvp, dtype=np.float64)
        return None

    def reset(self) -> None:
        self._prev_frame = None
        self._bvp.clear()
