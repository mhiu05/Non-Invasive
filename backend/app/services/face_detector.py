"""
face_detector.py — Fast and stable face bounding box extraction.

Responsibilities:
- Detect faces using MediaPipe Face Mesh.
- Expand and smooth the bounding box over time using Exponential Moving Average (EMA)
  to prevent jitters and stabilize the rPPG signal.
"""

import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """MediaPipe Face Mesh — ổn định, chống rung lắc, tránh false positive."""

    # Số frame liên tiếp không detect được mặt mới coi là mất mặt
    _MISS_TOLERANCE = 8
    # Hệ số EMA cho bbox smoothing (0 = không smooth, 1 = không update)
    _EMA_ALPHA = 0.7

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.7,   # tăng từ 0.5 → giảm false positive
            min_tracking_confidence=0.6,    # tăng từ 0.5
        )
        self._miss_count: int = 0
        self._smoothed: list[float] | None = None  # [x, y, w, h] EMA

    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Trả về (x, y, w, h) vùng mặt mở rộng với EMA smoothing, hoặc None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._miss_count += 1
            if self._miss_count >= self._MISS_TOLERANCE:
                # Mất mặt đủ lâu → reset hoàn toàn
                self._smoothed = None
                self._miss_count = self._MISS_TOLERANCE  # clamp
            # Giữ bbox cũ nếu chưa quá tolerance (giả sử tạm thời bị mất)
            return self._smoothed_bbox()

        # Detect thành công → reset miss counter
        self._miss_count = 0
        face_landmarks = results.multi_face_landmarks[0]
        H, W = frame_bgr.shape[:2]

        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        
        x_min, x_max = int(min(xs) * W), int(max(xs) * W)
        y_min, y_max = int(min(ys) * H), int(max(ys) * H)

        fw = x_max - x_min
        fh = y_max - y_min
        cx, cy = x_min + fw // 2, y_min + fh // 2

        # Mở rộng bbox 1.4x (nhỏ hơn 1.5x cũ để bám sát hơn)
        fw = int(fw * 1.4)
        fh = int(fh * 1.4)
        x = max(0, cx - fw // 2)
        y = max(0, cy - fh // 2)
        fw = min(fw, W - x)
        fh = min(fh, H - y)

        if fw <= 0 or fh <= 0:
            return None

        raw = [float(x), float(y), float(fw), float(fh)]

        # EMA smoothing
        if self._smoothed is None:
            self._smoothed = raw
        else:
            a = self._EMA_ALPHA
            self._smoothed = [
                a * r + (1 - a) * s
                for r, s in zip(raw, self._smoothed)
            ]

        return self._smoothed_bbox()

    def _smoothed_bbox(self) -> tuple[int, int, int, int] | None:
        if self._smoothed is None:
            return None
        x, y, w, h = self._smoothed
        return int(x), int(y), int(w), int(h)

    # ------------------------------------------------------------------

    def crop_resize(self, frame_bgr: np.ndarray, size: int = 72) -> tuple[np.ndarray, tuple | None]:
        """Crop mặt và resize về (size × size). Trả (crop_bgr, bbox | None)."""
        bbox = self.detect(frame_bgr)
        if bbox is not None:
            x, y, w, h = bbox
            crop = frame_bgr[y: y + h, x: x + w]
            if crop.size == 0:
                crop = frame_bgr
                bbox = None
        else:
            crop = frame_bgr

        resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return resized, bbox

    def reset(self) -> None:
        self._miss_count = 0
        self._smoothed = None

    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
