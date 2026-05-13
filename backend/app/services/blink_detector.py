import cv2
import numpy as np
from scipy import signal as sp


class BlinkDetector:
    def __init__(self, fps: int = 30, low_hz: float = 0.1, high_hz: float = 0.9):
        self.fps = fps
        self.low_hz = low_hz
        self.high_hz = high_hz
        self._brightness: list[float] = []

    def push(self, frame_bgr: np.ndarray, bbox: tuple | None) -> None:
        """Thêm 1 frame vào buffer. bbox = (x, y, w, h) hoặc None."""
        if bbox is not None:
            x, y, w, h = bbox
            eye_top = y + int(h * 0.20)
            eye_bot = y + int(h * 0.50)
            strip = frame_bgr[eye_top:eye_bot, x: x + w]
        else:
            H, W = frame_bgr.shape[:2]
            strip = frame_bgr[int(H * 0.20): int(H * 0.50), :]

        gray = cv2.cvtColor(strip if strip.size > 0 else frame_bgr, cv2.COLOR_BGR2GRAY)
        self._brightness.append(float(np.mean(gray)))

    def get_rate(self) -> float:
        """Trả về blinks/phút từ buffer hiện tại."""
        if len(self._brightness) < self.fps * 2:
            return 0.0

        sig = -np.array(self._brightness, dtype=np.float64)
        nyq = self.fps / 2.0
        b, a = sp.butter(2, [self.low_hz / nyq, self.high_hz / nyq], btype="bandpass")
        filtered = sp.filtfilt(b, a, sig)
        peaks, _ = sp.find_peaks(filtered, distance=self.fps)
        duration_min = len(self._brightness) / self.fps / 60.0
        return len(peaks) / max(duration_min, 1e-6)

    def reset(self) -> None:
        self._brightness.clear()
