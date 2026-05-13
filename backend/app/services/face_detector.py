import cv2
import numpy as np


class FaceDetector:
    """OpenCV Haar Cascade face detector — không cần model download thêm."""

    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(xml)

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Trả về (x, y, w, h) vùng mặt mở rộng 1.5x, hoặc None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None

        # Lấy mặt lớn nhất
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        H, W = frame_bgr.shape[:2]

        # Mở rộng bbox 1.5x
        cx, cy = x + fw // 2, y + fh // 2
        fw = int(fw * 1.5)
        fh = int(fh * 1.5)
        x = max(0, cx - fw // 2)
        y = max(0, cy - fh // 2)
        fw = min(fw, W - x)
        fh = min(fh, H - y)
        return int(x), int(y), int(fw), int(fh)

    def crop_resize(self, frame_bgr: np.ndarray, size: int = 72) -> tuple[np.ndarray, tuple | None]:
        """Crop mặt và resize về (size × size). Trả (crop_bgr, bbox)."""
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
