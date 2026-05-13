from pydantic import BaseModel
from typing import Optional


class FaceMessage(BaseModel):
    type: str = "face"
    detected: bool
    bbox: Optional[list[int]] = None  # [x, y, w, h]


class VitalsMessage(BaseModel):
    type: str = "vitals"
    heart_rate: float
    blink_rate: float
    snr_db: float
    bvp_window: list[float] = []  # last N BVP values for chart


class ErrorMessage(BaseModel):
    type: str = "error"
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class VideoResultResponse(BaseModel):
    filename: str
    total_frames: int
    duration_sec: float
    heart_rate: float
    blink_rate: float
    snr_db: float
    bvp_signal: list[float]
