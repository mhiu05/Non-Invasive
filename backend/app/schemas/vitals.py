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
    age: int | None = None
    age_group: str | None = None
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    hrv_ms: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    pnn50: float | None = None
    peak_count: int | None = None


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
    age: int | None = None
    age_group: str | None = None
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    hrv_ms: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    pnn50: float | None = None
    peak_count: int | None = None
