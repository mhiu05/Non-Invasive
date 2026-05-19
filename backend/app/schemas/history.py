from pydantic import BaseModel
from typing import Any


class HistorySummary(BaseModel):
    id: str
    created_at: str
    type: str
    filename: str | None = None
    session_id: str | None = None
    duration_sec: float | None = None
    heart_rate: float | None = None
    blink_rate: float | None = None
    snr_db: float | None = None
    age: int | None = None
    age_group: str | None = None
    bandpass_low_hz: float | None = None
    bandpass_high_hz: float | None = None
    hrv_ms: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    pnn50: float | None = None
    peak_count: int | None = None


class HistoryDetailResponse(HistorySummary):
    result: dict[str, Any] | None = None
