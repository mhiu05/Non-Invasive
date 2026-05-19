import numpy as np

from app.services.signal_processor import compute_hrv


def test_compute_hrv_returns_zero_for_short_signal():
    sig = np.zeros(20)
    hrv = compute_hrv(sig, fs=30.0)

    assert hrv["hrv_ms"] == 0.0
    assert hrv["sdnn_ms"] == 0.0
    assert hrv["rmssd_ms"] == 0.0
    assert hrv["pnn50"] == 0.0
    assert hrv["peak_count"] == 0


def test_compute_hrv_measures_variability_for_regular_peaks():
    sig = np.zeros(120)
    peaks = [5, 35, 65, 95]
    sig[peaks] = 1.0
    hrv = compute_hrv(sig, fs=30.0)

    assert hrv["peak_count"] == 4
    assert hrv["rmssd_ms"] >= 0.0
    assert hrv["sdnn_ms"] >= 0.0
    assert hrv["pnn50"] >= 0.0
