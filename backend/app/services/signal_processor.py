import numpy as np
from scipy import signal as sp
from scipy.signal import periodogram


def detrend(sig: np.ndarray, lambda_val: int = 100) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al.)."""
    T = len(sig)
    H = np.eye(T)
    ones = np.ones(T)
    D = np.diag(ones[:-2], -2) - 2 * np.diag(ones[:-1], -1) + np.diag(ones)
    D = D[2:]
    return (H - np.linalg.inv(H + lambda_val ** 2 * D.T @ D)) @ sig


def bandpass(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    b, a = sp.butter(2, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return sp.filtfilt(b, a, sig.astype(np.float64))


def fft_peak_hz(sig: np.ndarray, fs: float, low: float, high: float) -> float:
    N = 1
    while N < len(sig):
        N *= 2
    freqs, pxx = periodogram(sig, fs=fs, nfft=N * 4, detrend=False)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(pxx[mask])])


def compute_snr(
    sig: np.ndarray, hr_hz: float, fs: float, low: float, high: float
) -> float:
    N = 1
    while N < len(sig):
        N *= 2
    freqs, pxx = periodogram(sig, fs=fs, nfft=N * 4, detrend=False)
    dev = 6.0 / 60.0
    sig_mask = (
        ((freqs >= hr_hz - dev) & (freqs <= hr_hz + dev))
        | ((freqs >= 2 * hr_hz - dev) & (freqs <= 2 * hr_hz + dev))
    )
    noise_mask = (freqs >= low) & (freqs <= high) & ~sig_mask
    s = pxx[sig_mask].sum()
    n = pxx[noise_mask].sum()
    return float(10.0 * np.log10(s / n)) if n > 0 else 0.0


def compute_heart_rate(
    bvp_buffer: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.75,
    high_hz: float = 2.5,
) -> tuple[float, float]:
    """
    Full pipeline: cumsum → detrend → bandpass → FFT.
    Returns (heart_rate_bpm, snr_db).
    """
    sig = np.cumsum(bvp_buffer.astype(np.float64))
    sig = detrend(sig)
    sig = bandpass(sig, fs, low_hz, high_hz)
    hr_hz = fft_peak_hz(sig, fs, low_hz, high_hz)
    snr = compute_snr(sig, hr_hz, fs, low_hz, high_hz)
    return hr_hz * 60.0, snr
