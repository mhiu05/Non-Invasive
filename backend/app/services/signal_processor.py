import numpy as np
from scipy import signal as sp
from scipy.signal import find_peaks, periodogram


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


AGE_BANDPASS = [
    ((0, 0), (100, 205)),
    ((0, 1), (100, 180)),
    ((1, 2), (98, 140)),
    ((3, 5), (80, 120)),
    ((6, 7), (75, 118)),
    ((8, 200), (60, 100)),
]


def get_bandpass_by_age(age: int | None) -> tuple[float, float]:
    """Trả về giới hạn bandpass Hz theo nhóm tuổi."""
    if age is None or age < 0:
        return 1.0, 1.67
    for (low_age, high_age), (bpm_min, bpm_max) in AGE_BANDPASS:
        if low_age <= age <= high_age:
            return bpm_min / 60.0, bpm_max / 60.0
    return 1.0, 1.67


def get_age_group(age: int | None) -> str:
    if age is None or age < 0:
        return '>= 8 tuổi'
    for (low_age, high_age), _ in AGE_BANDPASS:
        if low_age <= age <= high_age:
            if low_age == 0 and high_age == 0:
                return 'Trẻ sơ sinh'
            if low_age == 0 and high_age == 1:
                return '2–12 tháng'
            if low_age == 1 and high_age == 2:
                return '1–2 năm'
            if low_age == 3 and high_age == 5:
                return '3–5 năm'
            if low_age == 6 and high_age == 7:
                return '6–7 năm'
            return '>= 8 tuổi'
    return '>= 8 tuổi'


def process_bvp(
    bvp_buffer: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.75,
    high_hz: float = 2.5,
) -> np.ndarray:
    sig = np.cumsum(bvp_buffer.astype(np.float64))
    sig = detrend(sig)
    sig = bandpass(sig, fs, low_hz, high_hz)
    return sig


def detect_bvp_peaks(sig: np.ndarray, fs: float = 30.0) -> np.ndarray:
    min_distance = int(fs * 0.35)
    peaks, _ = find_peaks(sig, distance=min_distance, prominence=0.03)
    return peaks


def compute_hrv(sig: np.ndarray, fs: float = 30.0) -> dict[str, float | int]:
    peaks = detect_bvp_peaks(sig, fs)
    if len(peaks) < 3:
        return {
            'hrv_ms': 0.0,
            'sdnn_ms': 0.0,
            'rmssd_ms': 0.0,
            'pnn50': 0.0,
            'peak_count': len(peaks),
        }

    ibi_ms = np.diff(peaks) * (1000.0 / fs)
    if len(ibi_ms) < 2:
        return {
            'hrv_ms': 0.0,
            'sdnn_ms': 0.0,
            'rmssd_ms': 0.0,
            'pnn50': 0.0,
            'peak_count': len(peaks),
        }

    sdnn = float(np.std(ibi_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    pnn50 = float(np.sum(np.abs(np.diff(ibi_ms)) > 50.0) / len(ibi_ms) * 100.0)
    return {
        'hrv_ms': rmssd,
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'pnn50': pnn50,
        'peak_count': len(peaks),
    }


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
    sig = process_bvp(bvp_buffer, fs, low_hz, high_hz)
    hr_hz = fft_peak_hz(sig, fs, low_hz, high_hz)
    snr = compute_snr(sig, hr_hz, fs, low_hz, high_hz)
    return hr_hz * 60.0, snr
