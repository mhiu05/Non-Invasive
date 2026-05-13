"""Standard evaluation metrics for rPPG heart rate estimation."""

import numpy as np
from scipy.signal import periodogram


def compute_mae(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (MAE, standard error) in BPM."""
    err = np.abs(preds - labels)
    return float(np.mean(err)), float(np.std(err) / np.sqrt(len(err)))


def compute_rmse(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (RMSE, standard error) in BPM."""
    sq = (preds - labels) ** 2
    return float(np.sqrt(np.mean(sq))), float(np.sqrt(np.std(sq) / np.sqrt(len(sq))))


def compute_mape(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (MAPE, standard error) as percentage."""
    rel = np.abs(preds - labels) / (np.abs(labels) + 1e-9)
    return float(np.mean(rel) * 100.0), float(np.std(rel) / np.sqrt(len(rel)) * 100.0)


def compute_pearson(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (Pearson r, standard error)."""
    n = len(preds)
    if n < 2:
        return float("nan"), float("nan")
    r = float(np.corrcoef(preds, labels)[0, 1])
    se = float(np.sqrt(max(0.0, (1 - r**2) / (n - 2))))
    return r, se


def compute_snr(
    pred_ppg: np.ndarray,
    hr_label_bpm: float,
    fs: float,
    low_pass: float = 0.6,
    high_pass: float = 3.3,
) -> float:
    """Signal-to-noise ratio (dB) at the HR frequency."""
    N = 1
    while N < len(pred_ppg):
        N *= 2
    freqs, pxx = periodogram(pred_ppg, fs=fs, nfft=N, detrend=False)
    f1, f2 = hr_label_bpm / 60.0, 2.0 * hr_label_bpm / 60.0
    dev = 6.0 / 60.0
    sig_mask = (
        ((freqs >= f1 - dev) & (freqs <= f1 + dev))
        | ((freqs >= f2 - dev) & (freqs <= f2 + dev))
    )
    noise_mask = (freqs >= low_pass) & (freqs <= high_pass) & ~sig_mask
    sig_power = pxx[sig_mask].sum()
    noise_power = pxx[noise_mask].sum()
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def aggregate_metrics(hr_preds: np.ndarray, gt_hrs: np.ndarray, snr_vals: np.ndarray) -> dict:
    """Compute and print all aggregate metrics, return as dict."""
    mae, mae_se = compute_mae(hr_preds, gt_hrs)
    rmse, rmse_se = compute_rmse(hr_preds, gt_hrs)
    mape, mape_se = compute_mape(hr_preds, gt_hrs)
    pearson, pearson_se = compute_pearson(hr_preds, gt_hrs)
    mean_snr = float(np.mean(snr_vals))
    snr_se = float(np.std(snr_vals) / np.sqrt(len(snr_vals)))

    print(f"MAE     : {mae:.4f} +/- {mae_se:.4f} bpm")
    print(f"RMSE    : {rmse:.4f} +/- {rmse_se:.4f} bpm")
    print(f"MAPE    : {mape:.4f} +/- {mape_se:.4f} %")
    print(f"Pearson : {pearson:.4f} +/- {pearson_se:.4f}")
    print(f"SNR     : {mean_snr:.4f} +/- {snr_se:.4f} dB")

    return {
        "MAE":     {"value": mae,     "se": mae_se,     "unit": "bpm"},
        "RMSE":    {"value": rmse,    "se": rmse_se,    "unit": "bpm"},
        "MAPE":    {"value": mape,    "se": mape_se,    "unit": "%"},
        "Pearson": {"value": pearson, "se": pearson_se, "unit": ""},
        "SNR":     {"value": mean_snr,"se": snr_se,     "unit": "dB"},
    }
