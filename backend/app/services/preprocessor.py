import numpy as np


def diff_normalize(
    frame: np.ndarray, prev_frame: np.ndarray
) -> np.ndarray:
    """(frame_t - frame_t-1) / (frame_t + frame_t-1 + eps) — shape HxWxC."""
    f = frame.astype(np.float32)
    p = prev_frame.astype(np.float32)
    return (f - p) / (f + p + 1e-7)


def standardize(frame: np.ndarray) -> np.ndarray:
    """Z-score toàn frame — shape HxWxC."""
    f = frame.astype(np.float32)
    return (f - f.mean()) / (f.std() + 1e-7)


def build_deepphys_input(
    frame: np.ndarray, prev_frame: np.ndarray
) -> np.ndarray:
    """
    DeepPhys input: 6-channel (DiffNorm 3ch + Standardized 3ch).
    frame/prev_frame: (H, W, 3) BGR uint8
    Returns: (1, 6, H, W) float32
    """
    motion = diff_normalize(frame, prev_frame)          # (H,W,3)
    appearance = standardize(frame)                     # (H,W,3)
    combined = np.concatenate([motion, appearance], axis=2)  # (H,W,6)
    return combined.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,6,H,W)


def build_chunk_input(frames_bgr: list) -> np.ndarray:
    """
    FactorizePhys / chunk-based model input.
    frames_bgr: list of (H, W, 3) BGR uint8, length = T+1  (e.g. 181)
    Returns: (1, 3, T+1, H, W) float32  — RGB, normalized to [0, 1]
    """
    rgb_frames = []
    for f in frames_bgr:
        # BGR → RGB, normalize to [0, 1]
        rgb = f[:, :, ::-1].astype(np.float32) / 255.0
        rgb_frames.append(rgb)

    # Stack: (T+1, H, W, 3) → transpose → (3, T+1, H, W)
    arr = np.stack(rgb_frames, axis=0)          # (T+1, H, W, 3)
    arr = arr.transpose(3, 0, 1, 2)             # (3, T+1, H, W)
    return arr[np.newaxis].astype(np.float32)   # (1, 3, T+1, H, W)
