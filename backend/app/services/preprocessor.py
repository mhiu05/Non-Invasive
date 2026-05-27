"""
preprocessor.py — Input preprocessing for rPPG models.

Responsibilities:
- Transform raw video frames (BGR) into tensor formats expected by AI models.
- Highly optimized for FactorizePhys (chunk-based).
"""

import numpy as np



def build_chunk_input(frames_bgr: list) -> np.ndarray:
    """
    Format input for FactorizePhys (and other chunk-based models).
    frames_bgr: list of (H, W, 3) BGR uint8, length = T
    Returns: (1, 3, T, H, W) float32 — RGB, normalized to [0, 1]
    """
    # 1. Stack frames into a single numpy array: (T, H, W, 3)
    arr = np.array(frames_bgr, dtype=np.float32)
    
    # 2. Vectorized conversion: BGR to RGB (::-1) and normalize to [0, 1]
    arr = arr[..., ::-1] / 255.0
    
    # 3. Transpose to (3, T, H, W) and add batch dimension -> (1, 3, T, H, W)
    return np.expand_dims(arr.transpose(3, 0, 1, 2), axis=0)
