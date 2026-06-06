# FactorizePhys — Hướng dẫn tối ưu toàn diện

> Tài liệu phân tích kỹ thuật và khuyến nghị cải tiến cho FactorizePhys
> nhằm cải thiện benchmark HR-MAE, RMSE, Pearson, SNR trên custom dataset.

---

## 1. Tổng quan kiến trúc hiện tại

### 1.1 FactorizePhys Pipeline
```
Input (N, 3, T+1, 72, 72)
  → torch.diff(dim=2)     # frame differencing
  → InstanceNorm3d         # per-clip normalization
  → rPPG_FeatureExtractor  # 3D CNN backbone (6 ConvBlock3D + 2 Dropout3d)
  → BVP_Head
    → conv_block (3 ConvBlock3D + Dropout3d)
    → FSAM (NMF rank-1 attention)
      → pre_conv: Conv3d(16→8) + ReLU
      → NMF: 3 multiplicative update steps
      → post_conv: ConvBNReLU(8→8) + Conv3d(8→16)
    → Element-wise multiplication + residual
    → final_layer (2 ConvBlock3D + Conv3d→1ch)
  → rPPG signal (N, T)
```

### 1.2 Params & Configs hiện tại
| Config | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `nf` | [8, 12, 16] | Filter sizes nhỏ, phù hợp model nhẹ |
| `MD_R` | 1 | Rank-1 NMF — optimal theo paper |
| `MD_STEPS` | 3 | NMF iterations |
| `MD_RESIDUAL` | True | Multiplication + residual connection |
| `dropout` | 0.1 | Có thể thấp cho dataset nhỏ |
| `EPOCHS` | 30 | Nhưng thực tế chỉ chạy 10 |
| `LR` | 3e-4 | Adam optimizer |
| `BATCH_SIZE` | 4 | |
| `PATIENCE` | 5 | Early stopping |

### 1.3 Benchmark Baseline (Pre-trained weights)
| Model | MAE (bpm) | RMSE | Pearson | SNR |
|---|---|---|---|---|
| UBFC-rPPG_FactorizePhys | **0.04** | 0.14 | 0.9999 | 3.81 |
| iBVP_FactorizePhys | 0.35 | 0.98 | 0.9960 | 5.36 |
| SCAMPS_FactorizePhys | 0.26 | 0.52 | 0.9988 | 1.34 |
| PURE_FactorizePhys | 0.66 | 1.82 | 0.9863 | 3.97 |

---

## 2. Phân tích điểm yếu & cải tiến

### 2.1 Loss Function — Composite Loss

**Vấn đề**: Chỉ dùng `Neg_Pearson` loss đơn lẻ.

**Giải pháp**: Composite loss 3 thành phần:

```python
L_total = α * L_NegPearson + β * L_Freq + γ * L_appx_error
```

#### a) Negative Pearson Correlation Loss (L_NegPearson)
- Đã có sẵn, tập trung vào **hình dạng sóng** (waveform shape)
- Limitation: không phạt sai tần số nếu correlation vẫn cao

#### b) Frequency Domain Loss (L_Freq) — MỚI
```python
class FrequencyLoss(nn.Module):
    """Penalize mismatch in cardiac frequency band (0.6-3.3 Hz)."""
    def forward(self, pred, label, fps=30):
        # FFT of both signals
        pred_fft = torch.fft.rfft(pred, dim=-1)
        label_fft = torch.fft.rfft(label, dim=-1)
        # Power spectral density
        pred_psd = torch.abs(pred_fft) ** 2
        label_psd = torch.abs(label_fft) ** 2
        # Frequency bins
        T = pred.shape[-1]
        freqs = torch.fft.rfftfreq(T, d=1.0/fps)
        # Focus on cardiac band
        mask = (freqs >= 0.6) & (freqs <= 3.3)
        # L1 loss on PSD in cardiac band
        return F.l1_loss(pred_psd[:, mask], label_psd[:, mask])
```

**Lý do**: Pulse signal cốt lõi nằm trong dải tần 0.6-3.3 Hz. Frequency loss
ép model tập trung vào spectral accuracy, không chỉ waveform correlation.

#### c) FSAM Approximation Error (L_appx_error)
- Model đã output `appx_error = torch.dist(x, att)` — reconstruction quality
- **Hiện tại bị ignore** trong training loop!
- Thêm vào loss → ép NMF factorization chính xác hơn

**Trọng số đề xuất**: `α=1.0, β=0.1, γ=0.01`

### 2.2 Data Augmentation

**Vấn đề**: Không có augmentation nào (chỉ face crop + resize).
Dataset nhỏ (10 subjects × 16 clips = 160 clips) → dễ overfit.

**Giải pháp**:

| Augmentation | Mô tả | Ảnh hưởng |
|---|---|---|
| **Temporal crop** | Random crop 160 frames từ sequence dài hơn | Tăng diversity temporal |
| **Horizontal flip** | Lật ngang video | ×2 data (pulse signal không đổi khi flip) |
| **Brightness jitter** | ±15% brightness | Robust với ánh sáng |
| **Temporal resampling** | Speed ±10% (stretch/compress) | Robust với heart rate variation |
| **Gaussian noise** | σ=0.01 trên pixel values | Regularization |

**Lưu ý quan trọng**: KHÔNG dùng spatial crop khác nhau cho mỗi frame
(phá temporal consistency). Augmentation phải apply **cùng transform cho toàn bộ clip**.

### 2.3 Overlap Chunking

**Vấn đề**: Chunking hiện tại: `clip_i = frames[i*160 : (i+1)*160]`
→ Mất data ở biên (nếu T=2700, chỉ lấy 16 clips × 160 = 2560 frames, bỏ 140 frames cuối).

**Giải pháp**: Overlap 50% (stride = 80 frames)
```python
stride = CHUNK_LENGTH // 2  # 80
clips = []
for start in range(0, T - CHUNK_LENGTH + 1, stride):
    clips.append(frames[start : start + CHUNK_LENGTH])
```

Kết quả: ~32 clips/subject thay vì 16 → **gấp đôi training data**.

### 2.4 Training Strategy

#### a) More Epochs + Larger Patience
- **60 epochs** (thay 30), **patience=10** (thay 5)
- Training logs cho thấy loss vẫn đang giảm ở epoch 10 (0.92→0.25)

#### b) CosineAnnealingWarmRestarts
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```
**Lý do**: Warm restarts giúp thoát khỏi local minima, đặc biệt hiệu quả cho dataset nhỏ.

#### c) AdamW + Weight Decay
```python
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
```
**Lý do**: Weight decay cải thiện generalization, AdamW decouple weight decay khỏi gradient.

#### d) Gradient Accumulation
Với batch_size nhỏ (4), accumulate gradient qua 2-4 steps → effective batch = 8-16.

#### e) Per-sample Normalization thay Per-batch
```python
# Hiện tại (per-batch):
pred_n = (pred - pred.mean()) / (pred.std() + 1e-8)
# Đề xuất (per-sample):
pred_n = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-8)
```

### 2.5 FSAM Configuration Tuning

#### a) Tăng MD_STEPS từ 3→5
- Paper thử nghiệm 4 steps; 5 steps cho thêm convergence cho NMF
- Trade-off: ~20% chậm hơn nhưng factorization chính xác hơn

#### b) MD_RESIDUAL=True (giữ nguyên)
- Kết quả inference cho thấy `_FSAM_Res` weights tốt nhất

#### c) align_channels
- Hiện tại: `nf[2] // 2 = 8` channels cho FSAM
- Có thể thử tăng lên 12 hoặc 16 để NMF có nhiều features hơn

### 2.6 Dropout & Regularization

- Tăng dropout từ **0.1 → 0.2** cho dataset nhỏ (10 subjects)
- Thêm **DropPath** (stochastic depth) cho Conv blocks
- **Label smoothing**: Thay vì z-score label chính xác, thêm noise nhỏ

### 2.7 K-Fold Cross-Validation (Leave-One-Subject-Out)

**Vấn đề**: Random split 80/20 có thể train/val chứa clips từ cùng subject
→ data leakage, val metrics quá lạc quan.

**Giải pháp**: Leave-One-Subject-Out (LOSO)
```
Fold 1: Train on S2-S10, Val on S1
Fold 2: Train on S1,S3-S10, Val on S2
...
Fold 10: Train on S1-S9, Val on S10
→ Final: average metrics across 10 folds
```

---

## 3. Inference Optimization

### 3.1 Test-Time Augmentation (TTA)

```python
# Multiple overlapping windows
stride = CHUNK_LENGTH // 4  # 40 frames
all_predictions = []
for start in range(0, T - CHUNK_LENGTH + 1, stride):
    clip = frames[start:start+CHUNK_LENGTH]
    pred = model(clip)
    all_predictions.append(pred)
# Weighted average where center has higher weight
final_pred = weighted_average(all_predictions)
```

### 3.2 Welch's Method thay Periodogram đơn

```python
from scipy.signal import welch
# Welch's method với multiple overlapping windows
freqs, psd = welch(signal, fs=30, nperseg=256, noverlap=128)
```

**Lý do**: Welch's method giảm variance của PSD estimate bằng cách 
average nhiều windowed periodograms → peak frequency ổn định hơn.

### 3.3 Adaptive Bandpass

```python
# Step 1: Wide-band FFT to find rough HR
rough_hr = fft_peak(signal, fs, low=0.6, high=3.3)
# Step 2: Narrow bandpass around detected HR
narrow_low = max(0.6, rough_hr/60 - 0.3)
narrow_high = min(3.3, rough_hr/60 + 0.3)
filtered = bandpass(signal, fs, narrow_low, narrow_high)
# Step 3: Fine HR from narrow band
final_hr = fft_peak(filtered, fs, narrow_low, narrow_high)
```

### 3.4 Ensemble với Multiple Pre-trained Weights

Nếu có nhiều weights (PURE, UBFC, SCAMPS, MyData), ensemble predictions:
```python
hr_estimates = [hr_from_model_1, hr_from_model_2, ..., hr_from_model_n]
final_hr = np.median(hr_estimates)  # Median robust hơn mean
```

---

## 4. Post-Processing Improvements

### 4.1 Improved Detrending
- Hiện tại: `lambda_val=100` (fixed)
- Đề xuất: **Adaptive lambda** dựa trên signal length
  ```python
  lambda_val = 10 * (T / 160)  # scale với clip length
  ```

### 4.2 Zero-Phase Bandpass
- Đã dùng `filtfilt` (zero-phase) — tốt
- Tăng filter order từ 1→2 cho rolloff sharper

### 4.3 Peak Refinement
- Sau FFT peak, dùng **parabolic interpolation** để refine frequency:
  ```python
  k_peak = np.argmax(psd[band])
  if 0 < k_peak < len(psd_band)-1:
      alpha = psd_band[k_peak-1]
      beta  = psd_band[k_peak]
      gamma = psd_band[k_peak+1]
      delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
      refined_freq = freqs_band[k_peak] + delta * freq_resolution
  ```

---

## 5. Tổng hợp cải tiến theo priority

| Priority | Cải tiến | Difficulty | Expected Impact |
|---|---|---|---|
| 🔴 P0 | Composite Loss (NegPearson + Freq + AppxErr) | Easy | High |
| 🔴 P0 | Tăng epochs 30→60, patience 5→10 | Trivial | Medium-High |
| 🟡 P1 | Data augmentation (flip, brightness, noise) | Medium | High |
| 🟡 P1 | Overlap chunking (50% overlap) | Easy | Medium |
| 🟡 P1 | Per-sample normalization | Trivial | Medium |
| 🟡 P1 | Welch's method cho inference | Easy | Medium |
| 🟢 P2 | LOSO cross-validation | Medium | Medium |
| 🟢 P2 | Test-Time Augmentation | Medium | Medium |
| 🟢 P2 | CosineAnnealingWarmRestarts | Easy | Low-Medium |
| 🟢 P2 | MD_STEPS 3→5 | Trivial | Low |
| 🔵 P3 | Dropout 0.1→0.2 | Trivial | Low |
| 🔵 P3 | AdamW + weight decay | Trivial | Low |
| 🔵 P3 | Parabolic peak interpolation | Easy | Low |

---

## 6. Files trong folder optimize

```
rPPG/optimize/
├── optimization_guide.md         ← (file này) Tài liệu phân tích
├── optimized_training.ipynb      ← Notebook training tối ưu
└── optimized_inference.ipynb     ← Notebook inference tối ưu + so sánh
```

## 7. References

1. Joshi et al., "FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing", NeurIPS 2024
2. Yu et al., "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer", CVPR 2022
3. Liu et al., "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement", WACV 2023
4. Narayanswamy et al., "BigSmall: Efficient Multi-Task Learning", NeurIPS 2023
5. Lee & Chen, "Temporal Shift Module for Efficient Video Understanding", ICCV 2019
