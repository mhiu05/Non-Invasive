# Architecture — Non-Invasive Health Analysis

---

## Tổng quan hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRONTEND  (port 3002)                    │
│   React 18 + TypeScript + Vite + Tailwind                        │
│                                                                  │
│   Webcam ──► useWebcam (15fps JPEG) ──► useWebSocket             │
│                                              │  WS               │
│   Upload ──► Axios POST ───────────────────►│                   │
└─────────────────────────────────────────────┼────────────────────┘
                                              │ ws://localhost:8001
                                              │ http://localhost:8001
┌─────────────────────────────────────────────┼────────────────────┐
│                    BACKEND  (port 8001)      │                   │
│                                             ▼                    │
│   FastAPI ──► WS /ws/stream                                      │
│            └► POST /video/upload                                 │
│                     │                                            │
│                     ▼                                            │
│   FaceDetector (OpenCV Haar Cascade) ──► crop 72×72             │
│                     │                                            │
│                     ▼                                            │
│   Preprocessor ──► DiffNorm + Standardize ──► (1,6,72,72)       │
│                     │                                            │
│                     ▼                                            │
│   RPPGEngine (ONNX Runtime) ──► BVP scalar per frame             │
│                     │                                            │
│                     ▼                                            │
│   SignalProcessor ──► cumsum → detrend → bandpass → FFT → HR    │
│   BlinkDetector   ──► eye brightness → peaks → blinks/min       │
│                     │                                            │
│                     └──► JSON vitals back to Frontend            │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (offline, 1 lần)
┌─────────────────────────────┴────────────────────────────────────┐
│                          rPPG                                    │
│   35+ .pth weights ──► export_onnx.py ──► .onnx                 │
│   (conda rppg_env)     (PyTorch → ONNX)    └── backend/weights  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack chi tiết

### Môi trường Python

**Conda env `rppg_env` (Python 3.10)** — dùng chung cho rPPG và backend.

> Luôn chạy `conda activate rppg_env` trước khi dùng Python.

### rPPG — Model & Research

| Công cụ | Phiên bản | Vai trò |
|---------|-----------|---------|
| **PyTorch** | 2.5 + CUDA | Load và chạy model `.pth` |
| **ONNX** | 1.21 | Export format trung gian |
| **onnxruntime** | 1.23 | Validate output sau export |
| **OpenCV** | 4.13 | Đọc video, crop face |
| **SciPy** | 1.15 | Bandpass filter, FFT, detrending |
| **NumPy** | 2.0 | Xử lý mảng |
| **Pandas** | 2.0 | Đọc/ghi CSV kết quả |
| **Matplotlib** | 3.10 | Vẽ biểu đồ |
| **Jupyter Lab** | 4.5 | Inference notebooks |
| **yacs** | 0.1.8 | YAML config system |
| **einops** | latest | PhysFormer, RhythmFormer |
| **timm** | latest | PhysMamba |
| **mamba-ssm** | latest | PhysMamba (cần CUDA) |

### Backend — API Server

| Công cụ | Phiên bản | Vai trò |
|---------|-----------|---------|
| **FastAPI** | 0.136 | Async API framework, auto Swagger |
| **uvicorn** | 0.46 | ASGI server |
| **WebSocket** | native FastAPI | Real-time frame/vitals stream |
| **ONNX Runtime** | 1.23 | Inference không cần PyTorch |
| **OpenCV** | 4.13 | Haar Cascade, decode JPEG, resize |
| **SciPy** | 1.15 | Detrend, bandpass, FFT, peak detection |
| **NumPy** | 2.0 | BVP buffer |
| **Pydantic v2** | 2.13 | Schema validation |
| **pydantic-settings** | 2.14 | Load `.env` |
| **python-multipart** | 0.0.28 | Upload video file |

### Frontend — Web UI

| Công cụ | Phiên bản | Vai trò |
|---------|-----------|---------|
| **React** | 18.3 | UI framework |
| **TypeScript** | 5.4 | Type safety |
| **Vite** | 5.4 | Build tool, port 3002 |
| **Tailwind CSS** | 3.4 | Utility-first styling, dark mode |
| **React Router** | v6 | Routing (Home / Upload) |
| **Zustand** | 4.5 | State management |
| **Recharts** | 2.12 | BVP signal real-time chart |
| **Axios** | 1.7 | HTTP (video upload) |
| **WebSocket API** | native | Real-time frames → vitals |
| **Lucide React** | 0.378 | Icons |

---

## Các model rPPG được hỗ trợ

| Nhóm | Model | Kiến trúc | Input shape | Đặc điểm |
|------|-------|-----------|-------------|---------|
| A | **DeepPhys** | CNN 2-nhánh | (N, 6, 72, 72) | Nhanh, dùng mặc định |
| A | **TS-CAN** | Temporal Shift + Attention | (N, 6, 72, 72) | Tốt hơn DeepPhys |
| B | **EfficientPhys** | EfficientNet | (N, 3, 72, 72) | Cân bằng |
| C | **PhysNet** | 3D CNN | (N, 3, 72, 72, D) | Xử lý temporal |
| D | **PhysFormer** | ViT + TDC | (N, 3, 128, 128, D) | Chính xác cao |
| E | **PhysMamba** | Mamba (SSM) | (N, 3, 128, 128, D) | Mới nhất |
| F | **iBVPNet** | 3D CNN | (N, 3, 72, 72, D) | Raw frames |
| F | **FactorizePhys** | NMF + 3D CNN | (N, 3, 72, 72, D) | Matrix factorization |
| G | **RhythmFormer** | Transformer | (N, D, 3, 128, 128) | NDCHW format |
| — | **BigSmall** | Dual-res CNN | Big(144²)+Small(9²) | BVP + nhịp thở |

---

## Các phương pháp Unsupervised

Không cần weights, tính từ RGB skin pixels:

| Method | Mô tả |
|--------|-------|
| **POS** | Plane Orthogonal to Skin |
| **CHROM** | Chromatic decomposition |
| **ICA** | Independent Component Analysis |
| **GREEN** | Kênh xanh lá đơn giản |
| **LGI** | Local Group Invariance |
| **PBV** | Pulse Blood Volume |
| **OMIT** | Orthogonal Matrix in Time |

---

## Signal Processing Pipeline

```
BVP raw values (per frame, từ ONNX inference)
        │
        ▼
  np.cumsum()          ← khôi phục dạng sóng từ difference signal
        │
        ▼
  detrend()            ← Tarvainen smoothness-priors, λ=100
        │
        ▼
  bandpass()           ← Butterworth order 2
        │              HR range: 0.75–2.5 Hz (45–150 BPM)
        ▼
  FFT + periodogram()  ← tìm tần số đỉnh
        │
        ▼
  peak_hz × 60         ← Hz → BPM
```

---

## WebSocket Data Flow

```
Browser                             FastAPI Backend
   │                                      │
   │──{"type":"frame","data":"..."}──────►│
   │                                      │ cv2.imdecode
   │                                      │ Haar Cascade detect face
   │◄──{"type":"face","bbox":[...]}───────│
   │                                      │ crop → 72×72
   │                                      │ DiffNorm + Standardize
   │                                      │ ONNX inference → BVP scalar
   │                                      │ push to deque(180)
   │                                      │
   │   (lặp ~180 frames / 12s @ 15fps)   │
   │                                      │ cumsum → detrend → bandpass → FFT
   │◄──{"type":"vitals","hr":72.5}────────│
   │                                      │ deque.clear() → tích lũy tiếp
```

**Lưu ý:** Webcam gửi 15fps, buffer 180 frames → kết quả sau ~12 giây.

---

## Preprocessing theo nhóm model

| Nhóm | Data Type | Channels | Resolution | Chunk | Format |
|------|-----------|----------|------------|-------|--------|
| A | DiffNorm + Std | 6 | 72×72 | 180 | NDCHW |
| B | Standardized | 3 | 72×72 | 180 | NDCHW |
| C | DiffNormalized | 3 | 72×72 | 128 | NCDHW |
| D | DiffNormalized | 3 | 128×128 | 160 | NCDHW |
| E | DiffNormalized | 3 | 128×128 | 128 | NCDHW |
| F | Raw (no norm) | 3 | 72×72 | 160 | NCDHW |
| G | Standardized | 3 | 128×128 | 160 | NDCHW |
| BigSmall | Std + DiffNorm | 3+3 | 144²+9² | 3 | NDCHW |

---

## Datasets được hỗ trợ

| Dataset | Subjects | FPS | Đặc điểm |
|---------|----------|-----|----------|
| **PURE** | 10 | 30 | Chuyển động đầu nhẹ/vừa |
| **UBFC-rPPG** | 42 | 30 | Indoor, ánh sáng ổn định |
| **SCAMPS** | 2800 | 30 | Synthetic |
| **MMPD** | 33 | 30 | Mobile, đa điều kiện |
| **BP4D+** | 140 | 25 | Cảm xúc, đa điều kiện |
| **UBFC-PHYS** | 56 | 35 | Stress response |
| **iBVP** | 31 | 30 | Near-infrared |
