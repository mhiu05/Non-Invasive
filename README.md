# Non-Invasive Health Analysis

Hệ thống đo sinh trắc học không xâm lấn từ video khuôn mặt — nhịp tim, tốc độ chớp mắt, tín hiệu BVP.

```
webcam / video  →  face detection (OpenCV Haar Cascade)
               →  rPPG inference (ONNX)
               →  signal processing (FFT)
               →  Heart Rate, Blink Rate, SNR
```

![Demo](figures/demo.png)

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|-----------|-----------|
| Python | 3.10 (conda env `rppg_env`) |
| Node.js | 18+ |
| CUDA | 11.8+ (tùy chọn) |
| RAM | 8 GB+ |
| Webcam | Bất kỳ (720p khuyến nghị) |

---

## Cấu trúc dự án

```
Non-Invasive/
├── rPPG/               # Model research & export
│   ├── models/         # Kiến trúc 10 model
│   ├── notebooks/      # Jupyter inference notebooks
│   ├── export/         # Script convert .pth → .onnx
│   ├── evaluation/     # Metrics: MAE, RMSE, SNR, Pearson
│   ├── weights/        # 35+ pretrained .pth files
│   └── requirements.txt
├── backend/            # FastAPI server (port 8001)
│   ├── app/
│   │   ├── services/   # face_detector, rppg_engine, signal_processor, blink_detector
│   │   └── api/        # routes + websocket
│   ├── weights/        # PURE_DeepPhys.onnx + model_config.json
│   ├── .env
│   └── requirements.txt
├── frontend/           # React web app (port 3002)
│   ├── src/
│   │   ├── pages/      # Home (live), Upload (offline)
│   │   ├── components/ # VitalSignCard, BVPChart, FaceOverlay
│   │   └── hooks/      # useWebSocket, useWebcam
│   └── package.json
├── figures/            # Ảnh demo và benchmark
├── doc/
│   ├── PLAN.md
│   └── ARCHITECTURE.md
└── README.md
```

---

## Cài đặt môi trường (1 lần)

Tất cả package Python được cài trong môi trường conda `rppg_env`:

```bash
conda activate rppg_env
pip install -r rPPG/requirements.txt
pip install -r backend/requirements.txt
```

---

## Chạy hệ thống

### Bước 1 — Export model ONNX (chạy 1 lần)

```bash
conda activate rppg_env
cd rPPG

python export/export_onnx.py \
  --model   DeepPhys \
  --weights weights/PURE_DeepPhys.pth \
  --output  weights/PURE_DeepPhys.onnx \
  --validate

cp weights/PURE_DeepPhys.onnx ../backend/weights/PURE_DeepPhys.onnx
```

Kết quả thành công:
```
[validate] Max diff PyTorch vs ONNX: 2.38e-07  [PASS]
```

> Bước này đã hoàn thành — `backend/weights/PURE_DeepPhys.onnx` đã có sẵn.

---

### Bước 2 — Chạy Backend

```bash
conda activate rppg_env
cd backend

python -m uvicorn app.main:app --reload --port 8001
```

Startup thành công:
```
INFO  app.core.lifespan — Loading ONNX model from: weights/PURE_DeepPhys.onnx
INFO  app.services.rppg_engine — ONNX model loaded | img_size=72 | buffer=180
INFO  app.core.lifespan — Startup complete.
INFO  Application startup complete.
```

Kiểm tra:
- `http://localhost:8001/health` → `{"status":"ok","model_loaded":true}`
- `http://localhost:8001/docs` → Swagger UI

---

### Bước 3 — Chạy Frontend

```bash
cd frontend
npm install     # chỉ lần đầu
npm run dev
```

Mở trình duyệt: `http://localhost:3002`

---

### Chạy cả hai cùng lúc

Mở **2 terminal**:

**Terminal 1 — Backend:**
```bash
conda activate rppg_env && cd backend
python -m uvicorn app.main:app --reload --port 8001
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
```

---

## Hướng dẫn sử dụng

### Live Analysis (webcam)

1. Mở `http://localhost:3002`
2. Click **Start Camera** — trình duyệt hỏi quyền camera
3. Nhìn thẳng vào webcam, đủ ánh sáng
4. Sau ~12 giây (180 frames @ 15fps), kết quả xuất hiện:
   - **Heart Rate** — nhịp tim (BPM)
   - **Blink Rate** — tốc độ chớp mắt (lần/phút)
   - **Signal SNR** — chất lượng tín hiệu (dB), > 5 dB là tốt
   - **BVP chart** — dạng sóng mạch máu real-time
5. Click **Stop** để dừng

> Ngồi thẳng, không cử động đầu. Ánh sáng từ phía trước, không ngược sáng.

### Upload video offline

1. Chuyển sang tab **Upload**
2. Kéo thả file video (MP4, AVI, MOV) hoặc click chọn file
3. Chờ xử lý → kết quả hiện ngay
4. Click **Export CSV** để tải dữ liệu BVP

---

## Kết quả Benchmark

Đánh giá trên 10 subjects với 3 điều kiện thực tế.  
Metrics: MAE (bpm), RMSE (bpm), MAPE (%), Pearson, SNR (dB) — **thấp hơn là tốt hơn** (trừ Pearson và SNR).

### Điều kiện bình thường

![Benchmark Normal](figures/benchmark_normal.png)

> **Top 1:** UBFC-rPPG_FactorizePhys — MAE **0.04 bpm**, Pearson **1.00**

### Điều kiện chuyển động đầu

![Benchmark Head Motion](figures/benchmark_headmotion.png)

> **Top 1:** PURE_FactorizePhys — MAE **0.83 bpm**, Pearson **1.00**

### Điều kiện nói chuyện

![Benchmark Talk](figures/benchmark_talk.png)

> **Top 1:** UBFC-rPPG_EfficientPhys — MAE **1.67 bpm**, Pearson **0.97**

### Nhận xét

| Điều kiện | Model tốt nhất | MAE tốt nhất |
|-----------|---------------|-------------|
| Bình thường | FactorizePhys | ~0.04 bpm |
| Chuyển động đầu | FactorizePhys | ~0.83 bpm |
| Nói chuyện | EfficientPhys | ~1.67 bpm |

**FactorizePhys** cho kết quả tốt nhất trong điều kiện tĩnh và chuyển động đầu.  
**EfficientPhys** ổn định hơn khi có hoạt động nói chuyện.

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

### `POST /video/upload`

**Request:** `multipart/form-data`, field `file`

**Response:**
```json
{
  "filename": "test.mp4",
  "total_frames": 900,
  "duration_sec": 30.0,
  "heart_rate": 72.5,
  "blink_rate": 14.1,
  "snr_db": 8.3,
  "bvp_signal": [0.012, -0.005, ...]
}
```

### `WS /ws/stream`

```json
// Client → Server
{ "type": "frame", "data": "<base64 JPEG>" }
{ "type": "reset" }

// Server → Client
{ "type": "face", "detected": true, "bbox": [x, y, w, h] }
{ "type": "vitals", "heart_rate": 72.5, "blink_rate": 14.1,
  "snr_db": 8.3, "bvp_window": [...] }
```

---

## Chạy notebooks đánh giá (rPPG)

```bash
conda activate rppg_env
cd rPPG
jupyter lab
```

Mở `notebooks/groupA_inference.ipynb` → chạy với dataset của bạn.  
Kết quả lưu vào `rPPG/results/` dạng CSV.

---

## Xử lý lỗi thường gặp

### `ModuleNotFoundError: No module named 'cv2'`
→ Chưa activate đúng môi trường. Chạy `conda activate rppg_env` trước.

### `Address already in use` (port 8001 hoặc 3002)
→ Đổi port: `--port 8002` hoặc Vite tự chọn port trống tiếp theo.

### WebSocket disconnect liên tục
→ Kiểm tra backend log. Nếu thấy lỗi `int32 is not JSON serializable` → cập nhật `face_detector.py` (đã fix).

### Webcam bị từ chối trên trình duyệt
→ Chrome: `chrome://flags` → "Insecure origins" → thêm `http://localhost:3002`.  
→ Firefox: cho phép trong popup.

### Kết quả HR không ổn định (SNR thấp)
→ Ánh sáng đủ sáng, ngồi yên trong 12 giây đầu.  
→ SNR < 3 dB: tín hiệu nhiễu — thử FactorizePhys hoặc PhysFormer.

### Face không được phát hiện
→ Khoảng cách: 40–80 cm, nhìn thẳng, ánh sáng từ phía trước.

---

## Thay đổi model

Mặc định dùng **DeepPhys** (nhanh nhất). Dựa trên benchmark, để dùng model chính xác hơn:

```bash
cd rPPG
python export/export_onnx.py \
  --model   FactorizePhys \
  --weights weights/PURE_FactorizePhys_FSAM_Res.pth \
  --output  weights/PURE_FactorizePhys.onnx \
  --img-size 72 --chunk 160 --validate

cp weights/PURE_FactorizePhys.onnx ../backend/weights/
```

Cập nhật `backend/.env`:
```env
MODEL_PATH=weights/PURE_FactorizePhys.onnx
```

Cập nhật `backend/weights/model_config.json`:
```json
{ "model": "FactorizePhys", "img_size": 72, "chunk": 160, "norm_type": "Raw", "fps": 30 }
```

> Xem `rPPG/notebooks/model_groups.md` để biết thông số của từng model.
