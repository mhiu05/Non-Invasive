# Non-Invasive Health Analysis

Hệ thống đo sinh trắc học không xâm lấn từ video khuôn mặt — nhịp tim, tốc độ chớp mắt, tín hiệu BVP, kèm chatbot AI hỗ trợ sức khỏe.

```
webcam / video  →  face detection (MediaPipe)
               →  rPPG inference (ONNX hoặc PyTorch fallback)
               →  signal processing (FFT)
               →  Heart Rate, Blink Rate, SNR, HRV
               →  AI Chatbot (RAG + Gemini) hỗ trợ tư vấn
```

![Architecture](figures/architecture.png)

![Demo](figures/demo.png)

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|-----------|-----------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA (rPPG) | 11.8+ (tùy chọn) |
| CPU | 
| RAM | 8 GB+ |
| Webcam | Bất kỳ (720p khuyến nghị) |

---

## Cấu trúc dự án

```
Non-Invasive/
├── rPPG/                    # Model research & export
│   ├── models/              # 10 kiến trúc: DeepPhys, TSCAN, PhysNet, EfficientPhys,
│   │                        #   PhysFormer, PhysMamba, RhythmFormer, BigSmall, iBVPNet, 
│   ├── export/              # Script convert .pth → .onnx
│   ├── evaluation/          # Metrics: MAE, RMSE, SNR, Pearson
│   ├── weights/             # 36 pretrained .pth + 34 .onnx đã convert
│   └── notebooks_inference/ # Jupyter inference notebooks
│
├── backend/                 # FastAPI server (port 8001)
│   ├── app/
│   │   ├── core/            # config, lifespan
│   │   ├── services/        # rppg_engine, face_detector, preprocessor,
│   │   │                    #   signal_processor, blink_detector, history_store
│   │   ├── chatbot/         # RAG chatbot: engine, loader, vectorstore,
│   │   │                    #   feedback_store, ingest, auto_update
│   │   ├── documents/       # Tài liệu cho chatbot học (PDF, .md, .txt)
│   │   ├── schemas/         # Pydantic models
│   │   └── api/             # routes + websocket
│   ├── scripts/             # Utility scripts (build_embeddings.py)
│   ├── vectorstore/         # FAISS index (generated, gitignored)
│   ├── weights/             # 34 .onnx models + model_config.json
│   ├── tests/               # Unit tests
│   └── .env
│
├── frontend/                # React + JavaScript web app (port 3002)
│   ├── src/
│   │   ├── pages/           # Home (landing), Live (realtime webcam), Upload (offline video)
│   │   ├── components/      # VitalSignCard, BVPChart, FaceOverlay, ChatBot
│   │   ├── hooks/           # useWebSocket, useWebcam
│   │   ├── lib/             # api, chatApi, utils, vitals
│   │   └── store/           # vitalsStore (Zustand)
│   └── package.json
│
├── docs/                    # Tài liệu hướng dẫn dự án
│   ├── ARCHITECTURE.md      # Kiến trúc hệ thống
│   ├── backend_plan.md      # Kế hoạch cải tiến backend
│   ├── frontend_plan.md     # Kế hoạch cải tiến frontend
│   ├── rPPG_plan.md         # Kế hoạch cải tiến rPPG
│   ├── chatbot_plan.md      # Kế hoạch mở rộng chatbot
│   ├── deploy.md            # Hướng dẫn deploy
│   ├── details.md           # Chi tiết từng file
│
├── figures/                 # Ảnh demo và benchmark
```

## 🧠 Các công nghệ AI/ML & Web sử dụng

Dự án áp dụng nhiều kỹ thuật tiên tiến trong Deep Learning, Computer Vision và Generative AI:

### 1. Trí tuệ nhân tạo (AI/DL) trong Computer Vision
- **MediaPipe Face Mesh (Google):** Trích xuất tọa độ khuôn mặt mật độ cao (468 landmarks) theo thời gian thực để xác định chính xác các Vùng quan tâm (Region of Interest - ROI) (như trán, má) nơi có nhiều vi mạch máu biểu bì.
- **Deep Learning Models (rPPG):** Sử dụng các kiến trúc mạng nơ-ron chuyên dụng để trích xuất tín hiệu khối lượng máu (Blood Volume Pulse - BVP) từ chuỗi khung hình:
  - **CNN-based:** DeepPhys (Spatial-Temporal 2D CNN), PhysNet (3D CNN).
  - **Transformer-based (ViT):** PhysFormer, RhythmFormer (sử dụng self-attention để mô hình hóa chuỗi thời gian xa).
  - **Efficient Networks:** TSCAN, EfficientPhys.
- **Inference Engine:** Tối ưu hóa triển khai mô hình với **ONNX Runtime** giúp giảm độ trễ, tăng FPS khi chạy trên CPU/Edge devices. Hỗ trợ **PyTorch fallback** cho các mô hình chưa thể convert (như PhysMamba).

### 2. Xử lý tín hiệu số (Digital Signal Processing)
- **Fast Fourier Transform (FFT):** Phân tích tín hiệu BVP từ miền thời gian sang miền tần số để tìm đỉnh tần số vượt trội (chính là Nhịp tim).
- **Butterworth Bandpass Filter:** Lọc nhiễu tín hiệu động học dựa trên dải nhịp tim chuẩn theo từng nhóm tuổi (ví dụ: trẻ sơ sinh 1.6-3.3 Hz, người lớn 0.7-3.0 Hz).
- **Peak Detection (SciPy):** Phát hiện các đỉnh tâm thu (Systolic peaks) trên sóng BVP để tính toán các chỉ số Biến thiên nhịp tim (Heart Rate Variability - HRV) như RMSSD, SDNN, pNN50.

### 3. Generative AI & NLP (RAG Pipeline)
- **Large Language Model (LLM):** Sử dụng **Google Gemini 2.5 Flash** để tư vấn sức khỏe dựa trên kết quả rPPG.
- **Retrieval-Augmented Generation (Advanced RAG):**
  - **Embeddings:** `Google Generative AI Embeddings` để chuyển đổi tài liệu y khoa thành vector.
  - **Vector Database:** `FAISS` (Facebook AI Similarity Search) lưu trữ vector và `BM25` lưu trữ từ khóa.
  - **Hybrid Search:** Kết hợp tìm kiếm theo ngữ nghĩa (Dense Retrieval qua FAISS) và tìm kiếm theo từ khóa (Sparse Retrieval qua BM25).
  - **Query Rewriting:** Sử dụng `MultiQueryRetriever` (LangChain) để tự động viết lại và mở rộng câu hỏi của người dùng.
  - **Re-ranking:** Tích hợp mô hình Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) để chấm điểm và sắp xếp lại văn bản truy xuất nhằm chọn lọc ra kết quả chính xác nhất.
  - **Orchestration:** `LangChain` kết nối toàn bộ pipeline, tạo ra chuỗi (chain) hỏi đáp có độ chính xác cao, hạn chế tối đa hallucination.

### 4. Web Development (Fullstack)
- **Backend:** `FastAPI` (Python) siêu tốc, hỗ trợ xử lý luồng `WebSocket` bất đồng bộ (`asyncio`) và chạy các mô hình AI nặng trên `ThreadPoolExecutor` để không block event loop. Lưu trữ lịch sử với `SQLite`.
- **Frontend:** `React 18` + `JavaScript`, build bằng `Vite`. Quản lý state bằng `Zustand`. Giao diện hiện đại với `Vanilla CSS`, hiển thị sóng realtime với `Recharts`.

---

## 🚀 Chạy hệ thống

### Bước 0 — Build chatbot vectorstore (1 lần)

```bash
cd backend
python scripts/build_embeddings.py
```

### Bước 1 — Chạy Backend (terminal 1)

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8001
```

### Bước 2 — Chạy Frontend (terminal 2)

```bash
cd frontend
npm install     # chỉ lần đầu
npm run dev
```

Mở trình duyệt: `http://localhost:3002`

> Lưu ý: frontend dev server chạy trên cổng `3002` theo cấu hình `frontend/vite.config.ts`.

---

## 📖 Hướng dẫn sử dụng

### 1. Landing Page (Trang chủ)
- Truy cập `http://localhost:3002/`.
- Xem giải thích các chỉ số sức khỏe, hướng dẫn các bước thiết lập ánh sáng/camera và chọn chức năng đo.

### 2. Live Analysis (Đo qua Webcam)

1. Chuyển sang tab **Live** (`/live`).
2. Click **Start Camera** — trình duyệt hỏi quyền camera.
3. Nhìn thẳng vào webcam, đủ ánh sáng.
4. Sau ~6 giây (181 frames @ 30fps), kết quả xuất hiện:
   - **Heart Rate** — nhịp tim (BPM)
   - **Blink Rate** — tốc độ chớp mắt (lần/phút)
   - **Signal SNR** — chất lượng tín hiệu (dB), > 2 dB là tốt
   - **BVP chart** — dạng sóng mạch máu real-time
5. Click **Stop** để dừng. Phiên đo (nếu > 5s) sẽ tự động lưu vào Lịch sử.
6. Nhấp **Xem lịch sử** ở góc trên để đối chiếu.

> Ngồi thẳng, không cử động đầu. Ánh sáng từ phía trước, không ngược sáng.

### 3. Upload video offline (Async Processing)

1. Chuyển sang tab **Upload** (`/upload`).
2. Kéo thả file video (MP4, AVI, MOV) hoặc click chọn file.
3. Hệ thống trả về `job_id` và tự động polling lấy trạng thái. Có thể đóng trình duyệt và mở lại lịch sử xem sau.
4. Quản lý lịch sử phân tích với tính năng lọc (Filter) theo thời gian/loại.

### 4. AI Chatbot

- Click nút chat ở góc phải màn hình (hiện trên mọi trang).
- Hỏi về kết quả đo sức khỏe, kiến trúc hệ thống, hoặc ý nghĩa các chỉ số.
- Chatbot sử dụng RAG từ tài liệu trong `backend/app/documents/`.

> ⚠️ Thông tin sức khỏe từ chatbot chỉ mang tính tham khảo, không thay thế bác sĩ.

---

## Kết quả Benchmark

Đánh giá trên 10 subjects với 3 điều kiện thực tế.  
Metrics: MAE (bpm), RMSE (bpm), MAPE (%), Pearson, SNR (dB).

### Điều kiện bình thường
![Benchmark Normal](figures/benchmark_normal.png)
> **Top 1:** UBFC-rPPG_FactorizePhys — MAE **0.04 bpm**, Pearson **1.00**

### Điều kiện chuyển động đầu
![Benchmark Head Motion](figures/benchmark_headmotion.png)
> **Top 1:** PURE_FactorizePhys — MAE **0.83 bpm**, Pearson **1.00**

### Điều kiện nói chuyện
![Benchmark Talk](figures/benchmark_talk.png)
> **Top 1:** UBFC-rPPG_EfficientPhys — MAE **1.67 bpm**, Pearson **0.97**

| Điều kiện | Model tốt nhất | MAE tốt nhất |
|-----------|---------------|-------------|
| Bình thường | FactorizePhys | ~0.04 bpm |
| Chuyển động đầu | FactorizePhys | ~0.83 bpm |
| Nói chuyện | EfficientPhys | ~1.67 bpm |


## Thay đổi model

### Dùng model đã có trong `backend/weights/`

Backend có sẵn **34 model ONNX**

Ví dụ dùng FactorizePhys:
```env
# backend/.env
MODEL_PATH=weights/UBFC-rPPG_FactorizePhys_FSAM_Res.onnx
```

Cập nhật `backend/weights/model_config.json` cho đúng model:
```json
{ "model": "FactorizePhys", "img_size": 72, "chunk": 181, "norm_type": "DiffNorm", "fps": 30 }
```

### Dùng PyTorch fallback (.pth trực tiếp)

Backend tự động fallback sang PyTorch khi:
- `MODEL_PATH` trỏ tới file `.pth`, hoặc
- File `.onnx` không tồn tại nhưng có file `.pth` tương ứng

```env
# backend/.env — dùng PhysMamba trực tiếp từ .pth
MODEL_PATH=../rPPG/weights/PURE_PhysMamba_DiffNormalized.pth
```

> **Lưu ý:** PyTorch fallback dành cho DeepPhys-style models (input 2 frame). 
> PhysMamba yêu cầu `selective_scan_cuda` CUDA kernel — chưa tương thích với CUDA 13.0.

### Export model mới sang ONNX

```bash
cd rPPG
python export/export_onnx.py \
  --model   DeepPhys \
  --weights weights/PURE_DeepPhys.pth \
  --output  weights/PURE_DeepPhys.onnx \
  --validate

cp weights/PURE_DeepPhys.onnx ../backend/weights/
```

Batch export toàn bộ:
```bash
cd rPPG
bash export_batch.sh
```

---

## Danh sách model có sẵn

| Dataset | DeepPhys | EfficientPhys | TSCAN | PhysNet | PhysFormer | RhythmFormer | FactorizePhys | iBVPNet | BigSmall | PhysMamba |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PURE    | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ⚠️ |
| SCAMPS  | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — | — |
| UBFC-rPPG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ⚠️ |
| BP4D PseudoLabel | ✅ | ✅ | ✅ | ✅ | — | — | — | — | ✅✅✅ | — |
| iBVP    | — | ✅ | — | — | — | — | ✅ | — | — | — |
| MA-UBFC | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — |

✅ = ONNX sẵn trong `backend/weights/`  
⚠️ = Chỉ có `.pth` (PyTorch fallback)

---

## Lưu ý

### Kết quả HR không ổn định (SNR thấp)
→ Ánh sáng đủ sáng, ngồi yên trong 12 giây đầu.  
→ SNR < 3 dB: thử FactorizePhys hoặc PhysFormer.

### Face không được phát hiện
→ Khoảng cách: 40–80 cm, nhìn thẳng, ánh sáng từ phía trước.

### Chatbot không phản hồi
→ Kiểm tra `GEMINI_API_KEY` trong `backend/.env`.  
→ Đảm bảo đã chạy `python scripts/build_embeddings.py` trong thư mục `backend/`.
