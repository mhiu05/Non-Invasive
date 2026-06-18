# Chi tiết cấu trúc dự án

Tài liệu mô tả chi tiết vai trò và nội dung của từng file, thư mục trong dự án **Non-Invasive Health Analysis System**.

---

## Cấu trúc thư mục gốc

```
Non-Invasive/
├── .docker/            # Dockerfile cho từng dịch vụ
├── .docs/              # Tài liệu thiết kế nội bộ
├── .figures/           # Hình ảnh minh họa cho README
├── .k8s/               # Cấu hình Kubernetes (Helm charts)
├── backend/            # Backend API (FastAPI + Python)
├── frontend/           # Frontend SPA (React + Vite)
├── hf_space/           # Bản deploy lên Hugging Face Spaces
├── rPPG/               # Nghiên cứu, huấn luyện và đánh giá mô hình rPPG
├── .gitignore          # Quy tắc loại trừ file khỏi Git
├── .gitattributes      # Cấu hình Git LFS cho file nhị phân lớn
├── docker-compose.yml  # Orchestration cho môi trường phát triển
├── docker-compose.prod.yml  # Orchestration cho môi trường production
├── README.md           # Tài liệu tổng quan dự án
├── SECURITY.md         # Chính sách bảo mật
└── detail.md           # (File này) Mô tả chi tiết cấu trúc dự án
```

---

## `.docker/` — Dockerfile cho từng dịch vụ

Chứa Dockerfile tách riêng cho mỗi service trong Docker Compose, giúp build image độc lập.

| File | Mô tả |
|---|---|
| `backend/Dockerfile` | Build image cho FastAPI backend. Cài đặt Python dependencies, copy mã nguồn `backend/app/`, expose port 8001. |
| `worker/Dockerfile` | Build image cho Celery worker. Dùng chung codebase với backend nhưng chạy lệnh `celery worker` thay vì `uvicorn`. |
| `frontend/Dockerfile` | Build image cho frontend. Gồm 2 stage: (1) Build React app bằng Vite, (2) Serve static files bằng Nginx. |
| `frontend/nginx.conf` | Cấu hình Nginx cho frontend: proxy API requests, serve SPA với fallback về `index.html`. |

---

## `.docs/` — Tài liệu thiết kế nội bộ

Tài liệu thiết kế kỹ thuật được tạo trong quá trình phát triển.

| File | Mô tả |
|---|---|
| `ARCHITECTURE.md` | Mô tả kiến trúc hệ thống tổng quan, các tầng, luồng dữ liệu. |
| `PLAN.md` | Kế hoạch phát triển chi tiết, các milestone và task breakdown. |
| `details.md` | Ghi chú kỹ thuật chi tiết về các quyết định thiết kế. |

---

## `.figures/` — Hình ảnh minh họa

Chứa các hình ảnh sử dụng trong `README.md`.

| File | Mô tả |
|---|---|
| `demo.png` | Ảnh chụp giao diện hệ thống đang hoạt động. |
| `architecture.png` | Sơ đồ kiến trúc hệ thống 4 tầng. |
| `benchmark_normal.png` | Kết quả đánh giá MAE trong điều kiện ngồi yên. |
| `benchmark_headmotion.png` | Kết quả đánh giá MAE trong điều kiện chuyển động đầu. |
| `benchmark_talk.png` | Kết quả đánh giá MAE trong điều kiện nói chuyện. |

---

## `.k8s/` — Cấu hình Kubernetes

| Thư mục | Mô tả |
|---|---|
| `charts/rppg-backend/` | Helm chart để triển khai backend + worker trên Kubernetes cluster. |

---

## `docker-compose.yml` — Orchestration (Development)

Định nghĩa 4 service chạy đồng thời:

| Service | Image/Build | Port | Vai trò |
|---|---|---|---|
| `redis` | `redis:7-alpine` | 6379 | Message broker cho Celery, lưu task queue |
| `backend` | Build từ `.docker/backend/Dockerfile` | 8001 | FastAPI API server |
| `worker` | Build từ `.docker/worker/Dockerfile` | — | Celery worker xử lý video nền |
| `frontend` | Build từ `.docker/frontend/Dockerfile` | 3002 | Nginx serving React build |

Các service mount code cục bộ (`./backend/app:/app/app`) để hỗ trợ hot-reload khi phát triển.

---

## `backend/` — Backend API

### Cấu trúc tổng quan

```
backend/
├── app/
│   ├── api/                # Định nghĩa API endpoints
│   ├── chatbot/            # Module chatbot RAG
│   ├── core/               # Cấu hình, bảo mật, lifecycle
│   ├── documents/          # Tài liệu y khoa cho RAG
│   ├── schemas/            # Pydantic schemas (request/response models)
│   ├── services/           # Business logic chính
│   ├── worker/             # Celery worker và task definitions
│   ├── __init__.py
│   └── main.py             # Entry point FastAPI application
├── scripts/                # Script tiện ích
├── tests/                  # Unit tests
├── vectorstore/            # Dữ liệu FAISS index (generated)
├── weights/                # Model weights (ONNX)
├── .env                    # Biến môi trường (không commit)
├── .env.example            # Template biến môi trường
├── Dockerfile              # Dockerfile cho deploy trực tiếp (không qua .docker/)
└── requirements.txt        # Python dependencies
```

---

### `backend/app/main.py` — Entry point

Khởi tạo FastAPI application:
- Cấu hình CORS middleware (cho phép cross-origin từ frontend).
- Include tất cả API routers.
- Kết nối lifespan events (khởi tạo model khi start, cleanup khi stop).
- Swagger UI tại `/docs`, ReDoc tại `/redoc`.

---

### `backend/app/core/` — Cấu hình hệ thống

| File | Mô tả |
|---|---|
| `config.py` | **Cấu hình trung tâm** — Load biến môi trường từ `.env` bằng `pydantic-settings`. Định nghĩa class `Settings` chứa tất cả cấu hình: đường dẫn model (`model_path`), device CPU/CUDA (`device`), FPS (`fps=30`), Gemini API key, Redis URL, Supabase credentials, S3 storage. |
| `lifespan.py` | **Quản lý vòng đời ứng dụng** — Dùng `@asynccontextmanager` của FastAPI. Khi startup: khởi tạo `RPPGEngine` (load ONNX model) và `FaceDetector` (load MediaPipe) một lần duy nhất, lưu vào biến module-level để chia sẻ giữa mọi request. Khi shutdown: giải phóng tài nguyên. |
| `security.py` | **Xác thực JWT** — Sử dụng Supabase client để verify access token. Cung cấp hai dependency: `get_current_user` (optional auth) và `get_current_user_required` (bắt buộc auth, raise 401 nếu thiếu). |

---

### `backend/app/api/` — API Endpoints

```
api/
├── router.py           # Router gốc, gom tất cả sub-routers
├── routes/
│   ├── auth.py         # POST /auth/login, POST /auth/register
│   ├── chat.py         # POST /chat — gọi chatbot RAG
│   ├── history.py      # GET /history — lấy lịch sử đo
│   ├── system.py       # GET /health — health check
│   └── video.py        # POST /video/upload, POST /video/upload-async, GET /video/jobs/{id}
└── websocket/
    └── stream.py       # WebSocket /ws/stream — đo real-time
```

#### `routes/video.py` — Xử lý video

Cung cấp 3 endpoints:

| Endpoint | Chức năng |
|---|---|
| `POST /video/upload` | Upload và xử lý **đồng bộ** (synchronous). Nhận file + age, trả kết quả ngay. Phù hợp video ngắn. |
| `POST /video/upload-async` | Upload và xử lý **bất đồng bộ**. Upload video lên S3 Storage, tạo job record trong PostgreSQL, gửi task vào Celery queue. Trả `job_id` ngay lập tức. |
| `GET /video/jobs/{job_id}` | Kiểm tra trạng thái job: `pending` → `processing` → `done`/`failed`. Trả kết quả khi `done`. |

Hàm `_process_video()`: Đọc video bằng OpenCV, lặp qua từng frame, phát hiện khuôn mặt → crop & resize → push vào `SessionState` → thu thập BVP values.

#### `websocket/stream.py` — Đo real-time

Giao thức WebSocket cho phép client gửi liên tục JPEG frames (base64 encoded) và nhận lại kết quả vitals:

**Luồng xử lý mỗi frame:**
1. Decode base64 → JPEG → OpenCV BGR image.
2. Face detection + crop + resize (chạy trong `run_in_executor` để không block event loop).
3. Push crop vào `SessionState.push_frame()` — trả BVP buffer nếu đủ dữ liệu.
4. Nếu có BVP buffer → tính HR, SNR, HRV (cũng chạy trong executor).
5. Gửi kết quả JSON về client.

**Xử lý đặc biệt:**
- Khi mất mặt → reset session ngay lập tức (tránh tính HR từ background).
- Khi session kết thúc (disconnect) và có đủ dữ liệu (≥ 5 giây) → tự động lưu kết quả vào history.

#### `routes/auth.py` — Xác thực

Proxy các request đăng nhập/đăng ký tới Supabase Auth, trả về access token cho frontend.

#### `routes/chat.py` — Chatbot

Nhận câu hỏi từ người dùng, gọi `chatbot.engine.ask()`, trả về câu trả lời + sources + cờ `from_internal_docs`.

#### `routes/history.py` — Lịch sử

Lấy danh sách lịch sử đo theo `user_id`, hỗ trợ phân trang (limit/offset) và lọc theo thời gian, loại đo (realtime/video).

---

### `backend/app/services/` — Business Logic

| File | Mô tả |
|---|---|
| `face_detector.py` | **Phát hiện khuôn mặt** — Class `FaceDetector` wrap MediaPipe Face Mesh. Tính bounding box từ 468 landmarks, mở rộng 1.4x, làm mượt bằng EMA (α=0.7). Miss tolerance = 8 frames. Phương thức `crop_resize()` trả face crop đã resize về kích thước model (72×72). |
| `preprocessor.py` | **Tiền xử lý input** — Hàm `build_chunk_input()` chuyển đổi list frames BGR (uint8) thành tensor ONNX `(1, 3, T, H, W)` float32: BGR→RGB, normalize [0,1], transpose NHWC→NCTHW. Toàn bộ vectorized bằng NumPy. |
| `rppg_engine.py` | **ONNX Inference Engine** — Class `RPPGEngine`: load ONNX session, chạy inference chunk-wise. Class `SessionState`: quản lý per-connection state, frame buffer (deque), background inference thread. Chỉ infer 1 lần mỗi 15 frames (~0.5s) để tránh lag. |
| `signal_processor.py` | **Xử lý tín hiệu** — Pipeline hoàn chỉnh: `detrend()` (Tarvainen Smoothness Priors), `bandpass()` (Butterworth bậc 2), `compute_heart_rate()` (Periodogram FFT → HR + SNR), `compute_hrv()` (peak detection → SDNN, RMSSD, pNN50). Hàm `get_bandpass_by_age()` tra bảng dải tần theo 6 nhóm tuổi. |
| `history_store.py` | **Lưu trữ lịch sử** — CRUD operations trên bảng `history` trong PostgreSQL (Supabase). Sử dụng `psycopg2` trực tiếp. Hỗ trợ upsert (ON CONFLICT), lọc theo user_id/type/thời gian. |
| `storage.py` | **Object Storage (S3)** — Kết nối Supabase Storage (S3-compatible) qua `boto3`. Ba hàm: `upload_video()`, `download_file()`, `delete_file()`. Worker download video từ S3 để xử lý, xóa sau khi xong. |

---

### `backend/app/chatbot/` — Module Chatbot RAG

| File | Mô tả |
|---|---|
| `loader.py` | **Document Loader** — Load tài liệu `.md`, `.txt`, `.pdf` từ `backend/app/documents/` bằng LangChain `DirectoryLoader`. Hàm `text_split()` chia thành chunks (500 ký tự, overlap 50) bằng `RecursiveCharacterTextSplitter`. Hàm `get_embeddings()` trả instance `HuggingFaceEmbeddings(all-MiniLM-L6-v2)`. |
| `vectorstore.py` | **Vector Database** — Quản lý FAISS index và BM25 index. Hàm `build_faiss()` xây index từ chunks. `load_faiss()` / `load_bm25()` load index đã build. `update_faiss()` hỗ trợ cập nhật incremental. Dữ liệu lưu tại `backend/vectorstore/faiss_index/`. |
| `engine.py` | **RAG Chain Engine** — Lazy-load RAG chain khi có request đầu tiên. Pipeline: EnsembleRetriever (FAISS 70% + BM25 30%) → MultiQueryRetriever (query rewriting) → CrossEncoderReranker (ms-marco-MiniLM, top 5) → StuffDocumentsChain → Google Gemini. Fallback logic: nếu không tìm thấy tài liệu → hỏi Gemini trực tiếp. Xử lý quota exceeded → retry với model fallback. |
| `feedback_store.py` | **Lưu feedback** — Lưu trữ phản hồi người dùng về chất lượng câu trả lời chatbot. |

---

### `backend/app/documents/` — Tài liệu y khoa cho RAG

Chứa tài liệu nguồn mà chatbot sử dụng để trả lời câu hỏi:

| File | Mô tả |
|---|---|
| `Giao-Trinh-SDH-Tim-Mach-Hoc-DHYD-Hue.pdf` | Giáo trình Tim mạch học — Đại học Y Dược Huế |
| `Heart_rate_variability.pdf` | Tài liệu về biến thiên nhịp tim (HRV) |
| `Medical_book.pdf` | Sách y khoa tổng hợp |

---

### `backend/app/schemas/` — Pydantic Schemas

| File | Mô tả |
|---|---|
| `vitals.py` | Định nghĩa `VideoResultResponse` — schema response cho kết quả phân tích video (heart_rate, snr_db, hrv_ms, sdnn_ms, rmssd_ms, pnn50, peak_count, bvp_signal, age_group, bandpass...). |
| `chat.py` | Schema cho request/response chatbot (question, answer, sources, from_internal_docs). |
| `history.py` | Schema cho lịch sử đo (list response, filter params). |
| `auth.py` | Schema đăng nhập (email, password). |

---

### `backend/app/worker/` — Celery Worker

| File | Mô tả |
|---|---|
| `celery_app.py` | **Cấu hình Celery** — Khởi tạo Celery app kết nối Redis (broker + backend). Cấu hình: `task_acks_late=True` (xác nhận sau khi xong), `worker_prefetch_multiplier=1` (1 task/lần), task routing `video.*` → queue riêng. |
| `tasks/video.py` | **Task xử lý video** — Signal `worker_process_init` khởi tạo RPPGEngine + FaceDetector một lần khi worker start. Task `process_video`: download video từ S3 → chạy rPPG pipeline → tính HR/HRV → lưu kết quả vào PostgreSQL → xóa file khỏi S3. Auto-retry 3 lần nếu thất bại. |

---

### `backend/weights/` — Model Weights

| File | Mô tả |
|---|---|
| `UBFC-rPPG_FactorizePhys_FSAM_Res.onnx` | Mô hình FactorizePhys đã export sang ONNX (~250 KB). Huấn luyện trên dataset UBFC-rPPG. |
| `model_config.json` | Cấu hình model: `img_size` (72), `chunk` (180), metadata inference. |

---

### `backend/vectorstore/faiss_index/` — Vector Database (Generated)

Thư mục chứa dữ liệu FAISS index và BM25 index đã build từ `scripts/build_embeddings.py`. Được tạo tự động, không cần commit vào Git.

---

### `backend/scripts/build_embeddings.py` — Script xây dựng Vector DB

Script chạy offline để:
1. Load tất cả tài liệu từ `backend/app/documents/`.
2. Chia nhỏ thành chunks.
3. Mã hóa embedding bằng `all-MiniLM-L6-v2`.
4. Build và lưu FAISS + BM25 index vào `backend/vectorstore/faiss_index/`.

Sử dụng: `cd backend && python scripts/build_embeddings.py [--force]`

---

### `backend/tests/` — Unit Tests

| File | Mô tả |
|---|---|
| `test_signal_processor.py` | Test các hàm xử lý tín hiệu: detrend, bandpass, compute_heart_rate, compute_hrv. |

---

## `frontend/` — Frontend SPA

### Cấu trúc tổng quan

```
frontend/
├── src/
│   ├── App/                # Root component
│   ├── components/         # Reusable UI components
│   ├── features/           # Feature modules
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utilities và API clients
│   ├── pages/              # Page components (routes)
│   ├── store/              # State management (Zustand)
│   ├── index.css           # Global CSS styles
│   └── main.jsx            # Entry point React app
├── public/
│   └── heart.svg           # Favicon / logo
├── index.html              # HTML template
├── package.json            # Dependencies và scripts
├── vite.config.js          # Vite configuration
├── vercel.json             # Vercel deployment config
├── .env                    # Biến môi trường (không commit)
└── .env.example            # Template biến môi trường
```

---

### `frontend/src/main.jsx` — Entry Point

Mount React app vào DOM, wrap với `BrowserRouter` cho routing.

### `frontend/src/App/`

| File | Mô tả |
|---|---|
| `App.jsx` | **Root component** — Định nghĩa routing (React Router v6): Home, Live, Upload, Login, Register, Profile. Wrap với `AuthProvider` và `ProtectedRoute`. Include `Header` và `ChatBot` widget. |
| `App.css` | CSS riêng cho root layout. |

---

### `frontend/src/pages/` — Trang chính

| Thư mục | Route | Mô tả |
|---|---|---|
| `Home/` | `/` | Dashboard tổng quan — giới thiệu tính năng, điều hướng tới Live/Upload. |
| `Live/` | `/live` | **Đo real-time** — Kích hoạt webcam, kết nối WebSocket, hiển thị video với face overlay, biểu đồ BVP, và các thẻ vital signs (HR, SNR, HRV). |
| `Upload/` | `/upload` | **Upload video** — Form upload file (MP4/AVI) + nhập tuổi. Hiển thị progress và kết quả phân tích. |
| `Login/` | `/login` | Form đăng nhập (email + password) qua Supabase Auth. |
| `Register/` | `/register` | Form đăng ký tài khoản mới. |
| `Profile/` | `/profile` | Hiển thị lịch sử đo, thống kê cá nhân. |

---

### `frontend/src/components/` — Components tái sử dụng

| Thư mục | Mô tả |
|---|---|
| `BVPChart/` | **Biểu đồ BVP waveform** — Dùng `recharts` vẽ đồ thị tín hiệu BVP real-time. Cập nhật mượt mà theo dữ liệu từ WebSocket. |
| `ChatBot/` | **Widget chatbot AI** — Cửa sổ chat ở góc dưới phải. Gửi câu hỏi qua `/chat` API, hiển thị câu trả lời + nguồn tài liệu. Phân biệt trả lời từ tài liệu nội bộ vs kiến thức tổng quát. |
| `FaceOverlay/` | **Overlay khuôn mặt** — Vẽ bounding box lên video webcam dựa trên tọa độ bbox nhận từ WebSocket. |
| `Header/` | **Thanh điều hướng** — Navigation bar trên cùng, hiển thị menu và trạng thái đăng nhập. |
| `HistoryView/` | **Xem lịch sử** — Bảng danh sách kết quả đo trước đó, hỗ trợ phân trang và lọc. |
| `ProtectedRoute/` | **Route bảo vệ** — HOC kiểm tra authentication trước khi cho truy cập trang. Redirect về Login nếu chưa đăng nhập. |
| `SnrBadge/` | **Badge chất lượng tín hiệu** — Hiển thị SNR bằng màu sắc: xanh (tốt), vàng (trung bình), đỏ (yếu). |
| `ViewCounter/` | **Bộ đếm lượt xem** — Theo dõi số lượt truy cập (Vercel Analytics). |
| `VitalSignCard/` | **Thẻ chỉ số sinh lý** — Card hiển thị từng chỉ số (HR, HRV, SNR) với icon, giá trị, và đơn vị. |

---

### `frontend/src/hooks/` — Custom React Hooks

| File | Mô tả |
|---|---|
| `useWebcam.js` | **Quản lý webcam** — Request camera permission, tạo video stream, capture JPEG frames theo interval. Xử lý cleanup khi unmount. |
| `useWebSocket.js` | **Quản lý WebSocket** — Kết nối tới `/ws/stream` với JWT token. Auto-reconnect khi mất kết nối. Gửi frames (base64) và nhận vitals data. Parse JSON messages và cập nhật state. |
| `useSmoothedValue.js` | **Làm mượt giá trị** — EMA smoothing cho các chỉ số hiển thị (HR, SNR) để tránh nhảy số đột ngột trên UI. |

---

### `frontend/src/lib/` — Utilities

| File | Mô tả |
|---|---|
| `api.js` | **Axios client** — Cấu hình base URL, interceptors tự động gắn JWT token vào header `Authorization`. Xử lý refresh token và error handling. |
| `chatApi.js` | **Chat API client** — Wrapper cho endpoint `/chat`, gửi câu hỏi và nhận response. |
| `supabase.js` | **Supabase client** — Khởi tạo `createClient()` với URL và anon key từ env vars. Dùng cho auth operations. |
| `utils.js` | **Tiện ích chung** — Format số, format thời gian, helper functions. |
| `vitals.js` | **Vitals utilities** — Hàm helper xử lý và format dữ liệu vitals cho hiển thị. |

---

### `frontend/src/store/` — State Management

| File | Mô tả |
|---|---|
| `vitalsStore.js` | **Zustand store** — Quản lý state global cho vitals data (heart_rate, snr_db, hrv, bvp_window...). Cung cấp actions để update từ WebSocket và reset khi disconnect. |

---

### `frontend/src/features/auth/`

| File | Mô tả |
|---|---|
| `AuthProvider.jsx` | **Context Provider cho Auth** — Wrap app với auth context. Theo dõi trạng thái đăng nhập qua `supabase.auth.onAuthStateChange()`. Cung cấp `user`, `session`, `signOut` cho toàn bộ app. |

---

### `frontend/vite.config.js` — Cấu hình Vite

- Plugin: `@vitejs/plugin-react`
- Alias: `@` → `./src/`
- Dev server port: `3002`
- Proxy: `/health` và `/video` → `http://localhost:8001` (backend)

### `frontend/vercel.json` — Cấu hình deploy Vercel

Rewrites tất cả route về `index.html` cho SPA routing.

---

## `rPPG/` — Nghiên cứu & Huấn luyện mô hình

Thư mục chứa toàn bộ mã nguồn nghiên cứu, huấn luyện, đánh giá và export mô hình rPPG.

### Cấu trúc tổng quan

```
rPPG/
├── configs/                    # File cấu hình YAML cho training/inference
│   ├── train_configs/          # 64 file YAML — cấu hình training
│   └── infer_configs/          # 65 file YAML — cấu hình inference
├── docs/                       # Tài liệu kỹ thuật về mô hình
│   ├── models/                 # Tài liệu từng mô hình
│   ├── README.md               # Tổng quan tài liệu rPPG
│   ├── folder_structure.md     # Mô tả cấu trúc thư mục rPPG
│   ├── inference_pipeline.md   # Pipeline suy luận
│   ├── training_pipeline.md    # Pipeline huấn luyện
│   └── notebook_groups.md      # Phân nhóm notebooks
├── documents_paper/            # 9 bài báo khoa học gốc (PDF)
├── export/                     # Script export model sang ONNX
├── final_model_release/        # 36 file weights (.pth) đã huấn luyện
├── notebooks_training/         # 8 notebooks huấn luyện
├── notebooks_inference/        # 8 notebooks inference + tài liệu
└── optimize/                   # Tối ưu hóa model
```

---

### `rPPG/configs/` — File cấu hình YAML

#### `train_configs/` — Cấu hình huấn luyện (64 files)

Mỗi file YAML định nghĩa một thí nghiệm huấn luyện cụ thể. Quy ước đặt tên:

```
{TRAIN_DATASET}_{TEST_DATASET}_{MODEL}_BASIC.yaml
```

Ví dụ: `UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml` = Huấn luyện trên UBFC-rPPG, đánh giá trên PURE, model FactorizePhys.

Nội dung mỗi file gồm: đường dẫn data, hyperparameters (lr, batch_size, epochs), model config, loss function, data augmentation.

#### `infer_configs/` — Cấu hình inference (65 files)

Tương tự format train configs nhưng chỉ cho inference: đường dẫn weights, test dataset, metrics đánh giá.

---

### `rPPG/docs/models/` — Tài liệu mô hình

Mỗi file mô tả chi tiết một mô hình rPPG đã nghiên cứu:

| File | Mô hình | Đặc điểm chính |
|---|---|---|
| `FactorizePhys.md` | FactorizePhys (NeurIPS 2024) | NMF rank-1 attention, ~220K params, **model chính của hệ thống** |
| `EfficientPhys.md` | EfficientPhys | Temporal Shift Module, ~9M params |
| `PhysFormer.md` | PhysFormer | Transformer-based, ~30M params |
| `RhythmFormer.md` | RhythmFormer | Rhythm-aware Transformer, ~13M params |
| `PhysNet.md` | PhysNet | 3D CNN baseline, ~3M params |
| `PhysMamba.md` | PhysMamba | State Space Model (Mamba), ~3M params |
| `DeepPhys.md` | DeepPhys | 2D CNN + attention, ~9M params |
| `TS-CAN.md` | TS-CAN | Temporal Shift + attention, ~9M params |
| `iBVPNet.md` | iBVPNet | Encoder-Decoder 3D CNN, ~6M params |
| `BigSmall.md` | BigSmall | Multi-task (BVP + AU + respiration), ~9M params |
| `README.md` | — | Bảng tổng hợp so sánh tất cả mô hình |

---

### `rPPG/documents_paper/` — Bài báo gốc (PDF)

Chứa 9 bài báo khoa học gốc tương ứng với các mô hình đã nghiên cứu:

| File | Bài báo |
|---|---|
| `FactorizePhys.pdf` | FactorizePhys: Matrix Factorization for Multidimensional Attention (NeurIPS 2024) |
| `EfficientPhys.pdf` | EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals |
| `PhysFormer.pdf` | PhysFormer: Facial Video-based Physiological Measurement with Temporal Transformer |
| `RhythmFormer.pdf` | RhythmFormer: Extracting rPPG Signals Based on Hierarchical Temporal Transformer |
| `PhysNet.pdf` | DeepPhys → PhysNet: Remote Heart Rate Estimation with CNNs |
| `PhysMamba.pdf` | PhysMamba: State Space Duality Based on rPPG |
| `DeepPhys.pdf` | DeepPhys: Video-Based Physiological Measurement Using CNNs |
| `TS-CAN.pdf` | TS-CAN: Multi-Task Temporal Shift Attention Networks |
| `BigSmall.pdf` | BigSmall: Efficient Multi-Task Learning for Physiological Measurement |

---

### `rPPG/export/` — Export mô hình

| File | Mô tả |
|---|---|
| `export_onnx.py` | Script export mô hình PyTorch (.pth) sang ONNX (.onnx). Chứa inline toàn bộ source code của các model classes để đảm bảo self-contained. Hỗ trợ export tất cả mô hình đã nghiên cứu. |

---

### `rPPG/final_model_release/` — Weights đã huấn luyện (36 files)

Chứa tất cả model weights (.pth) đã huấn luyện trên nhiều dataset. Quy ước đặt tên:

```
{DATASET}_{MODEL}.pth
```

**Các dataset đã train:** PURE, UBFC-rPPG, SCAMPS, iBVP, BP4D, MA-UBFC.

**Các mô hình:** FactorizePhys, EfficientPhys, PhysFormer, RhythmFormer, PhysNet, PhysMamba, DeepPhys, TS-CAN, iBVPNet, BigSmall.

---

### `rPPG/notebooks_training/` — Notebooks huấn luyện (8 files)

Mỗi notebook chứa pipeline huấn luyện đầy đủ (data loading, model definition, training loop, evaluation) cho một nhóm mô hình:

| File | Nhóm mô hình |
|---|---|
| `groupA_training.ipynb` | DeepPhys, EfficientPhys |
| `groupB_training.ipynb` | TS-CAN |
| `groupC_training.ipynb` | PhysNet |
| `groupD_training.ipynb` | PhysFormer |
| `groupE_training.ipynb` | RhythmFormer, PhysMamba |
| `groupF_training.ipynb` | iBVPNet, FactorizePhys |
| `groupG_training.ipynb` | Unsupervised methods |
| `bigsmall_training.ipynb` | BigSmall (multi-task) |

---

### `rPPG/notebooks_inference/` — Notebooks đánh giá (9 files)

Mỗi notebook chạy inference trên các mô hình đã train, tính metrics (MAE, RMSE, correlation) và vẽ biểu đồ so sánh:

| File | Nhóm mô hình |
|---|---|
| `groupA_inference.ipynb` | DeepPhys, EfficientPhys |
| `groupB_inference.ipynb` | TS-CAN |
| `groupC_inference.ipynb` | PhysNet |
| `groupD_inference.ipynb` | PhysFormer |
| `groupE_inference.ipynb` | RhythmFormer, PhysMamba |
| `groupF_inference.ipynb` | iBVPNet, FactorizePhys |
| `groupG_inference.ipynb` | Unsupervised methods |
| `bigsmall_inference.ipynb` | BigSmall |
| `model_groups.md` | Tài liệu phân nhóm và so sánh kết quả |

---

### `rPPG/optimize/` — Tối ưu hóa

| File | Mô tả |
|---|---|
| `optimization_guide.md` | Hướng dẫn tối ưu hóa: quantization, pruning, mixed precision, ONNX optimization. |
| `optimized_training.ipynb` | Notebook training với các kỹ thuật tối ưu (mixed precision, gradient accumulation). |
| `optimized_inference.ipynb` | Notebook inference tối ưu (batched inference, ONNX Runtime optimization). |

---

## `hf_space/` — Hugging Face Spaces Deployment

Bản sao backend được cấu hình để deploy lên Hugging Face Spaces:

| File/Thư mục | Mô tả |
|---|---|
| `app/` | Bản sao `backend/app/` |
| `scripts/` | Bản sao `backend/scripts/` |
| `weights/` | Model weights |
| `vectorstore/` | FAISS index |
| `Dockerfile` | Dockerfile tối ưu cho HF Spaces |
| `start.sh` | Script khởi động (build embeddings + start uvicorn) |
| `requirements.txt` | Dependencies (đồng bộ với backend) |
| `.env.example` | Template biến môi trường |
| `README.md` | Metadata cho HF Spaces (title, sdk, emoji) |

---

## Tóm tắt quan hệ giữa các module

```
                    ┌─────────────┐
                    │  Frontend   │
                    │ (React+Vite)│
                    └──────┬──────┘
                           │ HTTP / WebSocket
                    ┌──────▼──────┐
                    │   Backend   │
                    │  (FastAPI)  │
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼─────┐  ┌───────▼───────┐
   │  Services   │  │  Chatbot  │  │    Worker     │
   │ (CV+Signal) │  │   (RAG)   │  │   (Celery)    │
   └──────┬──────┘  └─────┬─────┘  └───────┬───────┘
          │                │                │
   ┌──────▼──────┐  ┌─────▼─────┐  ┌───────▼───────┐
   │ ONNX Model  │  │   FAISS   │  │    Redis      │
   │ (weights/)  │  │   + BM25  │  │  (Broker)     │
   └─────────────┘  │   + Gemini│  └───────┬───────┘
                    └───────────┘          │
                                   ┌───────▼───────┐
                                   │  PostgreSQL   │
                                   │  (Supabase)   │
                                   └───────────────┘
```
