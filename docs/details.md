# File Details - Frontend & Backend

## Frontend (Ứng dụng React + JavaScript)

### Root Files
- **index.html** - Điểm vào HTML chính của ứng dụng, chứa root div cho React
- **main.jsx** - Điểm vào ứng dụng, render App component vào DOM
- **App/App.jsx** - Component chính, định tuyến trang (Home, Live, Upload) và layout chính
- **index.css** - Style CSS toàn cục cho ứng dụng (Vanilla CSS, cấu hình biến màu sắc, font)
- **vite.config.js** - Cấu hình build tool Vite (bundler, dev server, port 3002)
- **package.json** - Quản lý dependencies và scripts

### Components (`src/components/`)
Chứa các React components tái sử dụng (mỗi component nằm trong một thư mục chứa `.jsx` và `.css`):
- **Header/** - Component header/navbar của ứng dụng
- **VitalSignCard/** - Card hiển thị chỉ số sức khỏe (nhịp tim, SNR)
- **BVPChart/** - Biểu đồ hiển thị dữ liệu Blood Volume Pulse
- **FaceOverlay/** - Overlay để vẽ bounding box khuôn mặt lên video
- **ChatBot/** - Widget chatbot AI (floating button + modal chat)
- **HistoryView/** - Component xem và lọc lịch sử đo (dùng chung)
- **SnrBadge/** - Component hiển thị nhãn chất lượng tín hiệu (SNR)

### Pages (`src/pages/`)
Chứa các trang chính:
- **Home/** - Trang Landing Page giải thích tính năng và hướng dẫn sử dụng.
- **Live/** - Trang realtime, dùng webcam và WebSocket để nhận vitals live.
- **Upload/** - Trang upload video offline để hệ thống phân tích bất đồng bộ.

### Store (`src/store/`)
- **vitalsStore.js** - Zustand store chứa dữ liệu chỉ số sức khỏe real-time

### Hooks (`src/hooks/`)
Custom React hooks:
- **useWebSocket.js** - Hook kết nối WebSocket với backend để nhận dữ liệu real-time
- **useWebcam.js** - Hook quản lý webcam, capture video frames

### Utils (`src/lib/`)
Hàm utility và helper:
- **api.js** - API client gọi backend endpoints (uploadVideoAsync, getJobStatus, fetchHistory, fetchHistoryDetail)
- **chatApi.js** - API client cho chatbot (sendChatMessage)
- **utils.js** - Hàm utility chung (cn helper cho class names)
- **vitals.js** - Hàm xử lý dữ liệu chất lượng tín hiệu (SNR)

### Config Files
- **.env** - Biến môi trường (VITE_API_URL, VITE_WS_URL)
- **.env.example** - Mẫu biến môi trường
- **public/** - Folder chứa static files (favicon)

---

## Backend (API FastAPI + Python)

### Root Files
- **.env** - Biến môi trường (model paths, API keys, signal params)
- **.env.example** - Mẫu biến môi trường

### App (`app/`)
Thư mục chính chứa ứng dụng:
- **__init__.py** - Khởi tạo package
- **main.py** - Tạo FastAPI app instance, mount routers, cấu hình CORS

### Core (`app/core/`)
Cấu hình và thiết lập core:
- **config.py** - Cấu hình ứng dụng (Pydantic Settings: model paths, thresholds, chatbot keys)
- **lifespan.py** - Quản lý vòng đời ứng dụng (startup: load model, init DB)

### Schemas (`app/schemas/`)
Định nghĩa data models (Pydantic):
- **vitals.py** - Models cho request/response (FaceMessage, VitalsMessage, VideoResultResponse, HealthResponse)
- **history.py** - Models cho lịch sử đo (HistorySummary, HistoryDetailResponse)

### Services (`app/services/`)
Logic xử lý chính:
- **rppg_engine.py** - Engine chính xử lý rPPG (ONNX + PyTorch fallback)
- **face_detector.py** - Detect khuôn mặt bằng MediaPipe Face Mesh

- **preprocessor.py** - Tiền xử lý video (resize, normalize)
- **signal_processor.py** - Xử lý tín hiệu (bandpass, FFT, HRV, SNR)
- **history_store.py** - Quản lý kết nối PostgreSQL (Supabase) cho các thao tác CRUD lịch sử đo và người dùng

### Chatbot (`app/chatbot/`)
Module Advanced RAG chatbot:
- **__init__.py** - Khởi tạo package
- **loader.py** - Load tài liệu PDF/Markdown/Text từ `app/documents/`
- **vectorstore.py** - Build/load FAISS (Dense) và BM25 (Sparse) index
- **engine.py** - Advanced RAG chain kết hợp Query Rewriting, Hybrid Search, và Re-ranking (Google Gemini, lazy-loaded)
- **feedback_store.py** - Lưu trữ feedback người dùng (câu hỏi, câu trả lời, rating)
- **ingest.py** - Ingest nội dung mới vào kho tài liệu `app/documents/`
- **auto_update.py** - Helper rebuild vectorstore index cho automation

### Documents (`app/documents/`)
Tài liệu cho chatbot học:
- **Medical_book.pdf** - Sách y khoa dùng làm knowledge base

### API (`app/api/`)
Endpoints API:
- **router.py** - Tổng hợp tất cả routers

#### Routes (`app/api/routes/`)
- **health.py** - Health check endpoint
- **video.py** - Endpoints xử lý video (upload sync/async, jobs)
- **history.py** - Endpoints lấy danh sách và chi tiết lịch sử
- **chat.py** - Endpoint chatbot RAG (POST /chat, POST /chat/feedback)

#### WebSocket (`app/api/websocket/`)
- **stream.py** - WebSocket handler cho real-time streaming

### Scripts (`scripts/`)
- **build_embeddings.py** - Build FAISS vectorstore từ tài liệu
- **rebuild_embeddings.py** - Rebuild toàn bộ FAISS index
- **list_gemini_models.py** - Liệt kê các model Gemini có sẵn

### Tests (`tests/`)
- **test_api.py** - Unit tests cho API endpoints
- **test_chatbot_imports.py** - Tests kiểm tra import chatbot
- **test_signal_processor.py** - Tests cho signal processing

### Weights (`weights/`)
- **model_config.json** - Cấu hình model (model name, img_size, chunk, norm_type, fps)
- **\*.onnx** - 34 model ONNX cho inference

---

## Tóm Tắt Flow

### Upload Video Flow (Async):
1. User tải video lên từ **Upload.jsx** (Frontend)
2. Gửi file đến `POST /video/upload-async` qua **api.js**
3. Backend tạo background task và trả về `job_id`
4. Frontend polling `GET /video/jobs/{job_id}` mỗi 2 giây
5. Background task: **face_detector.py** → **rppg_engine.py** → **signal_processor.py** → lưu vào **history.db**
6. Frontend nhận kết quả 'done' và hiển thị (VitalSignCard, BVPChart)

### Real-time Webcam Flow:
1. **useWebcam.js** capture frames từ webcam (trong trang **Live.jsx**)
2. Gửi frames đến backend WebSocket via **useWebSocket.js**
3. Backend xử lý real-time, gửi kết quả trở lại via **stream.py**
4. Frontend hiển thị kết quả real-time trong **Live.jsx**
5. Khi ngắt kết nối (> 5 giây đo), record tự động được lưu vào **history.db**

### Chatbot Flow:
1. User click floating chat button → mở **ChatBot.jsx**
2. Gửi câu hỏi tới backend via **chatApi.js** → `POST /chat`
3. Backend: **engine.py** → FAISS retriever → Gemini LLM → trả lời
4. Frontend hiển thị trả lời + sources trong chat widget
