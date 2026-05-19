# File Details - Frontend & Backend

## Frontend (Ứng dụng React + TypeScript)

### Root Files
- **index.html** - Điểm vào HTML chính của ứng dụng, chứa root div cho React
- **main.tsx** - Điểm vào TypeScript, render App component vào DOM
- **App.tsx** - Component chính, định tuyến trang (Home, Upload) và layout chính
- **index.css** - Style CSS toàn cục cho ứng dụng
- **vite.config.ts** - Cấu hình build tool Vite (bundler, dev server, port 3002)
- **tsconfig.json** - Cấu hình TypeScript cho mã nguồn
- **tsconfig.node.json** - Cấu hình TypeScript cho các file config (Vite, PostCSS)
- **postcss.config.js** - Cấu hình PostCSS (xử lý CSS, tương thích với Tailwind)
- **tailwind.config.ts** - Cấu hình Tailwind CSS (framework CSS utility-first)
- **package.json** - Quản lý dependencies và scripts

### Components (`src/components/`)
Chứa các React components tái sử dụng:
- **Header.tsx** - Component header/navbar của ứng dụng
- **VitalSignCard.tsx** - Card hiển thị chỉ số sức khỏe (nhịp tim, blink rate, SNR)
- **BVPChart.tsx** - Biểu đồ hiển thị dữ liệu Blood Volume Pulse
- **FaceOverlay.tsx** - Overlay để vẽ bounding box khuôn mặt lên video
- **ChatBot.tsx** - Widget chatbot AI (floating button + modal chat)

### Pages (`src/pages/`)
Chứa các trang chính:
- **Home.tsx** - Trang realtime, dùng webcam và WebSocket để nhận vitals live.
- **Upload.tsx** - Trang upload video offline và xem lịch sử phiên đo.

### Store (`src/store/`)
- **vitalsStore.ts** - Zustand store chứa dữ liệu chỉ số sức khỏe real-time

### Hooks (`src/hooks/`)
Custom React hooks:
- **useWebSocket.ts** - Hook kết nối WebSocket với backend để nhận dữ liệu real-time
- **useWebcam.ts** - Hook quản lý webcam, capture video frames

### Types (`src/types/`)
Định nghĩa TypeScript types:
- **vitals.ts** - Types/interfaces cho dữ liệu chỉ số sức khỏe (VideoResult, HistoryRecord)
- **chat.ts** - Types cho chatbot (ChatMessage, ChatResponse)

### Utils (`src/lib/`)
Hàm utility và helper:
- **api.ts** - API client gọi backend endpoints (uploadVideo, fetchHistory)
- **chatApi.ts** - API client cho chatbot (sendChatMessage)
- **utils.ts** - Hàm utility chung (cn helper cho class names)

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
- **blink_detector.py** - Detect chuyển động mắt/blink rate
- **preprocessor.py** - Tiền xử lý video (resize, normalize)
- **signal_processor.py** - Xử lý tín hiệu (bandpass, FFT, HRV, SNR)
- **history_store.py** - Quản lý SQLite history.db (CRUD lịch sử đo)

### Chatbot (`app/chatbot/`)
Module RAG chatbot:
- **__init__.py** - Khởi tạo package
- **loader.py** - Load tài liệu PDF/Markdown/Text từ `app/documents/`
- **vectorstore.py** - Build/load FAISS vector index
- **engine.py** - RAG chain (Google Gemini + FAISS retriever, lazy-loaded)

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
- **chat.py** - Endpoint chatbot RAG (POST /chat)

#### WebSocket (`app/api/websocket/`)
- **stream.py** - WebSocket handler cho real-time streaming

### Scripts (`scripts/`)
- **build_embeddings.py** - Build FAISS vectorstore từ tài liệu

### Tests (`tests/`)
- **test_api.py** - Unit tests cho API endpoints
- **test_chatbot_imports.py** - Tests kiểm tra import chatbot
- **test_signal_processor.py** - Tests cho signal processing

### Weights (`weights/`)
- **model_config.json** - Cấu hình model (model name, img_size, chunk, norm_type, fps)
- **\*.onnx** - 34 model ONNX cho inference

---

## Tóm Tắt Flow

### Upload Video Flow:
1. User tải video lên từ **Upload.tsx** (Frontend)
2. Gửi file đến **video.py** endpoint via **api.ts**
3. Backend xử lý: **face_detector.py** → **rppg_engine.py** → **signal_processor.py**
4. Kết quả trả về qua HTTP response
5. Frontend hiển thị kết quả (VitalSignCard, BVPChart)
6. Record được lưu vào **history.db** qua **history_store.py**

### Real-time Webcam Flow:
1. **useWebcam.ts** capture frames từ webcam
2. Gửi frames đến backend WebSocket via **useWebSocket.ts**
3. Backend xử lý real-time, gửi kết quả trở lại via **stream.py**
4. Frontend hiển thị kết quả real-time trong **Home.tsx**

### Chatbot Flow:
1. User click floating chat button → mở **ChatBot.tsx**
2. Gửi câu hỏi tới backend via **chatApi.ts** → `POST /chat`
3. Backend: **engine.py** → FAISS retriever → Gemini LLM → trả lời
4. Frontend hiển thị trả lời + sources trong chat widget
