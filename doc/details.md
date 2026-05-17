# File Details - Frontend & Backend

## Frontend (Ứng dụng React + TypeScript)

### Root Files
- **index.html** - Điểm vào HTML chính của ứng dụng, chứa root div cho React
- **main.tsx** - Điểm vào TypeScript, render App component vào DOM
- **App.tsx** - Component chính của ứng dụng, định tuyến trang và layout chính
- **index.css** - Style CSS toàn cục cho ứng dụng
- **vite.config.ts** - Cấu hình build tool Vite (bundler, dev server)
- **tsconfig.json** - Cấu hình TypeScript cho mã nguồn
- **tsconfig.node.json** - Cấu hình TypeScript cho các file config (Vite, PostCSS)
- **postcss.config.js** - Cấu hình PostCSS (xử lý CSS, tương thích với Tailwind)
- **tailwind.config.ts** - Cấu hình Tailwind CSS (framework CSS utility-first)
- **package.json** - Quản lý dependencies và scripts (npm packages, build scripts)

### Components (`src/components/`)
Chứa các React components tái sử dụng:
- **Header.tsx** - Component header/navbar của ứng dụng
- **VitalSignCard.tsx** - Card hiển thị chỉ số sức khỏe (nhịp tim, huyết áp, v.v.)
- **BVPChart.tsx** - Biểu đồ hiển thị dữ liệu máu (Blood Volume Pulse)
- **FaceOverlay.tsx** - Overlay để vẽ lên khuôn mặt (landmarks, detection)

### Pages (`src/pages/`)
Chứa các trang chính:
- **Home.tsx** - Trang chủ, hiển thị kết quả phân tích
- **Upload.tsx** - Trang upload video/webcam để phân tích

### Store (`src/store/`)
Quản lý state toàn cục:
- **vitalsStore.ts** - Store chứa dữ liệu chỉ số sức khỏe (nhịp tim, huyết áp, etc.)

### Hooks (`src/hooks/`)
Custom React hooks:
- **useWebSocket.ts** - Hook kết nối WebSocket với backend để nhận dữ liệu real-time
- **useWebcam.ts** - Hook quản lý webcam, capture video

### Types (`src/types/`)
Định nghĩa TypeScript types:
- **vitals.ts** - Types/interfaces cho dữ liệu chỉ số sức khỏe

### Utils (`src/lib/`)
Hàm utility và helper:
- **api.ts** - Hàm API client, gọi backend endpoints
- **utils.ts** - Hàm utility chung (format dữ liệu, tính toán, v.v.)

### Config Files
- **.env** - Biến môi trường (API_URL, ports, etc.)
- **public/** - Folder chứa static files (images, favicon, v.v.)

---

## Backend (API FastAPI + Python)

### Root Files
- **main.py** - Điểm vào ứng dụng FastAPI
- **.env** - Biến môi trường (DB config, model paths, etc.)

### App (`app/`)
Thư mục chính chứa ứng dụng:
- **__init__.py** - Khởi tạo package
- **main.py** - Tạo FastAPI app instance, mount routers, cấu hình CORS

### Core (`app/core/`)
Cấu hình và thiết lập core:
- **config.py** - Cấu hình ứng dụng (environment, model paths, thresholds)
- **lifespan.py** - Quản lý vòng đời ứng dụng (startup, shutdown events)

### Schemas (`app/schemas/`)
Định nghĩa data models (Pydantic):
- **vitals.ts** - Models cho request/response data (VitalsResponse, ProcessingResult)

### Services (`app/services/`)
Logic xử lý chính:
- **rppg_engine.py** - Engine chính xử lý rPPG (remote Photoplethysmography)
- **face_detector.py** - Detect khuôn mặt trong video
- **blink_detector.py** - Detect chuyển động mắt/blink
- **preprocessor.py** - Tiền xử lý video (resize, normalize)
- **signal_processor.py** - Xử lý tín hiệu (filtering, peak detection)

### API (`app/api/`)
Endpoints API:
- **router.py** - Tổng hợp tất cả routers

#### Routes (`app/api/routes/`)
- **health.py** - Health check endpoint (kiểm tra API hoạt động)
- **video.py** - Endpoints xử lý video (upload, process)

#### WebSocket (`app/api/websocket/`)
- **stream.py** - WebSocket handler cho real-time streaming (gửi dữ liệu live đến frontend)

### Tests (`tests/`)
- **test_api.py** - Unit tests cho API endpoints

### Weights (`weights/`)
- **model_config.json** - Cấu hình model
- **PURE_DeepPhys.onnx** - Model ONNX (inference, nhẹ hơn PyTorch)
- **PURE_DeepPhys_meta.json** - Metadata model (version, input/output shapes)

---

## Tóm Tắt Flow

### Upload Video Flow:
1. User tải video lên từ **Upload.tsx** (Frontend)
2. Gửi file đến **video.py** endpoint via **api.ts**
3. Backend xử lý: **preprocessor.py** → **face_detector.py** → **rppg_engine.py** → **signal_processor.py**
4. Kết quả được gửi qua WebSocket (**stream.py**)
5. Frontend nhận dữ liệu via **useWebSocket.ts** hook, cập nhật store
6. **Home.tsx** hiển thị kết quả từ store (BVPChart, VitalSignCard)

### Real-time Webcam Flow:
1. **useWebcam.ts** capture frames từ webcam
2. Gửi frames đến backend WebSocket
3. Backend xử lý real-time, gửi kết quả trở lại via **stream.py**
4. Frontend hiển thị kết quả real-time trong **Home.tsx**
