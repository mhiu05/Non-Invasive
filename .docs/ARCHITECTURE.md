# Kiến trúc Hệ thống Non-Invasive Health (rPPG)

Tài liệu này mô tả kiến trúc hiện tại của dự án, bao gồm backend FastAPI, frontend React/Vite, luồng xử lý video, lịch sử đo, WebSocket realtime và chatbot AI.

## 1. Tổng quan luồng dữ liệu

Dự án hiện có ba luồng chính:

- `HTTP` upload video offline: người dùng upload video và backend xử lý để trả về kết quả HR / SNR / HRV.
- `WebSocket` realtime: frontend gửi frame webcam đến backend, backend trả về vitals livestream.
- `HTTP` chatbot: người dùng hỏi câu hỏi, backend dùng RAG (FAISS + Gemini) để trả lời.

Frontend hiện tại có ba trang chính:

- `Home` là landing page giới thiệu tính năng và hướng dẫn sử dụng.
- `Live` cho realtime webcam và phân tích live.
- `Upload` để upload video offline và xem lịch sử phiên đo.

Chatbot widget hiển thị trên mọi trang (floating button góc phải).

## 2. Cấu trúc backend

### 2.1 Startup và shared state

- `backend/app/main.py` khởi tạo FastAPI.
- `backend/app/core/lifespan.py` tải model ONNX, khởi tạo `FaceDetector`, và gọi `init_history_db()`.
- Các global singleton ở runtime:
  - `engine: RPPGEngine`
  - `face_detector: FaceDetector`

### 2.2 Cấu hình

- `backend/app/core/config.py` chứa cấu hình chung:
  - `model_path`, `model_config_path`, `device`, `fps`
  - `max_upload_mb`
  - tham số bandpass
  - `gemini_api_key`, `chatbot_model` (cho chatbot RAG)

### 2.3 API endpoints

- `GET /health` — kiểm tra tình trạng dịch vụ.
- `POST /video/upload` — xử lý video offline ngay lập tức.
- `POST /video/upload-async` — nhận upload video và tạo job nền.
- `GET /video/jobs/{job_id}` — tra cứu trạng thái job async.
- `GET /history` — lấy danh sách phiên đo đã lưu.
- `GET /history/{history_id}` — lấy chi tiết record history.
- `POST /chat` — chatbot RAG (hỏi đáp qua tài liệu nội bộ).
- `POST /chat/feedback` — lưu phản hồi người dùng về câu trả lời chatbot.
- `ws://localhost:8001/ws/stream` — WebSocket realtime nhận frame webcam.

### 2.4 Lưu trữ lịch sử

- `backend/app/services/history_store.py` quản lý kết nối cơ sở dữ liệu **PostgreSQL (Supabase)**.
- Cấu trúc bảng `history` lưu metadata của mỗi phiên:
  - `id`, `created_at`, `type`, `filename`, `session_id`, `duration_sec`, `heart_rate`, `snr_db`, `age`, `age_group`, `bandpass_low_hz`, `bandpass_high_hz`, `hrv_ms`, `sdnn_ms`, `rmssd_ms`, `pnn50`, `peak_count`, `result`.
- `GET /history` trả list record theo thứ tự mới nhất, với lọc `type`, `start_at`, `end_at`.

### 2.5 Video job persistence & Job Queue

- `backend/app/api/routes/video.py` lưu trữ trạng thái job async vào bảng `jobs` trên cơ sở dữ liệu **PostgreSQL (Supabase)**.
- Bảng `jobs` chứa `id`, `status`, `created_at`, `updated_at`, `result`, `error`, `file_path`.
- `POST /video/upload-async` tạo job, upload trực tiếp video lên **S3 Object Storage** thông qua `backend/app/services/storage.py`, và đẩy Message Task vào hàng đợi.
- **Celery Worker** (chạy song song và lắng nghe **Redis** broker) sẽ tự động nhận job, download file từ S3, khởi tạo model và tiến hành phân tích rPPG. Sau khi phân tích thành công, file trên S3 sẽ được xóa để giải phóng dung lượng.
- `GET /video/jobs/{job_id}` trả trạng thái và kết quả.

### 2.6 WebSocket realtime

- Endpoint `ws/stream` nằm trong `backend/app/api/websocket/stream.py`.
- Protocol:
  - Client gửi `{"type":"frame","data":"..."}` hoặc `{"type":"reset"}`.
  - Server trả `face` / `vitals` / `error`.
- Mỗi kết nối WebSocket dùng `SessionState` riêng để giữ buffer rPPG và trạng thái.

### 2.7 Chatbot RAG

- Module `backend/app/chatbot/` chứa các file chính:
  - `loader.py` — load tài liệu từ `backend/app/documents/` (PDF, .md, .txt).
  - `vectorstore.py` — build/load FAISS index (Dense) và BM25 index (Sparse) tại `backend/vectorstore/faiss_index/`.
  - `engine.py` — Advanced RAG chain sử dụng Google Gemini, bao gồm Query Rewriting, Hybrid Search và Re-ranking (lazy-loaded khi có request đầu tiên).
  - `feedback_store.py` — thu thập và lưu trữ phản hồi người dùng về câu trả lời chatbot.
  - `ingest.py` — nạp tài liệu mới vào vectorstore.
  - `auto_update.py` — tự động cập nhật vectorstore khi tài liệu thay đổi.
- Tài liệu chatbot được lưu trong `backend/app/documents/` (hiện có `Medical_book.pdf`).
- Script `backend/scripts/build_embeddings.py` chạy 1 lần để build vectorstore.
- Endpoint `POST /chat` nhận câu hỏi và trả lời dựa trên tài liệu nội bộ.

## 3. Các dịch vụ xử lý chính

### 3.1 Face detection

- `backend/app/services/face_detector.py` dùng MediaPipe Face Mesh.
- Chỉ detect 1 mặt, mở rộng bbox và resize crop về kích thước phù hợp với model.

### 3.2 rPPG engine

- `backend/app/services/rppg_engine.py` khởi tạo `RPPGEngine` từ cấu hình model.
- Hỗ trợ ONNX runtime và fallback PyTorch.
- Phân biệt model `frame-wise` và `chunk-wise`.
- `SessionState` giữ buffer input và BVP tạm cho từng phiên.

### 3.3 Xử lý tín hiệu

- `backend/app/services/signal_processor.py` xử lý BVP:
  - detrend,
  - bandpass Butterworth,
  - FFT peak tìm HR,
  - tính SNR,
  - tính HRV (RMSSD/SDNN/PNN50).
- Dải bandpass điều chỉnh theo tuổi người dùng qua `get_bandpass_by_age()`.

## 4. Kiến trúc frontend

- `frontend` là ứng dụng React + Vite + JavaScript (sử dụng CSS Modules).
- `Home` page là landing page giới thiệu tính năng, giải thích các chỉ số sức khỏe và hướng dẫn sử dụng.
- `Live` page là giao diện đo realtime qua webcam và WebSocket, hiển thị:
  - luồng webcam với bounding box khuôn mặt,
  - kết quả vitals real-time (HR, SNR, HRV),
  - biểu đồ sóng BVP,
  - phần history nhúng trực tiếp.
- `Upload` page là trung tâm UX offline, hiển thị:
  - video upload (async processing),
  - tuổi người dùng,
  - trạng thái xử lý (polling job),
  - kết quả HR / SNR / HRV,
  - export CSV BVP signal,
  - phần history nhúng trực tiếp.
- `ChatBot` component — widget chat floating trên mọi trang, gọi `POST /chat`.
- `frontend/src/lib/api.js` cung cấp API client:
  - `uploadVideoAsync()` → `POST /video/upload-async` (upload bất đồng bộ).
  - `getJobStatus()` → `GET /video/jobs/{job_id}` (polling trạng thái job).
  - `fetchHistory()` → `GET /history`.
- `frontend/src/lib/chatApi.js` cung cấp chatbot client:
  - `sendChatMessage()` → `POST /chat`.
- `frontend/src/hooks/useWebSocket.js` và `frontend/src/hooks/useWebcam.js` cung cấp realtime webcam/WebSocket logic.
- `frontend/src/lib/vitals.js` chứa các hàm xử lý hiển thị chất lượng tín hiệu SNR.

## 5. Thực trạng hiện tại

- History persistence đã được triển khai đầy đủ cho cả đo video offline và phiên đo realtime qua webcam. WebSocket `stream.py` tự động lưu phiên realtime vào lịch sử khi kết thúc (nếu duration > 5 giây).
- Frontend có `Home` page realtime dùng webcam/WebSocket và `Upload` page cho upload video offline + lịch sử.
- Lịch sử được hiển thị ngay trong trang `Upload`, không còn route lịch sử riêng biệt.
- Chatbot RAG đã tích hợp hoàn chỉnh theo kiến trúc Advanced RAG (bao gồm Query Rewriting bằng LLM, Hybrid Search kết hợp FAISS + BM25, và Re-ranking bằng mô hình Cross-Encoder). Sử dụng Google Gemini làm LLM chính. Ngoài ra có module `feedback_store.py` thu thập phản hồi, `ingest.py` nạp tài liệu, và `auto_update.py` tự động cập nhật vectorstore.
- Frontend `Upload` sử dụng `uploadVideoAsync()` để upload bất đồng bộ và `getJobStatus()` để polling trạng thái job cho đến khi hoàn thành.

## 6. Hạn chế kiến trúc
- Chatbot vectorstore cần rebuild thủ công khi tài liệu thay đổi.
- Chưa có cơ chế người dùng đăng nhập phức tạp, phân quyền bảo mật nhiều lớp.
