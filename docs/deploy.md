# Hướng dẫn deploy với Hugging Face Spaces (backend) + Vercel (frontend)

Đây là cách triển khai nhanh cho dự án Non-Invasive Health của bạn:
- Backend FastAPI + WebSocket + ONNX trên **Hugging Face Spaces (Docker Space)**
- Frontend React/Vite trên **Vercel**

## 1. Tại sao chọn cách này

- `Hugging Face Spaces` Free có cấu hình tốt cho AI/ML: **2 vCPU + 16 GB RAM**.
- `Docker Space` cho phép bạn chạy backend FastAPI nguyên gốc, không cần chuyển sang Gradio hoặc Streamlit.
- `Vercel` là môi trường ideal cho frontend React tĩnh và bắt buộc dùng HTTPS để mở webcam.
- `Vercel + Hugging Face` phù hợp để người khác dùng thử trực tiếp mà vẫn dùng được webcam và WebSocket bảo mật.

## 2. Quan trọng trước khi làm

### 2.1 Cần dùng `wss://` chứ không phải `ws://`

Frontend phải nối WebSocket tới:

```text
wss://<username>-<spaces-name>.hf.space/ws/stream
```

Nếu dùng `ws://` trên trang HTTPS thì trình duyệt sẽ bị chặn.

### 2.2 Lưu ý kích thước model

Repo của bạn có nhiều file model `.onnx` rất nặng. Để deploy dễ dàng:
- Tốt nhất chỉ giữ một model `.onnx` cần dùng trong Space
- Hoặc tải model vào container khi khởi động
- Nếu file quá lớn, cân nhắc dùng Git LFS hoặc một repo riêng cho Space

## 3. Chuẩn bị backend cho Hugging Face Spaces

### 3.1 Tạo Space mới

1. Vào https://huggingface.co/spaces
2. Nhấn `Create new Space`
3. Chọn `SDK: Docker`
4. Chọn `Template: Blank`
5. Đặt tên Space, ví dụ: `non-invasive-health`

### 3.2 Thêm Dockerfile vào `backend/`

Tạo tệp `backend/Dockerfile` với nội dung:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

> Hugging Face Spaces Docker mặc định dùng port `7860`, nên backend phải lắng nghe port này.

### 3.3 Kiểm tra file `.env`

Backend của bạn dùng `backend/.env` để đọc cấu hình. Ví dụ tối thiểu cần có:

```env
MODEL_PATH=weights/PURE_DeepPhys.onnx
MODEL_CONFIG_PATH=weights/model_config.json
DEVICE=cpu

# Chatbot (nếu sử dụng)
GEMINI_API_KEY=your_gemini_api_key_here
CHATBOT_MODEL=gemini-2.5-flash
VECTORSTORE_PATH=vectorstore/faiss_index
```

Nếu bạn dùng model khác thì sửa `MODEL_PATH` cho đúng.
Nếu không sử dụng chatbot, có thể bỏ qua `GEMINI_API_KEY`.

### 3.4 Cấu trúc thư mục cho Space

Đảm bảo trong repository gửi lên Hugging Face chỉ gồm:

- `Dockerfile`
- `requirements.txt`
- `backend/app/` và tất cả code backend
- `backend/weights/` chứa model cần dùng
- `backend/.env`

Nếu muốn, bạn có thể tạo một Space repo riêng chỉ chứa `backend/` để tránh đẩy cả dự án lớn.

## 4. Triển khai backend lên Hugging Face

### 4.1 Đẩy code lên Space

1. Clone Space hoặc dùng nút Upload trên UI
2. Đẩy `backend/` và `Dockerfile` vào repo Space
3. Chờ Spaces build Docker image

### 4.2 Kiểm tra backend chạy tốt

Sau khi build xong, mở Space và thử:
- `https://<username>-<space>.hf.space/health`
- `https://<username>-<space>.hf.space/docs`

Nếu bạn thấy swagger và `GET /health` trả về đúng thì backend đã chạy.

## 5. Triển khai frontend lên Vercel

### 5.1 Đẩy frontend lên GitHub

1. Nếu chưa có repo GitHub cho frontend, tạo thêm repo mới hoặc dùng repo hiện tại.
2. Đảm bảo cấu trúc `frontend/` có đủ `package.json`, `vite.config.js`, `src/`,...
3. Commit và đẩy code lên GitHub.

### 5.2 Kết nối Vercel

1. Vào https://vercel.com/dashboard
2. Nhập `New Project`
3. Chọn repo GitHub của frontend
4. Chọn root project là `frontend`
5. Để Vercel tự nhận `Vite` và build command thông thường

### 5.3 Thêm biến môi trường

Trong `Project Settings > Environment Variables` của Vercel thêm:

| Key | Value |
| --- | --- |
| `VITE_WS_URL` | `wss://<username>-<space>.hf.space/ws/stream` |

> Chú ý: `VITE_*` phải bắt đầu với `VITE_` để Vite đưa vào frontend build.

### 5.4 Build settings (nếu cần)

Nếu Vercel không tự nhận thì dùng:
- Build Command: `npm install && npm run build`
- Output Directory: `dist`

## 6. Điều chỉnh kết nối WebSocket trong frontend

File `frontend/src/hooks/useWebSocket.js` đã dùng biến môi trường:

```js
const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8001/ws/stream'
```

Vì vậy khi deploy trên Vercel, chỉ cần khai báo `VITE_WS_URL` là frontend sẽ kết nối tới Hugging Face Space.

Ghi chú thực tế cho dự án này:
- Frontend có ba trang: `Home` (landing page), `Live` (realtime via webcam/WebSocket) và `Upload` (upload video offline + xem lịch sử).
- Môi trường sản xuất (Vercel + Hugging Face Space) phải thiết lập `VITE_WS_URL` với `wss://...` (ví dụ `wss://<username>-<space>.hf.space/ws/stream`).
- Khi phát triển local, fallback `ws://localhost:8001/ws/stream` là phù hợp nếu backend chạy trên port `8001`.
- Ngoài ra frontend dùng `VITE_API_URL` (mặc định `http://localhost:8001`) cho HTTP API (`/video/upload`, `/history`).

Frontend `Upload` page hiện đang sử dụng luồng async (`POST /video/upload-async` + polling `GET /video/jobs/{job_id}`) để xử lý video.

Nếu sử dụng chatbot khi deploy, cần:
1. Đảm bảo `GEMINI_API_KEY` được cấu hình trong environment của Space.
2. Chạy `python scripts/build_embeddings.py` để build vectorstore trước khi start server.
3. Hoặc thêm bước build vectorstore vào Dockerfile.

## 7. Test hệ thống đầy đủ

### 7.1 Kiểm tra frontend

Mở URL Vercel sau khi deploy, ví dụ:

```text
https://<your-frontend>.vercel.app
```

### 7.2 Kiểm tra kết nối WebSocket

- Trang web phải mở được webcam qua HTTPS
- WebSocket phải kết nối tới `wss://...hf.space/ws/stream`
- Nếu trình duyệt báo lỗi Mixed Content hoặc WebSocket blocked, chắc chắn bạn đang dùng `ws://` hoặc Tham số URL sai

### 7.3 Kiểm tra backend

Dùng browser thử trực tiếp:

```text
https://<username>-<space>.hf.space/health
https://<username>-<space>.hf.space/docs
```

## 8. Một số lưu ý khi deploy

- `Hugging Face Spaces` phù hợp demo, nhưng không phải môi trường production lớn.
- Nếu model quá nặng và Space build thất bại, hãy giữ lại chỉ 1 file `.onnx` hoặc tải model từ external URL trong Dockerfile.
- `Vercel` chỉ dùng cho frontend tĩnh, không host backend WebSocket.
- Backend đã bật CORS với `*`, nên frontend sẽ không bị chặn khi kết nối từ Vercel.

## 9. Nếu muốn giảm dung lượng deploy

Cách đơn giản nhất:
- Tạo repo Space riêng chỉ chứa backend và weights cần dùng
- Loại bỏ các model `.onnx` khác nếu không cần
- Nếu ONNX lớn >100MB, cân nhắc dùng `git lfs` trên Hugging Face

## 10. Kết luận

Với kiến trúc hiện tại, phương án **Hugging Face Spaces (Docker)** cho backend và **Vercel** cho frontend là phù hợp.

- Backend: `wss://<username>-<space>.hf.space/ws/stream`
- Frontend: deploy trên Vercel, dùng `VITE_WS_URL`

Chỉ cần đảm bảo backend chạy được trên port `7860`, frontend dùng `wss://`, và model `.onnx` đủ nhỏ hoặc được quản lý hợp lý.
