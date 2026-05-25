# Hướng dẫn Triển khai (Deployment)

Dự án Non-Invasive Health Analysis có kiến trúc Fullstack kết hợp AI, yêu cầu backend xử lý WebSocket liên tục và frontend tĩnh. Phương án tối ưu và miễn phí:
- **Backend (FastAPI + AI Models)**: Triển khai trên **Hugging Face Spaces (Docker Space)**.
- **Frontend (React/Vite)**: Triển khai trên **Vercel** hoặc **Netlify**.

---

## 1. Triển khai Backend (Hugging Face Spaces)

Hugging Face cung cấp cấu hình 2 vCPU + 16 GB RAM miễn phí, đủ để chạy rPPG (CPU Inference) và RAG Chatbot.

### Bước 1: Tạo Space mới
1. Truy cập [Hugging Face Spaces](https://huggingface.co/spaces) -> **Create new Space**.
2. Chọn SDK: **Docker** -> **Blank**.
3. Truy cập **Settings > Variables and secrets**:
   - Thêm secret `GEMINI_API_KEY` (Chứa Google Gemini API Key của bạn).

### Bước 2: Cấu hình Dockerfile
Tại thư mục `backend/`, đảm bảo bạn có file `Dockerfile` với nội dung chuẩn như sau:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho OpenCV/MediaPipe
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải trước model Cross-Encoder (tránh cold-start)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Chạy FastAPI uvicorn trên port 7860 (Port bắt buộc của Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Bước 3: Đẩy code và dữ liệu
1. **Model rPPG**: Để tránh nghẽn bộ nhớ, hãy chỉ đẩy **1 model ONNX tối ưu nhất** (ví dụ: `FactorizePhys.onnx`) lên thư mục `backend/weights/`. Cập nhật `model_config.json` tương ứng.
2. **Vectorstore Chatbot**: Việc build file vector rất nặng và cần gọi API, do đó bạn hãy chạy `python scripts/build_embeddings.py` ở Local trước. Sau đó **copy cả thư mục `vectorstore/`** đẩy lên Space.
3. Commit và đẩy toàn bộ thư mục `backend/` (cùng Dockerfile) lên repo của Hugging Face Space.

*Khi build thành công, URL backend của bạn sẽ có định dạng: `https://<username>-<space_name>.hf.space`*

---

## 2. Triển khai Frontend (Vercel)

Vercel (hoặc các host tương đương) bắt buộc sử dụng HTTPS. Do đó kết nối WebSockets tới backend phải đổi thành `wss://`.

### Bước 1: Cấu hình Vercel
1. Đẩy thư mục `frontend/` lên một GitHub repository riêng biệt (hoặc cấu hình Root Directory trên Vercel trỏ vào thư mục `frontend`).
2. Đăng nhập [Vercel](https://vercel.com/), chọn **Add New... > Project** và liên kết với repo GitHub của bạn.

### Bước 2: Thiết lập Biến môi trường
Trong phần **Environment Variables** trên Vercel, thêm 2 biến sau:
- `VITE_API_URL` = `https://<username>-<space_name>.hf.space`
- `VITE_WS_URL` = `wss://<username>-<space_name>.hf.space/ws/stream`

*(Lưu ý: Thay thế URL bằng đường dẫn thực tế của Hugging Face Space bạn vừa tạo).*

### Bước 3: Deploy
Nhấn **Deploy** và chờ Vercel build. Sau khi hoàn thành, mở địa chỉ `.vercel.app` để kiểm tra. Trình duyệt sẽ yêu cầu cấp quyền Camera để bắt đầu sử dụng.

---