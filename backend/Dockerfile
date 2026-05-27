FROM python:3.10-slim

# Thiết lập user non-root (UID 1000) để tương thích với các nền tảng như Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho OpenCV/MediaPipe (chạy bằng quyền root)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy file cấu hình thư viện trước để tận dụng Docker Cache
COPY --chown=user ./requirements.txt requirements.txt

# Cài đặt thư viện và Tải trước các mô hình AI (tránh bị cold-start khi user chat lần đầu)
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Copy toàn bộ mã nguồn dự án vào container
COPY --chown=user . /app

# Chuyển sang user non-root để tăng tính bảo mật
USER user

# Mặc định chạy FastAPI trên port 7860 (Port tiêu chuẩn của Hugging Face)
# Khi chạy Celery Worker, lệnh này sẽ bị ghi đè bởi cấu hình trong docker-compose.yml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
