FROM python:3.10-slim

# Create user with UID 1000 as required by Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho OpenCV/MediaPipe (Run as root)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (chown to user so pip can write if needed, though running as root then switching is fine too)
COPY --chown=user ./requirements.txt requirements.txt

# Tải trước model Cross-Encoder (tránh cold-start)
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the application with proper permissions
COPY --chown=user . /app

# Ensure the /app directory itself is owned by user so SQLite can create new .db files
RUN chown -R user /app

# Switch to the non-root user
USER user

# Chạy FastAPI uvicorn trên port 7860 (Port bắt buộc của Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
