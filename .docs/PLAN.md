# Production Roadmap — Non-Invasive Health Analysis System (rPPG)

> **Mục tiêu:** Nâng cấp hệ thống từ prototype lên production chuyên nghiệp, đảm bảo khả năng mở rộng, bảo mật, quan sát được và dễ vận hành.

---

## 0. Tổng quan Priority Matrix

| Nhóm | Ưu tiên | Độ phức tạp | Tác động |
|---|---|---|---|
| Auth & Multi-tenancy | 🔴 P0 | Cao | Nền tảng cho toàn bộ hệ thống |
| Infrastructure as Code | 🔴 P0 | Trung bình | Triển khai ổn định, lặp lại |
| Job Queue & Workers | 🔴 P0 | Trung bình | Thay thế BackgroundTasks |
| Observability | 🟠 P1 | Trung bình | Debug, SLA |
| Security Hardening | 🟠 P1 | Cao | Bắt buộc cho y tế |
| CI/CD Pipeline | 🟠 P1 | Trung bình | Tự động hóa release |
| API Versioning & Contracts | 🟡 P2 | Thấp | Backward compat |
| Frontend Architecture | 🟡 P2 | Trung bình | UX, maintainability |
| ML Model Management | 🟡 P2 | Cao | Tái sử dụng model |
| Chatbot Enhancements | 🟢 P3 | Cao | Chất lượng RAG |

---

## 1. Authentication & Multi-Tenancy (P0)

### 1.1 Lựa chọn kiến trúc Auth

Dùng **Supabase Auth** (đã có Supabase DB) để tránh tự quản lý JWT secret, refresh token rotation và OAuth provider.

```
User → Supabase Auth (email/password, Google OAuth)
     → JWT access token (15 min) + refresh token (7 days)
     → Backend FastAPI verify via JWKS endpoint
```

**Không tự build auth từ đầu.** Supabase Auth tích hợp Row Level Security (RLS) trực tiếp với PostgreSQL.

### 1.2 Backend — FastAPI Auth Middleware

```python
# app/core/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from app.core.config import settings

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Tất cả endpoint `/history`, `/video/*`, `/chat` đều phải nhận `user_id` từ token thay vì không phân biệt người dùng.

### 1.3 Database — Row Level Security

```sql
-- Mỗi user chỉ đọc/ghi được dữ liệu của chính họ
ALTER TABLE history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own history"
  ON history FOR ALL
  USING (auth.uid() = user_id);

ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own jobs"
  ON jobs FOR ALL
  USING (auth.uid() = user_id);
```

### 1.4 Schema Migration — thêm user_id

```sql
-- Migration: 001_add_user_id.sql
ALTER TABLE history ADD COLUMN user_id UUID REFERENCES auth.users(id);
ALTER TABLE jobs    ADD COLUMN user_id UUID REFERENCES auth.users(id);

CREATE INDEX idx_history_user_id ON history(user_id);
CREATE INDEX idx_jobs_user_id    ON jobs(user_id);
```

Dùng **Alembic** để quản lý migrations, không chạy SQL tay.

### 1.5 Frontend — Auth Flow

```
src/
  features/
    auth/
      AuthProvider.jsx      # Context + Supabase client
      LoginPage.jsx
      RegisterPage.jsx
      useAuth.js            # Hook trả về user, signIn, signOut
```

Dùng `@supabase/supabase-js` client trực tiếp ở frontend. Mọi request API đều gắn `Authorization: Bearer <token>` vào header.

---

## 2. Infrastructure as Code (P0)

### 2.1 Docker hóa toàn bộ hệ thống

```
docker/
  backend/
    Dockerfile          # Multi-stage: builder + runtime
    .dockerignore
  frontend/
    Dockerfile          # Multi-stage: node builder + nginx
    nginx.conf
  worker/
    Dockerfile          # Image riêng cho Celery worker
docker-compose.yml      # Local dev: backend + worker + redis + frontend
docker-compose.prod.yml # Override cho production
```

**Dockerfile Backend (Multi-stage):**

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY weights/ ./weights/

RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
```

**Điểm quan trọng:**
- Image không chạy dưới user `root`.
- `weights/` ONNX model được COPY vào image, không mount volume (đảm bảo tính bất biến của image).
- Dùng `python:3.11-slim` thay vì `python:3.11` giảm ~700MB.

### 2.2 docker-compose cho local dev

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  backend:
    build: ./docker/backend
    env_file: backend/.env
    ports: ["8001:8001"]
    depends_on: [redis]

  worker:
    build: ./docker/worker
    env_file: backend/.env
    command: celery -A app.worker worker -Q video,default -c 4
    depends_on: [redis, backend]

  frontend:
    build: ./docker/frontend
    ports: ["3002:80"]
    depends_on: [backend]
```

### 2.3 Kubernetes (production)

Dùng Helm chart để deploy lên GKE / EKS / DigitalOcean Kubernetes.

```
k8s/
  charts/
    rppg-backend/
      Chart.yaml
      values.yaml
      templates/
        deployment.yaml
        service.yaml
        hpa.yaml            # HorizontalPodAutoscaler
        ingress.yaml
    rppg-worker/
      ...
    rppg-frontend/
      ...
```

**HorizontalPodAutoscaler** cho backend:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: rppg-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70
```

---

## 3. Job Queue với Celery + Redis (P0)

`BackgroundTasks` của FastAPI không có retry, không có visibility, không recover khi restart. Thay bằng **Celery + Redis Broker**.

### 3.1 Kiến trúc

```
POST /video/upload-async
  └─► FastAPI → lưu file vào object storage (S3/Supabase Storage)
             → celery.send_task("video.process", args=[job_id, file_key])
             → trả ngay job_id

Celery Worker (process riêng):
  video.process(job_id, file_key)
    ├─ download file từ object storage
    ├─ face_detector → rppg_engine → signal_processor
    ├─ update jobs SET status='done', result=... WHERE id=job_id
    └─ (nếu lỗi) update jobs SET status='failed', error=...
```

### 3.2 Cấu hình Celery

```python
# app/worker/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "rppg",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks.video"],
)

celery_app.conf.update(
    task_serializer="json",
    result_expires=3600,
    task_acks_late=True,          # ACK sau khi xử lý xong (tránh mất task khi crash)
    worker_prefetch_multiplier=1, # Mỗi worker lấy 1 task 1 lúc (phù hợp CPU-heavy)
    task_routes={
        "video.*": {"queue": "video"},
        "chatbot.*": {"queue": "default"},
    },
)
```

### 3.3 Task video processing

```python
# app/worker/tasks/video.py
from app.worker.celery_app import celery_app
from app.services import rppg_engine, face_detector, signal_processor
from app.services.history_store import save_session
from app.services.storage import download_file

@celery_app.task(
    name="video.process",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def process_video(self, job_id: str, file_key: str, user_id: str):
    try:
        update_job_status(job_id, "processing")
        file_bytes = download_file(file_key)
        result = run_rppg_pipeline(file_bytes)
        save_session(user_id, result)
        update_job_status(job_id, "done", result=result)
        cleanup_temp_file(file_key)
    except Exception as exc:
        update_job_status(job_id, "failed", error=str(exc))
        raise self.retry(exc=exc)
```

### 3.4 Flower — Monitor Jobs

```bash
celery -A app.worker flower --port=5555
```

Dashboard xem queue length, task history, worker health — thiết yếu cho production.

---

## 4. Object Storage cho File Video (P0)

Không lưu file video trên disk của container (ephemeral, không scale).

**Dùng Supabase Storage** (đã có Supabase) hoặc AWS S3.

```python
# app/services/storage.py
import boto3
from app.core.config import settings

s3 = boto3.client(
    "s3",
    endpoint_url=settings.storage_endpoint,   # Supabase S3-compat endpoint
    aws_access_key_id=settings.storage_key,
    aws_secret_access_key=settings.storage_secret,
)

def upload_video(file_bytes: bytes, job_id: str) -> str:
    key = f"uploads/{job_id}.mp4"
    s3.put_object(Bucket=settings.storage_bucket, Key=key, Body=file_bytes)
    return key

def download_file(key: str) -> bytes:
    obj = s3.get_object(Bucket=settings.storage_bucket, Key=key)
    return obj["Body"].read()

def delete_file(key: str):
    s3.delete_object(Bucket=settings.storage_bucket, Key=key)
```

**Lifecycle policy:** tự động xóa file sau 24h sau khi job hoàn thành.

---

## 5. Observability — Logging, Metrics, Tracing (P1)

"You can't manage what you can't measure."

### 5.1 Structured Logging với structlog

```python
# app/core/logging.py
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),  # Log dưới dạng JSON
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

# Sử dụng
log.info("video.processing.started", job_id=job_id, user_id=user_id)
log.error("video.processing.failed", job_id=job_id, exc_info=True)
```

**Ship logs** lên **Grafana Loki** hoặc **Datadog** thông qua Loki Docker plugin.

### 5.2 Metrics với Prometheus + Grafana

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "rppg_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

PROCESSING_DURATION = Histogram(
    "rppg_video_processing_seconds",
    "Video processing duration",
    buckets=[5, 10, 30, 60, 120, 300],
)

ACTIVE_WS_CONNECTIONS = Gauge(
    "rppg_websocket_connections_active",
    "Active WebSocket connections",
)

HEART_RATE_ACCURACY = Histogram(
    "rppg_heart_rate_bpm",
    "Measured heart rate distribution",
    buckets=range(40, 180, 5),
)
```

Expose `/metrics` endpoint với `prometheus-fastapi-instrumentator`.

**Grafana Dashboards cần có:**
- Request rate, error rate, latency (P50/P95/P99) — RED Method
- Job queue depth, worker utilization — Celery metrics
- WebSocket connection count, frame processing FPS
- Heart rate distribution, SNR quality heatmap
- Memory / CPU / GPU usage per pod

### 5.3 Distributed Tracing với OpenTelemetry

```python
# app/core/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
)
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)
```

Dùng **Jaeger** hoặc **Grafana Tempo** để xem trace từ HTTP request → Celery task → DB query.

### 5.4 Alerting

```yaml
# alertmanager rules
groups:
  - name: rppg
    rules:
      - alert: HighErrorRate
        expr: rate(rppg_http_requests_total{status_code=~"5.."}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Error rate > 5% for 2 minutes"

      - alert: JobQueueDepth
        expr: celery_queue_length{queue="video"} > 50
        for: 5m
        annotations:
          summary: "Video job queue backed up"

      - alert: WorkerDown
        expr: celery_workers_total < 1
        for: 1m
```

---

## 6. Security Hardening (P1)

### 6.1 API Security

```python
# Rate limiting — dùng slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)

@router.post("/video/upload-async")
@limiter.limit("10/minute")  # Max 10 upload/phút/IP
async def upload_video_async(...): ...

@router.post("/chat")
@limiter.limit("30/minute")
async def chat(...): ...
```

**Headers bảo mật** (middleware):

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from secure import Secure

secure_headers = Secure()

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    secure_headers.framework.fastapi(response)
    return response
```

Các header tối thiểu: `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`.

### 6.2 Input Validation

```python
# Kiểm tra file upload nghiêm ngặt
import magic

ALLOWED_MIME_TYPES = {"video/mp4", "video/avi", "video/quicktime"}
MAX_FILE_SIZE_MB = 200

async def validate_video_upload(file: UploadFile):
    # 1. Kiểm tra kích thước
    content = await file.read(MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    # 2. Magic bytes — không tin vào extension hay Content-Type header
    mime = magic.from_buffer(content[:2048], mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(415, f"Unsupported media type: {mime}")

    await file.seek(0)
    return content
```

### 6.3 Secrets Management

Không commit `.env` production lên git. Dùng:
- **Docker Secrets** hoặc **Kubernetes Secrets** (base64 encoded, encrypted at rest nếu bật KMS)
- **HashiCorp Vault** cho môi trường enterprise
- **Doppler** hoặc **AWS Secrets Manager** cho managed solution

```bash
# Không bao giờ làm thế này
GEMINI_API_KEY=AIzaSy... # hardcode trong code hay commit vào git

# Thay vào đó
kubectl create secret generic rppg-secrets \
  --from-literal=gemini-api-key=$GEMINI_API_KEY \
  --from-literal=supabase-db-url=$SUPABASE_DB_URL
```

### 6.4 CORS — Production Config

```python
# Không dùng allow_origins=["*"] trên production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### 6.5 WebSocket Security

```python
# stream.py — authenticate WebSocket handshake
@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, token: str = Query(...)):
    try:
        user = verify_jwt_token(token)
    except Exception:
        await websocket.close(code=1008)  # Policy violation
        return
    await websocket.accept()
    ...
```

---

## 7. CI/CD Pipeline (P1)

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
      redis:
        image: redis:7-alpine
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r backend/requirements.txt
      - run: pip install pytest pytest-asyncio pytest-cov httpx
      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v --cov=app --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json
      - run: cd frontend && npm ci
      - run: cd frontend && npm run lint
      - run: cd frontend && npm run test -- --coverage

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: "fs"
          scan-ref: "."
          severity: "CRITICAL,HIGH"
      - name: Bandit — Python security linter
        run: pip install bandit && bandit -r backend/app/ -ll

  build-and-push:
    needs: [test-backend, test-frontend, security-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push backend
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/backend/Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Kubernetes
        run: |
          helm upgrade --install rppg ./k8s/charts/rppg-backend \
            --set image.tag=${{ github.sha }} \
            --wait --timeout 5m
```

### 7.2 Branch Strategy

```
main        ─── production (auto deploy, protected branch)
develop     ─── staging (auto deploy)
feature/*   ─── development (PR → develop)
hotfix/*    ─── critical fix (PR → main + develop)
```

---

## 8. Database — Migration & Connection Pooling (P1)

### 8.1 Alembic Migration

```
backend/
  alembic/
    versions/
      001_initial_schema.py
      002_add_user_id.py
      003_add_indexes.py
    env.py
    alembic.ini
```

```bash
# Workflow migration
alembic revision --autogenerate -m "add user_id to history"
alembic upgrade head

# Rollback
alembic downgrade -1
```

**Quy tắc:** migration phải backward compatible (blue-green deploy).

### 8.2 Connection Pooling với PgBouncer

Backend dùng nhiều process/thread, mỗi cái mở kết nối DB riêng → PostgreSQL không chịu được.

```
FastAPI instances ──► PgBouncer (transaction mode, pool_size=20)
                                └──► PostgreSQL Supabase
```

Hoặc dùng `asyncpg` + SQLAlchemy async pool:

```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    settings.db_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # Kiểm tra connection còn sống trước khi dùng
    pool_recycle=1800,        # Recycle connection sau 30 phút
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

### 8.3 Indexes quan trọng

```sql
-- Các query thường gặp nhất
CREATE INDEX CONCURRENTLY idx_history_user_created
  ON history(user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_history_type
  ON history(user_id, type) WHERE type IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_jobs_status
  ON jobs(status, created_at) WHERE status IN ('pending', 'processing');
```

---

## 9. API Versioning & Contract Testing (P2)

### 9.1 Versioning

```python
# app/api/v1/router.py  — current endpoints
# app/api/v2/router.py  — new breaking changes

app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")
```

Expose OpenAPI spec tại `/api/v1/openapi.json` và `/api/v2/openapi.json`.

### 9.2 Contract Testing với Schemathesis

```bash
# Tự động test tất cả endpoint dựa trên OpenAPI schema
schemathesis run http://localhost:8001/openapi.json \
  --checks all \
  --hypothesis-max-examples 100
```

### 9.3 Deprecation Policy

```python
from fastapi import Response

@router.post("/video/upload", deprecated=True)  # Đánh dấu trong OpenAPI docs
async def upload_sync(response: Response, ...):
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2025-12-31"  # Ngày ngừng hỗ trợ
    ...
```

---

## 10. Frontend Architecture (P2)

### 10.1 Cấu trúc theo Feature-Sliced Design

```
frontend/src/
  app/               # App-wide: router, providers, global styles
    App.jsx
    Router.jsx
    Providers.jsx    # QueryClient, AuthProvider, ThemeProvider

  pages/             # Route-level components (thin, chỉ compose features)
    Home/
    Live/
    Upload/
    Auth/

  features/          # Vertical slices theo domain
    auth/
      AuthProvider.jsx
      LoginForm.jsx
      useAuth.js
    vitals/
      VitalCard.jsx
      BVPChart.jsx
      useVitals.js
    history/
      HistoryList.jsx
      HistoryFilter.jsx
      useHistory.js
    chatbot/
      ChatWidget.jsx
      ChatMessage.jsx
      useChatbot.js
    video-upload/
      UploadZone.jsx
      JobStatus.jsx
      useVideoUpload.js
    webcam/
      WebcamFeed.jsx
      FaceOverlay.jsx
      useWebcam.js

  shared/            # Truly reusable, không phụ thuộc domain
    ui/              # Button, Input, Modal, Badge, Spinner...
    hooks/           # useDebounce, useLocalStorage...
    lib/             # api.js, utils.js, vitals.js
```

### 10.2 Server State — TanStack Query

Thay toàn bộ polling thủ công bằng TanStack Query:

```javascript
// features/video-upload/useVideoUpload.js
import { useMutation, useQuery } from "@tanstack/react-query";
import { uploadVideoAsync, getJobStatus } from "@/shared/lib/api";

export function useVideoUpload() {
  const uploadMutation = useMutation({
    mutationFn: uploadVideoAsync,
    onSuccess: (data) => {
      // TanStack Query tự refetch khi cần
    },
  });

  return { upload: uploadMutation.mutate, isPending: uploadMutation.isPending };
}

export function useJobStatus(jobId) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId),
    refetchInterval: (data) =>
      data?.status === "done" || data?.status === "failed" ? false : 2000,
    enabled: !!jobId,
  });
}
```

### 10.3 Error Boundary + Loading States

```jsx
// Mọi route đều có Error Boundary và Suspense
<ErrorBoundary fallback={<ErrorPage />}>
  <Suspense fallback={<PageSkeleton />}>
    <UploadPage />
  </Suspense>
</ErrorBoundary>
```

### 10.4 Performance Frontend

```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          charts: ["recharts"],
          supabase: ["@supabase/supabase-js"],
        },
      },
    },
  },
  plugins: [
    react(),
    compression(),        // Brotli/gzip assets
    imageOptimize(),      // Optimize ảnh tĩnh
  ],
});
```

---

## 11. ML Model Management (P2)

### 11.1 Model Registry

Đừng commit file `.onnx` vào git (file lớn, slow clone).

Dùng **DVC (Data Version Control)** hoặc **MLflow Model Registry**:

```bash
# Lưu model vào S3, track bằng DVC
dvc add weights/model.onnx
dvc push  # upload lên S3

# Pull model khi deploy
dvc pull
```

### 11.2 Model Hot-swap

```python
# app/services/rppg_engine.py
class RPPGEngine:
    def __init__(self):
        self._model = None
        self._model_version = None

    def load_model(self, model_path: str, version: str):
        """Load model mới mà không restart server"""
        new_model = load_onnx(model_path)
        self._model = new_model
        self._model_version = version
        log.info("model.loaded", version=version)

    @property
    def model_version(self):
        return self._model_version
```

Expose endpoint `POST /admin/model/reload` (chỉ admin) để swap model mà không cần restart.

### 11.3 A/B Testing Model

```python
# Chạy 2 model song song, so sánh accuracy trên production traffic
class ABTestEngine:
    def __init__(self, model_a, model_b, traffic_split=0.1):
        self.model_a = model_a  # 90% traffic
        self.model_b = model_b  # 10% traffic (new model)

    def predict(self, frames, user_id):
        use_b = hash(user_id) % 100 < self.traffic_split * 100
        model = self.model_b if use_b else self.model_a
        result = model.predict(frames)
        log_metric("model_version", "b" if use_b else "a", result)
        return result
```

---

## 12. Chatbot RAG Enhancements (P3)

### 12.1 Evaluation Pipeline

```python
# scripts/eval_chatbot.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

eval_dataset = load_eval_dataset("tests/chatbot_eval_set.json")
scores = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print(scores)  # Faithfulness: 0.92 | Relevancy: 0.87 | Recall: 0.79
```

Chạy evaluation trong CI sau mỗi lần thay đổi `engine.py` hoặc document mới.

### 12.2 Vectorstore Auto-rebuild

```python
# app/chatbot/auto_update.py
import hashlib
from pathlib import Path

def get_documents_hash() -> str:
    """Hash toàn bộ file trong documents/"""
    h = hashlib.md5()
    for f in sorted(Path("app/documents").rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()

async def check_and_rebuild():
    current_hash = get_documents_hash()
    stored_hash = await redis.get("documents:hash")
    if current_hash != stored_hash:
        await rebuild_vectorstore()
        await redis.set("documents:hash", current_hash)
```

### 12.3 Streaming Response Chatbot

```python
# app/api/routes/chat.py
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user=Depends(get_current_user)):
    async def generate():
        async for chunk in rag_engine.astream(request.question):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

Frontend dùng `EventSource` hoặc `fetch` với stream reader để hiển thị câu trả lời token-by-token.

---

## 13. Testing Strategy (P1)

### 13.1 Pyramid Testing

```
                    ┌─────────┐
                    │  E2E    │  Playwright — 10 test cases
                   ┌┴─────────┴┐
                   │Integration│  pytest + testcontainers — 50 tests
                  ┌┴───────────┴┐
                  │    Unit     │  pytest / vitest — 200+ tests
                  └─────────────┘
```

### 13.2 Backend Tests

```python
# tests/test_video_pipeline.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_upload_video_async_returns_job_id():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/video/upload-async",
            files={"file": ("test.mp4", b"fake_video_bytes", "video/mp4")},
            headers={"Authorization": f"Bearer {TEST_JWT}"},
        )
    assert response.status_code == 202
    assert "job_id" in response.json()

@pytest.mark.asyncio
async def test_signal_processor_bandpass():
    import numpy as np
    from app.services.signal_processor import bandpass_filter
    signal = np.random.randn(300)
    filtered = bandpass_filter(signal, low=0.75, high=2.5, fs=30)
    assert filtered.shape == signal.shape
```

### 13.3 Frontend Tests

```javascript
// features/vitals/VitalCard.test.jsx
import { render, screen } from "@testing-library/react";
import VitalCard from "./VitalCard";

test("renders heart rate with bpm unit", () => {
  render(<VitalCard label="Heart Rate" value={72} unit="bpm" />);
  expect(screen.getByText("72")).toBeInTheDocument();
  expect(screen.getByText("bpm")).toBeInTheDocument();
});

test("shows warning when SNR is low", () => {
  render(<VitalCard label="SNR" value={2.1} threshold={3.0} />);
  expect(screen.getByRole("alert")).toBeInTheDocument();
});
```

### 13.4 Load Testing

```python
# locustfile.py
from locust import HttpUser, task, between

class HealthAPIUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def get_history(self):
        self.client.get("/api/v1/history", headers=self.headers)

    @task(1)
    def chat(self):
        self.client.post(
            "/api/v1/chat",
            json={"question": "Nhịp tim bình thường là bao nhiêu?"},
            headers=self.headers,
        )
```

```bash
locust --headless -u 100 -r 10 --run-time 5m --host http://api.yourdomain.com
```

**Target SLO:** P95 latency < 500ms cho `/history`, P95 < 3s cho `/chat`.

---

## 14. Roadmap Thực Thi

### Phase 1 — Foundation (Tuần 1–3)

- [ ] Docker hóa backend, worker, frontend
- [ ] Alembic migrations setup
- [ ] Supabase Auth tích hợp (backend middleware + frontend)
- [ ] Celery + Redis thay BackgroundTasks
- [ ] Object storage cho video files

### Phase 2 — Reliability (Tuần 4–6)

- [ ] Structured logging với structlog → Loki
- [ ] Prometheus metrics + Grafana dashboards
- [ ] GitHub Actions CI pipeline (test + build + push)
- [ ] Rate limiting + security headers
- [ ] Input validation nghiêm ngặt (magic bytes check)

### Phase 3 — Scale (Tuần 7–10)

- [ ] Kubernetes deployment với Helm charts
- [ ] HPA cho backend và worker
- [ ] PgBouncer / async SQLAlchemy connection pool
- [ ] OpenTelemetry tracing
- [ ] Load testing + SLO definition

### Phase 4 — Polish (Tuần 11–14)

- [ ] Frontend Feature-Sliced Design refactor
- [ ] TanStack Query migration
- [ ] Chatbot streaming response
- [ ] ML Model Registry (DVC)
- [ ] RAGAS evaluation pipeline cho chatbot
- [ ] Playwright E2E tests

---

## 15. Checklist Production Readiness

Trước khi go-live, kiểm tra đầy đủ các mục sau:

```
Infrastructure
  ☐ All secrets vào Vault / K8s Secrets, không có gì trong code
  ☐ TLS/HTTPS enabled, HTTP redirect to HTTPS
  ☐ Backup PostgreSQL tự động (daily, 30-day retention)
  ☐ Container images không chạy dưới root

Security
  ☐ OWASP Top 10 review
  ☐ Trivy scan: không có CRITICAL CVE trong images
  ☐ Bandit scan: không có high-severity issue trong Python code
  ☐ CORS chỉ allow production domain
  ☐ Rate limiting trên tất cả public endpoints

Reliability
  ☐ Health check endpoint trả về DB + Redis status
  ☐ Graceful shutdown (SIGTERM handling)
  ☐ Job retry + dead letter queue
  ☐ Circuit breaker cho external calls (Gemini API)

Observability
  ☐ Centralized logs, search được
  ☐ Alerting cho error rate, queue depth, worker down
  ☐ Dashboards cho RED metrics (Rate, Errors, Duration)

Performance
  ☐ Load test pass tại 100 concurrent users
  ☐ P95 latency trong SLO
  ☐ DB indexes đã tạo, query plan đã kiểm tra

Operations
  ☐ Runbook viết cho incident response
  ☐ On-call rotation setup
  ☐ Rollback procedure documented và tested
```

---
