# Kế hoạch cải tiến

> Dự án đã có backend FastAPI xử lý upload video, history persistence, frontend React/Vite, và chatbot RAG (Gemini + FAISS).

---

## 1. Trạng thái hiện tại

- Backend có history persistence qua SQLite `history.db`.
- Frontend có `Home` page realtime dùng webcam/WebSocket và `Upload` page cho upload video offline + lịch sử.
- Upload video offline trả kết quả HR / Blink / SNR / HRV và cho phép export CSV BVP.
- Chatbot RAG đã tích hợp: Google Gemini + FAISS vectorstore + tài liệu từ `backend/app/documents/`.
- Backend có route `POST /video/upload-async` và `GET /video/jobs/{job_id}` nhưng frontend hiện chỉ dùng upload đồng bộ.
- Realtime live page hoạt động, nhưng lịch sử chỉ lưu upload offline.

## 2. Đã hoàn thành ✅

- ✅ Backend core: rPPG engine, face detection, signal processing, blink detection.
- ✅ Frontend Home page: webcam realtime + WebSocket.
- ✅ Frontend Upload page: video upload + lịch sử.
- ✅ History persistence (SQLite).
- ✅ Chatbot RAG: backend module + frontend widget.
- ✅ 34/36 model ONNX đã convert.

## 3. Ưu tiên cao (High) — ✅ Đã hoàn thành

- ✅ Hoàn thiện UI upload async:
  - form upload gọi `POST /video/upload-async`,
  - hiển thị `job_id` và polling trạng thái mỗi 2s,
  - thông báo khi job hoàn tất hoặc thất bại.
- ✅ Hoàn thiện lịch sử:
  - lọc theo `type`, `start_at`, `end_at` với filter panel,
  - modal chi tiết record history (click vào row),
  - hiển thị `duration_sec`, `age_group`, `bandpass_low_hz`, `bandpass_high_hz`, `pnn50`, `sdnn_ms`, `peak_count`.
- ✅ Thêm phản hồi chất lượng tín hiệu:
  - cảnh báo khi `snr_db` < 3 dB (banner vàng),
  - badge SNR (Xuất sắc/Tốt/Trung bình/Yếu) trên kết quả và bảng lịch sử.

## 4. Ưu tiên trung bình (Medium)

- 🟡 Hỗ trợ model `BigSmall` để tính thêm `respiration_rate`.
- 🟡 Đồng bộ history realtime với offline:
  - khi session WebSocket hoàn thành, ghi record vào `history.db`.
- 🟡 Nâng cấp độ bền job async:
  - chuyển `BackgroundTasks` sang worker queue nhẹ (RQ, Celery, Dramatiq),
  - hoặc thêm cơ chế recovery khi server restart.
- 🟡 Cải thiện chatbot:
  - thêm context vitals vào prompt (HR, SNR hiện tại),
  - hỗ trợ conversation history (multi-turn).

## 5. Ưu tiên thấp (Low)

- 🟢 User authentication / multi-user history.
- 🟢 Dashboard tổng quan ngày/tuần.
- 🟢 Export lịch sử CSV/JSON + biểu đồ HRV / SNR theo thời gian.
- 🟢 Auto-rebuild vectorstore khi tài liệu thay đổi.

## 6. Gợi ý triển khai tiếp theo

- Giữ `Upload` page làm trung tâm, nhưng thêm modal/section detail history.
- Tập trung vào chất lượng dữ liệu trước khi mở rộng UX.
- Ưu tiên cải thiện backend inference và giảm độ trễ `ws/stream`.
