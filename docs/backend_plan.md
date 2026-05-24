# Kế hoạch Phát triển Backend (Backend Plan)

Tài liệu này mô tả chi tiết các kế hoạch cải tiến, nâng cấp và mở rộng cho phần backend (FastAPI) của dự án Non-Invasive Health.

## 1. Kế hoạch ngắn hạn (Short-term)

- **Nâng cấp độ bền job async**: Cần chuyển đổi cơ chế xử lý upload nền (`POST /video/upload-async`) sang một worker queue thực thụ như `RQ` (Redis Queue), `Celery`, hoặc `Dramatiq` để tránh mất dữ liệu job khi server khởi động lại (hiện đang dùng `BackgroundTasks`).
- **Bổ sung API lấy trạng thái hệ thống chi tiết**: Mở rộng endpoint `/health` để hiển thị trạng thái của DB, vectorstore, và bộ nhớ GPU/CPU đang dùng.

## 2. Kế hoạch trung hạn (Medium-term)

- **Hỗ trợ model BigSmall đầy đủ**: Thêm logic để trích xuất tín hiệu hô hấp (Respiration Rate) ngoài nhịp tim từ các model hỗ trợ multi-task (như BigSmall).

## 3. Kế hoạch dài hạn (Long-term)

- **Tối ưu inference**: Triển khai batch inference hoặc TensorRT để tăng FPS cho luồng WebSocket khi có nhiều kết nối đồng thời.
- **Data Export & Phân tích chuyên sâu**: Viết API cho phép export lịch sử dưới dạng CSV/JSON kèm theo các báo cáo tổng quan (tuần/tháng) qua dạng biểu đồ nâng cao.
- **Mở rộng RAG thành Agent**: Tích hợp công cụ tìm kiếm web vào chatbot RAG để xử lý những câu hỏi y khoa vượt quá phạm vi của tài liệu nội bộ (cần đi kèm với bộ lọc an toàn và warning y tế rõ ràng).
