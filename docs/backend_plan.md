# Kế hoạch Phát triển Backend (Backend Plan)

Tài liệu này mô tả chi tiết các kế hoạch cải tiến, nâng cấp và mở rộng cho phần backend (FastAPI) của dự án Non-Invasive Health.

## 1. Trạng thái hiện tại

Backend hiện tại được xây dựng với **FastAPI** và hỗ trợ các tính năng chính:
- **rPPG Engine**: Hỗ trợ 34 model ONNX và fallback PyTorch, tự động phân biệt frame-wise và chunk-wise.
- **WebSocket Streaming**: Nhận frame từ webcam và trả về chỉ số realtime (Heart Rate, Blink Rate, SNR).
- **Video Upload (Offline & Async)**: Phân tích video tĩnh, lưu lịch sử bằng SQLite (`history.db`).
- **Chatbot RAG**: Tích hợp Google Gemini và FAISS vectorstore qua thư mục `app/documents/`.

## 2. Kế hoạch ngắn hạn (Short-term)

- **Đồng bộ lịch sử Realtime**: Cập nhật logic WebSocket để lưu trữ kết quả của các phiên đo qua webcam (`ws/stream`) vào cơ sở dữ liệu `history.db` sau khi phiên đo kết thúc.
- **Nâng cấp độ bền job async**: Hiện tại endpoint `POST /video/upload-async` sử dụng `BackgroundTasks` của FastAPI. Cần chuyển đổi sang một worker queue nhẹ như `RQ` (Redis Queue), `Celery`, hoặc `Dramatiq` để tránh mất dữ liệu job khi server khởi động lại.
- **Bổ sung API lấy trạng thái hệ thống chi tiết**: Mở rộng endpoint `/health` để hiển thị trạng thái của DB, vectorstore, và bộ nhớ GPU/CPU đang dùng.

## 3. Kế hoạch trung hạn (Medium-term)

- **Hỗ trợ model BigSmall đầy đủ**: Thêm logic để trích xuất tín hiệu hô hấp (Respiration Rate) ngoài nhịp tim từ các model hỗ trợ multi-task (như BigSmall).
- **User Authentication & Multi-user**: Thêm xác thực người dùng (JWT) để phân chia dữ liệu lịch sử theo từng cá nhân. Cập nhật `history.db` để chứa thông tin `user_id`.
- **Cải thiện Chatbot RAG**: 
  - Truyền thêm context về các chỉ số sức khỏe hiện tại của người dùng (HR, SNR) vào prompt của LLM.
  - Hỗ trợ lưu trữ context giao tiếp (multi-turn conversation history).
  - Thêm chức năng tự động rebuild vectorstore bằng webhook khi có file mới được thêm vào `app/documents/`.

## 4. Kế hoạch dài hạn (Long-term)

- **Tối ưu inference**: Triển khai batch inference hoặc TensorRT để tăng FPS cho luồng WebSocket khi có nhiều kết nối đồng thời.
- **Data Export & Phân tích chuyên sâu**: Viết API cho phép export lịch sử dưới dạng CSV/JSON kèm theo các báo cáo tổng quan (tuần/tháng) qua dạng biểu đồ nâng cao.
- **Mở rộng RAG thành Agent**: Tích hợp công cụ tìm kiếm web vào chatbot RAG để xử lý những câu hỏi y khoa vượt quá phạm vi của tài liệu nội bộ (cần đi kèm với bộ lọc an toàn và warning y tế rõ ràng).
