# Chính sách Bảo mật (Security Policy)

Hệ thống được thiết kế với mức độ bảo mật tiêu chuẩn, tuân thủ các quy tắc bảo vệ quyền riêng tư của dữ liệu y tế (Health Data).

## Xác thực & Ủy quyền (Authentication & Authorization)
- Toàn bộ cơ chế đăng nhập và cấp quyền được ủy thác cho **Supabase Auth**.
- API sử dụng chuẩn **JWT (JSON Web Token)** để bảo mật. Các file frontend đều tự động đính kèm `Bearer Token` khi thực hiện Request.
- **Row Level Security (RLS)** được kích hoạt trên Supabase PostgreSQL để đảm bảo người dùng nào chỉ được phép xem lịch sử đo đạc của chính người đó.

## Lưu trữ Dữ liệu (Data Storage & Privacy)
- **Video:** File video tải lên được lưu trữ tạm thời tại S3 Object Storage (Supabase). Sau khi Celery Worker phân tích và lấy xong nhịp tim, **file video sẽ bị xóa vĩnh viễn ngay lập tức** khỏi hệ thống đám mây để đảm bảo quyền riêng tư về hình ảnh.
- **Webcam Real-time:** Dữ liệu hình ảnh từ Webcam chỉ được gửi qua WebSocket để xử lý trực tiếp trên RAM và trả về kết quả, **không hề lưu lại bất kỳ khung hình nào** xuống ổ cứng.

## Báo cáo Lỗ hổng (Reporting a Vulnerability)
Nếu bạn phát hiện bất kỳ lỗ hổng bảo mật nào (chẳng hạn như lộ API Key trong mã nguồn, lỗi RLS ở Supabase, SQL Injection), vui lòng **không công khai** nó dưới dạng GitHub Issue.
Hãy liên hệ trực tiếp với người quản trị kho lưu trữ qua email hoặc các kênh nội bộ để chúng tôi có thể khắc phục ngay lập tức. Cảm ơn bạn!
