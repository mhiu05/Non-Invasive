# Kế hoạch Nghiên cứu và Cải tiến Mô hình rPPG (rPPG Plan)

Tài liệu này đề ra các hướng nghiên cứu, cải tiến và tích hợp mô hình đo sinh trắc học mới cho dự án.

## 1. Kế hoạch ngắn hạn

- **Đánh giá lại Pipeline tiền xử lý**: Tối ưu hóa quá trình cắt vùng mặt (Face Cropping) và chuẩn hóa dữ liệu đầu vào. Tránh việc thay đổi ánh sáng đột ngột làm nhiễu sóng BVP.
- **Tối ưu hóa tài nguyên phần cứng**: Tích hợp ONNX Runtime với các Execution Provider như OpenVINO (cho thiết bị Intel) hoặc TensorRT (nếu có GPU NVIDIA) để tăng FPS và giảm mức tiêu thụ CPU.

## 2. Kế hoạch trung hạn

- **Hỗ trợ Respiration Rate (Nhịp thở)**: Nghiên cứu trích xuất tần số hô hấp từ tín hiệu rPPG hoặc sử dụng các mô hình multi-task như BigSmall để dự đoán đồng thời nhịp tim và nhịp thở.
- **Nghiên cứu mô hình nhẹ cho Edge Devices**: Cải tiến hoặc huấn luyện lại các mô hình gọn nhẹ (như EfficientPhys) trên tập dữ liệu đa dạng hơn để có thể chạy mượt mà ngay trên điện thoại di động (trình duyệt web mobile) thông qua WebAssembly / ONNX.js.

## 3. Kế hoạch dài hạn

- **Giải quyết nhiễu chuyển động (Motion Artifacts)**: Tích hợp các thuật toán loại bỏ nhiễu chuyển động tiên tiến (chẳng hạn dùng tín hiệu từ cảm biến gia tốc trên điện thoại hoặc kết hợp mạng sinh đối kháng GAN) để đo chính xác ngay cả khi người dùng đang nói chuyện hoặc di chuyển đầu nhiều.
- **Blood Pressure (Huyết áp) / SpO2**: Thử nghiệm và đánh giá tính khả thi của việc đo độ bão hòa oxy trong máu (SpO2) và huyết áp tâm thu/tâm trương dựa trên sự thay đổi màu sắc vi mô của da ở các bước sóng khác nhau (yêu cầu phân tích sâu phổ màu).
