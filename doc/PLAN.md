# Kế hoạch cải tiến

> Hệ thống hiện tại đã chạy được (backend port 8001 + frontend port 3002).  
> Tài liệu này liệt kê các hướng cải tiến có thể thực hiện tiếp theo.

---

## Mức độ ưu tiên

| Ký hiệu | Nghĩa |
|---------|-------|
| 🔴 High | Ảnh hưởng trực tiếp đến độ chính xác hoặc trải nghiệm |
| 🟡 Medium | Cải thiện đáng kể nhưng không cấp thiết |
| 🟢 Low | Nice-to-have |

---

## 1. Model & Inference

### 🔴 Cho phép chọn model trên frontend
Thêm dropdown chọn model (DeepPhys / FactorizePhys / PhysFormer/...) — backend load model theo request.

### 🟡 Bật GPU inference
Cài `onnxruntime-gpu` và đổi `DEVICE=cuda:0` trong `.env` để tăng tốc inference ~3–5x.

```bash
pip install onnxruntime-gpu
```

### 🟡 Thêm đo nhịp thở (BigSmall)
Model BigSmall dự đoán cả BVP + nhịp thở trong 1 lần chạy.  
Cần thêm field `respiration_rate` vào WebSocket response.

---

## 2. Face Detection

### 🔴 Thay Haar Cascade bằng detector tốt hơn
Haar Cascade bỏ sót mặt khi nghiêng hoặc ánh sáng thay đổi.  
Các lựa chọn tốt hơn:

| Detector | Ưu điểm | Cài đặt |
|---------|---------|---------|
| **YOLOv5Face** | Chính xác nhất, đã có trong codebase gốc | `pip install ultralytics` |
| **OpenCV DNN** | Nhẹ, chính xác hơn Haar, không cần thêm dep | Có sẵn trong OpenCV |
| **dlib** | Ổn định với nhiều góc độ | `pip install dlib` |

### 🟡 Dynamic face detection
Hiện tại detect face mỗi frame. Có thể detect mỗi N frames và track giữa các frame để giảm CPU.

---

## 3. Backend

### 🔴 Test và hoàn thiện video upload
Endpoint `POST /video/upload` đã có code nhưng chưa test end-to-end.  
Cần test với các loại video khác nhau (MP4, AVI, MOV, độ phân giải khác nhau).

### 🟡 Cải thiện signal processing
- Thêm **adaptive bandpass** tự động điều chỉnh dải tần theo tuổi người dùng
- Thêm **HRV** (Heart Rate Variability) từ khoảng cách giữa các đỉnh BVP
- Tính **confidence score** dựa trên SNR để cho biết kết quả đáng tin cậy không

### 🟡 Xử lý video dài (> 30s)
Hiện tại xử lý toàn bộ video synchronous.  
Video dài sẽ block request — cần dùng `BackgroundTasks` của FastAPI.

### 🟢 Lưu lịch sử đo
Dùng SQLite (không cần cài thêm) để lưu kết quả mỗi phiên.  
Thêm endpoint `GET /history` để xem lại.

---

## 4. Frontend

### 🔴 Hiển thị confidence / cảnh báo chất lượng
Khi SNR < 3 dB, hiển thị cảnh báo "Tín hiệu yếu — điều chỉnh ánh sáng hoặc vị trí".

### 🟡 Thêm trang kết quả sau khi đo xong
Sau khi stop webcam, hiển thị tóm tắt: HR trung bình, min, max, biểu đồ toàn phiên.

### 🟡 Responsive mobile
Giao diện hiện tại chưa tối ưu cho màn hình nhỏ.  
Cần điều chỉnh layout 2 cột thành 1 cột trên mobile.

### 🟢 Dark mode mặc định
Hiện tại mặc định là light mode. Đổi theo system preference (`prefers-color-scheme`).

---

## 5. Chất lượng code

### 🟡 Viết test cho backend
File `tests/test_api.py` đã có skeleton nhưng chưa có test thực.  
Tối thiểu cần test: `/health`, `/video/upload` với video mẫu, WebSocket handshake.

### 🟢 Thêm `.env.example` cho backend
File `.env.example` bị mất, cần tạo lại để người khác clone repo biết cần set gì.

---

## 6. Tương lai xa

### 🟢 Stress / emotion detection
Kết hợp HR + HRV để ước lượng mức độ căng thẳng.

### 🟢 PWA (Progressive Web App)
Cho phép cài trên điện thoại như app native, truy cập camera dễ hơn.

### 🟢 Hỗ trợ nhiều người dùng đồng thời
Hiện mỗi WebSocket session độc lập — đã hỗ trợ nhưng chưa test với nhiều client.
