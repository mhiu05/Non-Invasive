# Kiến trúc và Tài liệu Frontend

Tài liệu này mô tả kiến trúc chi tiết, các công nghệ sử dụng và luồng dữ liệu của frontend trong dự án Non-Invasive Health.

## 1. Tổng quan Công nghệ

Frontend được phát triển như một Single Page Application (SPA) với các công nghệ hiện đại:
- **Framework**: React 18 + TypeScript.
- **Build Tool**: Vite (để phát triển nhanh và tối ưu hóa bundle).
- **State Management**: Zustand (nhẹ, nhanh và quản lý state dễ dàng cho các biến real-time).
- **Routing**: React Router DOM (quản lý điều hướng giữa Home, Upload).
- **Styling**: Tailwind CSS (tiện dụng để tạo UI nhanh chóng và responsive).
- **Data Visualization**: Recharts (vẽ biểu đồ BVP real-time).

## 2. Cấu trúc Thư mục

- `src/components/`: Chứa các React components có thể tái sử dụng.
  - `ChatBot.tsx`: Widget Chatbot RAG.
  - `BVPChart.tsx`: Biểu đồ dạng sóng thể tích máu.
  - `VitalSignCard.tsx`: Hiển thị chỉ số (Nhịp tim, Blink rate).
- `src/pages/`: Các trang chính.
  - `Home.tsx` / `Live.tsx`: Trang thực hiện đo real-time qua webcam.
  - `Upload.tsx`: Trang upload video để đo offline và xem lịch sử.
- `src/hooks/`: Custom hooks.
  - `useWebcam.ts`: Hook lấy luồng media từ webcam.
  - `useWebSocket.ts`: Quản lý kết nối Socket liên tục tới backend.
- `src/store/`: Quản lý state toàn cục.
  - `vitalsStore.ts`: Lưu trữ trạng thái của các chỉ số sức khỏe đang được tính toán.
- `src/lib/`: Các hàm tiện ích và API services.
  - `api.ts`: Helper cho REST API gọi tới backend.
  - `chatApi.ts`: Xử lý giao tiếp với chatbot endpoint.

## 3. Luồng hoạt động chính

### 3.1. Luồng Real-time (Webcam)
1. Người dùng vào trang Home/Live, cho phép truy cập webcam (`useWebcam`).
2. Frontend sử dụng `useWebSocket` để mở kết nối tới `ws/stream` của backend.
3. Liên tục lấy frame từ luồng video, gửi base64 qua WebSocket.
4. Backend trả kết quả, `vitalsStore` cập nhật và React render lại biểu đồ BVP, nhịp tim.

### 3.2. Luồng Offline (Video Upload)
1. Người dùng truy cập trang Upload, thả file video vào khu vực dropzone.
2. Ứng dụng gọi REST API `POST /video/upload` (hoặc `upload-async` tùy cấu hình) kèm file và các tham số (tuổi).
3. Hiển thị trạng thái tiến trình loading.
4. Nhận kết quả toàn diện, hiển thị tóm tắt, biểu đồ BVP, và tự động load lại danh sách lịch sử.

### 3.3. Luồng Chatbot (RAG)
1. Component ChatBot render một icon nổi ở góc dưới.
2. Khi mở hộp thoại, người dùng nhập câu hỏi.
3. Ứng dụng gửi request qua `chatApi.ts` tới `POST /chat`.
4. Nhận được câu trả lời và thông tin tài liệu tham chiếu (sources), hiển thị vào khung hội thoại.

## 4. Hướng dẫn Phát triển

- **Chạy local**: Dùng `npm run dev` (sẽ chạy ở port 3002).
- **Cấu hình môi trường**: Dùng file `.env` để cấu hình `VITE_API_URL` (cho HTTP API) và `VITE_WS_URL` (cho WebSocket).
- **UI/UX**: Hãy tiếp tục duy trì nguyên tắc thiết kế tối giản, dark mode tương thích, và tránh sử dụng placeholder không cần thiết. Thêm hiệu ứng micro-animations khi có tương tác (ví dụ: gửi tin nhắn chatbot, upload file).
