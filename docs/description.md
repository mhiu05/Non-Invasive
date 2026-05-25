# Kiến trúc hệ thống: Non-Invasive Health Analysis System

Dự án **Non-Invasive Health Analysis System** là một hệ thống toàn diện kết hợp Computer Vision (rPPG), Xử lý tín hiệu, và Xử lý ngôn ngữ tự nhiên (Advanced RAG) để đo lường các chỉ số sức khỏe từ video khuôn mặt và cung cấp trợ lý y tế ảo. 

Hệ thống được thiết kế linh hoạt và chia thành 4 phân lớp chính:

## 1. Frontend Layer
- **Công nghệ:** React, Vite.
- **Nhiệm vụ:** Cung cấp giao diện người dùng tương tác, mượt mà.
- **Thành phần chính:** 
  - Giao diện đo trực tiếp (Live Webcam) truyền frames liên tục qua WebSocket.
  - Giao diện tải video lên (Offline Upload) giao tiếp qua HTTP REST API.
  - Dashboard thống kê lịch sử đo (Charts, Metrics).
  - Cửa sổ Chatbot tích hợp AI để hỏi đáp về sức khỏe dựa trên kết quả đo.

## 2. Backend Layer (FastAPI)
- **Công nghệ:** FastAPI (Python), WebSocket, LangChain.
- **Nhiệm vụ:** Xử lý logic nghiệp vụ, quản lý luồng dữ liệu thời gian thực và bất đồng bộ.
- **Thành phần chính:**
  - **REST API & WebSocket Server:** Nhận request tải file và stream frames từ client.
  - **rPPG Engine Pipeline:** 
    - Real-time Engine (xử lý khung hình nhận từ WebSocket).
    - Async Offline Engine (xử lý video được tải lên chạy nền).
    - Signal Processor (Áp dụng thuật toán FFT, bộ lọc Bandpass Filter để trích xuất Heart Rate, HRV, SNR).
  - **Chat Agent (RAG):**
    - Quản lý hội thoại, Context Window.
    - **Hybrid Retriever:** Kết hợp Query Rewriting (viết lại truy vấn), Vector Search (từ điển Dense bằng FAISS) + Keyword Search (từ điển Sparse bằng BM25) và Re-ranking (sử dụng Cross-Encoder) để chọn ra ngữ cảnh y khoa chính xác nhất.

## 3. ML Models & AI Services Layer
- **Công nghệ:** PyTorch, ONNX, MediaPipe, Sentence-Transformers, Google Gemini.
- **Nhiệm vụ:** Thực thi các tác vụ học máy nặng (Computer Vision và NLP).
- **Thành phần chính:**
  - **Vision & rPPG Models:** 
    - *MediaPipe Face Mesh:* Trích xuất tọa độ khuôn mặt (Landmarks) siêu nhẹ với tốc độ cao để lấy vùng ROI (vùng da).
    - *ONNX Runtime:* Chạy các mô hình Deep Learning rPPG (DeepPhys, FactorizePhys, PhysFormer, EfficientPhys) đã được tối ưu hóa và xuất ra từ PyTorch.
  - **NLP Models:** 
    - *Embedding Model:* (vd: `all-MiniLM-L6-v2`) để tạo vector không gian cho tài liệu y khoa.
    - *Cross-Encoder:* (vd: `ms-marco-MiniLM-L-6-v2`) dùng cho bước Re-ranking sắp xếp lại kết quả tìm kiếm theo độ liên quan ngữ nghĩa chính xác tuyệt đối.
  - **LLM Provider:** 
    - *Google Gemini API (Gemini 2.5 Flash):* Tạo phản hồi tự nhiên, chuyên sâu cho chatbot dựa trên ngữ cảnh y khoa cung cấp từ Retriever.

## 4. Data Layer
- **Công nghệ:** SQLite, FAISS, Cấu trúc Pickle, File System.
- **Nhiệm vụ:** Lưu trữ dữ liệu lịch sử hệ thống và cơ sở tri thức cho AI (Knowledge Base).
- **Thành phần chính:**
  - **Relational Database (SQLite):** Lưu trữ lịch sử đo lường (`history.db`) và trạng thái các tiến trình xử lý video (`video_jobs.db`).
  - **Vector & Sparse Indexes:** Lưu trữ bộ tìm kiếm `index.pkl` (FAISS cho Dense Retrieval) và `bm25_retriever.pkl` (cho Sparse Retrieval).
  - **Knowledge Base:** Kho lưu trữ tài liệu thô (.md, .pdf) để pipeline ingest dữ liệu nạp vào chatbot.

---

# Prompt dành cho Claude AI (Vẽ sơ đồ kiến trúc)

Bạn hãy sao chép toàn bộ đoạn prompt dưới đây (bằng tiếng Anh để AI hiểu rõ ngữ cảnh kỹ thuật nhất) và dán vào Claude AI. 
*Lưu ý: Bạn nên sử dụng **Claude 3.5 Sonnet** và đảm bảo đã bật tính năng **Artifacts** để Claude có thể tự động viết code React/TailwindCSS tạo ra một sơ đồ cực kỳ bắt mắt tương tự như phong cách hình ảnh tham khảo của bạn.*

```text
Act as an expert software architect and frontend UI/UX developer. I want you to create a highly detailed, visually stunning architecture diagram using React and Tailwind CSS in a Claude Artifact.

The design should closely mimic the visual style of modern dark-mode architecture diagrams (like the NexusRAG architecture style), featuring a dark grid background, glowing dashed borders for container regions, neon color accents (cyan, purple, orange, green), and Lucide-react icons. 

The architecture is for a "Non-Invasive Health Analysis System". Please divide the diagram horizontally into 4 main vertical columns/layers:

1. **Frontend** (Cyan glow border)
   - Include a "Browser/Client" icon outside pointing in.
   - Contains a main box: "React + Vite" (handles Chat UI, Video Upload, Live Webcam, Dashboard).
   - Arrows from React point to the Backend via "HTTP (Upload/Chat)" and "WebSocket (Live Frames)".

2. **Backend - FastAPI** (Teal/Blue glow border)
   - Top box: "FastAPI" (REST API + WebSocket Streaming).
   - Below FastAPI, split into two main downstream functional workflows:
     a) **rPPG Pipeline**: Consists of "Real-time & Offline Engines" pointing to "Signal Processing (FFT, Bandpass)".
     b) **Chat Agent**: Consists of "LangChain Agent" pointing to "Hybrid Retriever (FAISS + BM25 + Query Rewriting)".

3. **ML Models & AI Services** (Purple glow border)
   - Box 1: **Vision & rPPG** (MediaPipe Face Mesh, ONNX Runtime: DeepPhys, PhysFormer). Arrow coming from the rPPG Pipeline.
   - Box 2: **NLP Models** (HuggingFace Embeddings, Cross-Encoder Re-ranker). Arrow coming from Hybrid Retriever.
   - Box 3: **LLM Provider** (Google Gemini 2.5 Flash API). Arrow coming from Chat Agent.

4. **Data Layer** (Orange/Yellow glow border)
   - Box 1: **SQLite** (Stores Measurement History & Async Video Jobs). Arrow coming from Signal Processing and Backend.
   - Box 2: **Vector & Sparse Store** (FAISS Dense Index + BM25 Sparse Index). Arrow coming from NLP Models / Retriever.
   - Box 3: **Knowledge Base** (Medical Documents: PDF, MD, TXT).

**Design Requirements:**
- Use modern Tailwind utility classes (e.g., bg-slate-900, backdrop-blur, border-dashed, ring, glow effects via box-shadow).
- Draw logical connection arrows (you can use SVG lines layered behind or absolute positioned CSS lines/arrows) between these components to strictly show the data flow.
- Make the UI responsive, extremely polished, with smooth scale-up hover effects on the component boxes.
- Include a bold title at the very top: "Non-Invasive Health Analysis Architecture".
- Do not use placeholder text, make the diagram complete and readable directly in the Artifact window.
```
