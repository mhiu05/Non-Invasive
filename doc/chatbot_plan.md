# Chatbot Plan — Ứng dụng RAG cho sản phẩm Non-Invasive

## 1. Mục tiêu

Xây một chatbot thông minh cho sản phẩm RPPG/GPU, dùng RAG (Retrieval-Augmented Generation) để:
- Hỗ trợ người dùng cuối hiểu kết quả đo và cách cải thiện chất lượng đo.
- Đưa ra gợi ý hướng dẫn người dùng nên làm gì dựa trên các chỉ số sức khỏe thu được (ví dụ: nhịp tim, nhịp thở, chất lượng tín hiệu, HRV).
- Cung cấp lời khuyên y tế cơ bản, hiểu biết về sức khỏe và cảnh báo khi số liệu có dấu hiệu bất thường.
- Hỗ trợ developer/QA hiểu kiến trúc, cách chạy backend/frontend, lựa chọn model, và debug.
- Trả lời câu hỏi chuyên môn về hệ thống, các model rPPG, pipeline xử lý tín hiệu.

## 2. Ý tưởng RAG cho sản phẩm

RAG là giải pháp giúp chatbot không chỉ dựa vào mô hình sinh ngôn ngữ mà còn truy vấn kiến thức thực tế từ tài liệu nội bộ.

### Nguồn dữ liệu cần dùng
- `README.md`
- `doc/ARCHITECTURE.md`
- `doc/PLAN.md`
- `doc/details.md`
- Các file code chính trong `backend/app/`, `frontend/src/`, `rPPG/`
- File config và schema (`backend/app/core/config.py`, `backend/app/schemas/`, `.env.example`)
- Nội dung comment, docs, và tên model/weights trong `weights/` và `rPPG/`

### Cách hoạt động
1. Tạo embedding cho từng đoạn văn bản từ các tài liệu và code.
2. Lưu embeddings vào vector store.
3. Khi người dùng hỏi, truy vấn vector store để lấy top-k đoạn phù hợp.
4. Kết hợp đoạn truy xuất được vào prompt để LLM trả lời chính xác, có nguồn dẫn.

## 3. Kiến trúc chi tiết

### 3.1. Data ingestion

- Quét thư mục `doc/` và các file tài liệu quan trọng.
- Quét các file `*.py`, `*.md`, `*.ts`, `*.tsx` trong `backend/` và `frontend/`.
- Tách văn bản thành các chunk phù hợp (200-400 token) với ngữ cảnh rõ ràng.

### 3.2. Vector store

- Dùng `FAISS` cho local deployment nhẹ.
- Nếu cần scale cloud: `Milvus` hoặc `Pinecone`.
- Lưu metadata: file path, đoạn văn bản, block id.

### 3.3. Embeddings

- Nếu dùng OpenAI: `text-embedding-3-large`
- Nếu dùng local: `sentence-transformers/all-MiniLM-L6-v2` hoặc `intfloat/e5-large`

### 3.4. LLM

- Triển khai với một trong các lựa chọn:
  - OpenAI / Azure OpenAI
  - Local LLM như `gpt4all`, `llama.cpp`, `mistral` nếu cần offline
- Prompt template chứa:
  - Mục tiêu: trả lời chính xác, không bịa đặt
  - Nguồn dữ liệu trích xuất
  - Hạn chế: nếu không tìm thấy thông tin thì trả lời "không rõ" hoặc "tham khảo tài liệu nội bộ"

### 3.5. Backend endpoint

- Thêm endpoint mới: `POST /chat` hoặc `POST /rag/chat`
- Dữ liệu input:
  - `question`
  - `session_id` (nếu muốn giữ conversation state)
  - `history` (các câu hỏi trước đó)
- Response:
  - `answer`
  - `sources`: list tài liệu/trích dẫn
  - `retrieved_chunks`
  - `confidence` (nếu cần)

### 3.6. Frontend UI

- Thêm component chatbot:
  - Khung chat đơn giản tại `src/components/`
  - Trang hoặc modal chat ở `src/pages/`
- UI hiển thị:
  - Câu hỏi của user
  - Trả lời của chatbot
  - Nguồn tham khảo (ví dụ: `ARCHITECTURE.md`, `PLAN.md`, `backend/app/services/rppg_engine.py`)

## 4. Kịch bản sử dụng

### 4.1. Người dùng cuối
- "Chatbot ơi, kết quả HR 83 có chính xác không?"
- "Nhịp tim 83 bpm có bình thường không?"
- "Tôi nên làm gì nếu kết quả nhịp tim cao hoặc dao động lớn?"
- "Tín hiệu yếu, tôi cần điều chỉnh ánh sáng hoặc vị trí thế nào?"
- "Có lời khuyên sức khỏe nào khi HRV thấp hoặc nhịp thở không ổn định?"

### 4.2. Developer/QA
- "Backend đang dùng model nào?"
- "Pipeline xử lý video upload hoạt động ra sao?"
- "Làm sao để chuyển sang FactorizePhys?"
- "Cần thay Haar Cascade bằng detector nào?"

### 4.3. Product manager
- "Sản phẩm này hỗ trợ model nào?"
- "Có kế hoạch thêm nhịp thở không?"
- "Dữ liệu nào cần lưu vào history?"

## 5. Chi tiết triển khai RAG vào sản phẩm

### 5.1. Giai đoạn 1: Proof of concept

- Tạo thư mục `backend/app/chatbot/` hoặc `backend/app/api/routes/chat.py`
- Cài package cần thiết:
  - `langchain`
  - `faiss-cpu`
  - `sentence-transformers`
  - `openai` (hoặc `llama-cpp-python` nếu local)
- Viết script dựng vector database từ `doc/` và `backend/`.
- Triển khai endpoint `POST /chat` trả về answer + source.
- Tạo frontend đơn giản để gửi câu hỏi và nhận câu trả lời.

### 5.2. Giai đoạn 2: Tối ưu retrieval

- Xây `DocumentLoader` cho:
  - Markdown docs
  - Python/TypeScript code
  - JSON config nếu cần
- Dùng `recursive character splitter` để tách code/document theo function/class.
- Tối ưu top-k retrieval, dùng `max_token_limit` để tránh quá dài.
- Thêm kỹ thuật `re-ranking` nếu cần: lấy 20 chunk, sau đó chọn 5 chunk tốt nhất.

### 5.3. Giai đoạn 3: Conversation memory

- Lưu lịch sử ngắn hạn trong session:
  - `question` + `answer`
  - `retrieved_sources`
- Khi người dùng hỏi tiếp tục, ghép `history` vào prompt để chatbot giữ ngữ cảnh.

### 5.4. Giai đoạn 4: Thực tế sản phẩm

- Tích hợp vào app frontend hiện tại như một feature hỗ trợ.
- Thêm nút "Ask bot" khi user ở trang Upload hoặc Home.
- Cần hạn chế scope: focus vào câu hỏi liên quan code / docs / product, tránh trả lời y tế tuyệt đối.

## 6. Cấu trúc prompt mẫu

```text
You are a health-aware assistant for a remote PPG health analysis application.
Use only referenced internal documentation and code snippets.
Provide practical suggestions based on the reported vital sign metrics.
If the answer involves health advice, keep it general and add a disclaimer that it is for reference only, not a medical diagnosis.
If the answer is not available from the provided sources, say "Không có thông tin trong tài liệu nội bộ".

Question:
{question}

Relevant documents:
{retrieved_chunks}

Answer:
```

## 7. Các thành phần cần triển khai cụ thể

### Backend
- `backend/app/chatbot/index.py`
- `backend/app/chatbot/loader.py`
- `backend/app/chatbot/retriever.py`
- `backend/app/chatbot/engine.py`
- `backend/app/api/routes/chat.py`

### Frontend
- `frontend/src/components/ChatBot.tsx`
- `frontend/src/pages/Chat.tsx` hoặc `frontend/src/components/ChatModal.tsx`
- `frontend/src/lib/chatApi.ts`
- `frontend/src/types/chat.ts`

### Devops / Tooling
- `scripts/build_embeddings.py`
- `scripts/update_chat_vectorstore.py`
- `.env.example` thêm biến:
  - `OPENAI_API_KEY`
  - `CHATBOT_MODEL`
  - `VECTORSTORE_PATH`

## 8. Lộ trình triển khai đề xuất

1. Tạo endpoint `POST /chat` và cấu trúc trả lời cơ bản.
2. Build vector store từ `doc/` và `backend/`.
3. Thử nghiệm với vài câu hỏi kỹ thuật.
4. Làm frontend chat nhỏ để demo.
5. Thêm trace source và kiểm soát hallucinaton.
6. Mở rộng sang dữ liệu `history` và `logs` nếu cần.

## 9. Lưu ý quan trọng

- Chatbot không phải là chuyên gia y tế. Khi trả lời về sức khỏe, cần thêm cảnh báo "Thông tin mang tính chất tham khảo".
- Đối với câu hỏi sức khỏe, chatbot nên cung cấp: tình trạng chung, lời khuyên chăm sóc tại nhà (nghỉ ngơi, uống nước, kiểm tra ánh sáng, v.v.), đồng thời khuyến nghị gặp bác sĩ nếu có dấu hiệu nghiêm trọng.
- Ưu tiên trả lời chính xác từ nguồn nội bộ; không tự tạo thông tin.
- Cập nhật lại vector store khi tài liệu/code thay đổi.
- Nếu dùng OpenAI, cần bảo mật API key và không đưa dữ liệu nhạy cảm vào prompt.

---

## 10. Tóm tắt

Ứng dụng RAG sẽ biến `Non-Invasive` thành một sản phẩm có trợ lý nội bộ, giúp cả user và developer nhanh chóng truy vấn kiến thức về hệ thống, model, pipeline và cách sử dụng. Bắt đầu bằng một chatbot backend đơn giản, sau đó mở rộng dần với memory, source citation và interface trên frontend.