# Chatbot Plan — Non-Invasive RAG Chatbot

## Mục tiêu

Tài liệu này mô tả kiến trúc hiện tại của chatbot trong dự án Non-Invasive và hướng dẫn cách mở rộng để:

- dùng hiệu quả tài liệu nội bộ trong `backend/app/documents/`
- xây dựng và cập nhật FAISS vectorstore
- đảm bảo nguồn tham khảo rõ ràng trong câu trả lời
- hướng đến mở rộng tìm kiếm web và self-learning trong tương lai

---

## 1. Kiến trúc hiện tại

### 1.1. Dòng dữ liệu chính

1. Người dùng gửi câu hỏi đến API chatbot.
2. Backend gọi `app.chatbot.engine.ask(question)`.
3. `engine.py` lazy-load RAG chain nếu chưa khởi tạo.
4. RAG chain sử dụng:
   - FAISS vectorstore cục bộ trên `backend/vectorstore/faiss_index/`
   - embeddings từ `sentence-transformers/all-MiniLM-L6-v2`
   - Google Gemini qua `langchain_google_genai`
5. Kết quả trả về gồm `answer` và danh sách `sources`.

### 1.2. Các module chính

- `backend/app/chatbot/loader.py`
  - nạp tài liệu từ `backend/app/documents/`
  - hỗ trợ `.md`, `.txt`, `.pdf`
  - chia tài liệu thành chunk với `chunk_size=500`, `chunk_overlap=50`
  - thanh lọc và giữ metadata `source` cho mỗi chunk
- `backend/app/chatbot/vectorstore.py`
  - xây, load và cập nhật FAISS index
  - hỗ trợ rebuild và incremental update
  - `VECTORSTORE_PATH` có thể cấu hình bằng biến môi trường
- `backend/scripts/build_embeddings.py`
  - nạp tài liệu, chia chunk, tạo embedding, lưu FAISS
  - chạy khi tài liệu nội bộ thay đổi
  - hỗ trợ `--force` để rebuild sạch
- `backend/scripts/rebuild_embeddings.py`
  - rebuild toàn bộ FAISS index dùng helper chung
- `backend/app/chatbot/engine.py`
  - tạo retriever từ FAISS
  - cấu hình LLM Gemini với prompt hệ thống
  - trả về answer và sources từ metadata tài liệu
- `backend/app/chatbot/feedback_store.py`
  - lưu câu hỏi/answer/sources/rating để mở rộng self-learning
- `backend/app/chatbot/ingest.py`
  - ingest nội dung mới vào `backend/app/documents/`
- `backend/app/chatbot/auto_update.py`
  - helper rebuild index dùng chung cho script và automation

### 1.3. Cách dùng hiện tại

```bash
cd backend
python scripts/build_embeddings.py --force
```

Hoặc dùng helper rebuild:

```bash
cd backend
python scripts/rebuild_embeddings.py
```

Sau khi thêm hoặc sửa tài liệu trong `backend/app/documents/`, cần chạy lại một trong hai script này và khởi động lại backend.

---

## 2. Chức năng hiện tại

### 2.1. Chatbot đang làm được

- trả lời câu hỏi dựa trên nội dung trong `backend/app/documents/`
- tìm tài liệu liên quan bằng FAISS similarity search
- bổ sung cảnh báo y tế trong prompt khi câu hỏi liên quan sức khỏe
- trả về nguồn tham khảo qua `sources`

### 2.2. Hạn chế hiện tại

- không có tìm kiếm web, chỉ dùng dữ liệu nội bộ
- chưa có cơ chế feedback / self-learning hoàn chỉnh, nhưng đã có module lưu feedback và API feedback
- đã có helper rebuild index; incremental update FAISS còn cần kiểm soát duplicate
- nếu tài liệu nội bộ thiếu, bot trả "không tìm thấy"

---

## 3. Làm sao để phù hợp với dự án hiện tại

### 3.1. Cải thiện nguồn tài liệu

- tập trung vào `backend/app/documents/` như nguồn chính
- giữ tài liệu rõ ràng, có cấu trúc, dễ đọc bằng OCR/Markdown
- thêm file `README`, `architecture.md`, `medical_book.pdf`, `faq.md`...
- sau mỗi thay đổi, chạy lại `python scripts/build_embeddings.py`

### 3.2. Ghi nguồn tham khảo rõ ràng

- mỗi chunk cần metadata `source`
- gom `sources` từ các chunk top-k trả về
- trả về `sources` trong API response để frontend hiển thị

Ví dụ API response:

```json
{
  "answer": "...",
  "sources": ["Medical_book.pdf", "architecture.md"]
}
```

### 3.3. Mở rộng tìm kiếm web (future)

Đây là bước mở rộng, không phải tính năng hiện tại.

- thêm module web search (Google Custom Search, Bing Search, SERP API)
- scrape/thu thập nội dung từ URL trả về
- chuyển nội dung web thành document chunks, tạo embeddings
- kết hợp truy vấn local FAISS và web search
- luôn kèm nguồn URL khi trả lời web

Lưu ý:
- ưu tiên nội dung nội bộ
- nếu dùng nội dung web, cần disclaimer rõ ràng
- không để bot trả lời bằng web nếu không có nguồn xác thực

### 3.4. Self-learning và auto-update (future)

- lưu trữ lịch sử câu hỏi/answer/feedback
- chỉ dùng dữ liệu xác thực để học thêm
- tạo pipeline ingest để thêm nội dung tốt vào corpus
- rebuild index định kỳ hoặc khi có tài liệu mới

Hiện đã bổ sung module:
- `backend/app/chatbot/feedback_store.py`
- `backend/app/chatbot/ingest.py`
- `backend/app/chatbot/auto_update.py`
- `backend/scripts/rebuild_embeddings.py`

---

## 4. Đề xuất roadmap

### Stage 1: Ổn định hiện tại

- [ ] kiểm tra `build_embeddings.py` chạy đúng
- [ ] xác thực FAISS index tồn tại
- [ ] đảm bảo `ask()` trả về `answer` và `sources`

### Stage 2: Nguồn tham khảo rõ ràng

- [ ] chuẩn hóa metadata `source`
- [ ] hiển thị source trong frontend
- [ ] bổ sung logging câu hỏi + sources

### Stage 3: Bổ sung web search

- [ ] thêm module tìm kiếm web
- [ ] thêm xử lý nội dung web thành document
- [ ] fallback khi nội dung nội bộ không có

### Stage 4: Self-learning có kiểm soát

- [ ] thu thập feedback user
- [ ] lọc dữ liệu tốt
- [ ] cập nhật index định kỳ
- [ ] review con người cho nội dung y tế

---

## 5. Tài liệu tham khảo

- `backend/app/chatbot/engine.py`
- `backend/app/chatbot/loader.py`
- `backend/app/chatbot/vectorstore.py`
- `backend/scripts/build_embeddings.py`
- `backend/app/documents/`

---

## 6. Ghi chú quan trọng

- Chatbot hiện tại vẫn là RAG cục bộ, dựa trên tài liệu nội bộ.
- Tìm kiếm web và self-learning chưa có sẵn trong mã nguồn.
- Với dữ liệu y tế, luôn cần cảnh báo và kiểm tra nguồn.
