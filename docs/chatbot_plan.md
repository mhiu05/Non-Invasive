# Chatbot Plan — Tích hợp RAG Chatbot vào Non-Invasive (rPPG)

> **Trạng thái**: ✅ Đã triển khai hoàn chỉnh (backend + frontend).
> **LLM**: Google Gemini (thay vì OpenAI như kế hoạch ban đầu).
> **Vector Store**: FAISS (local, không cần API key).
> **Tài liệu**: PDF/Markdown/Text trong `backend/app/documents/`.

---

## 1. Mục tiêu

Tích hợp chatbot RAG (Retrieval-Augmented Generation) vào hệ thống Non-Invasive, giúp:

- **Người dùng cuối**: Hiểu kết quả đo (HR, HRV, Blink, SNR), nhận lời khuyên sức khỏe cơ bản.
- **Developer / QA**: Tra cứu kiến trúc, pipeline xử lý tín hiệu, model đang dùng.
- **Product**: Cảnh báo khi chỉ số bất thường, hướng dẫn cải thiện chất lượng đo.

---

## 2. Kiến trúc đã triển khai

```
User (React Frontend — ChatBot.tsx)
    │  POST /chat  {question}
    ▼
FastAPI Backend  ── backend/app/api/routes/chat.py
    │
    ├── Embeddings: HuggingFace all-MiniLM-L6-v2
    ├── Vector Store: FAISS  (file: backend/vectorstore/faiss_index/)
    │       └── built from: backend/app/documents/ (PDFs, .md, .txt)
    ├── Retriever: similarity search, top-k=10
    ├── LLM: Google Gemini (gemini-2.5-flash)
    └── RAG Chain: LangChain create_retrieval_chain
```

---

## 3. Cấu trúc file đã tạo

```
backend/
├── app/
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── loader.py         ← load & split PDF/Markdown/Text
│   │   ├── vectorstore.py    ← build & load FAISS index
│   │   └── engine.py         ← RAG chain (Gemini + FAISS)
│   ├── documents/
│   │   └── Medical_book.pdf  ← tài liệu chatbot học
│   └── api/
│       └── routes/
│           └── chat.py       ← FastAPI endpoint POST /chat
│
├── scripts/
│   └── build_embeddings.py   ← script chạy 1 lần để build vectorstore
│
└── vectorstore/
    └── faiss_index/          ← FAISS index (generated, gitignored)

frontend/
└── src/
    ├── components/
    │   └── ChatBot.tsx        ← UI chat widget (floating button)
    ├── lib/
    │   └── chatApi.ts         ← gọi POST /chat
    └── types/
        └── chat.ts            ← types cho ChatMessage, ChatResponse
```

---

## 4. Biến môi trường

```ini
# backend/.env
GEMINI_API_KEY=your_gemini_api_key_here
CHATBOT_MODEL=gemini-2.5-flash
VECTORSTORE_PATH=vectorstore/faiss_index
```

---

## 5. Cách sử dụng

### Build vectorstore (lần đầu)

```bash
cd backend
python scripts/build_embeddings.py
```

### Test endpoint

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Nhịp tim bao nhiêu là bình thường?"}'
```

### Thêm tài liệu mới

1. Đặt file PDF/Markdown/Text vào `backend/app/documents/`.
2. Chạy lại `python scripts/build_embeddings.py`.
3. Restart backend (hoặc RAG chain sẽ tự reload khi server restart).

---

## 6. Packages đã cài

```
langchain>=0.3.26
langchain-community>=0.3.26
langchain-google-genai>=2.1.0
sentence-transformers>=4.1.0
faiss-cpu>=1.11.0
pypdf>=5.0.0
```

---

## 7. Kịch bản sử dụng thực tế

### Người dùng sau khi đo xong
```
User: "Nhịp tim 95 bpm có bình thường không?"
Bot:  "Nhịp tim 95 bpm nằm ở mức cao bình thường (60-100 bpm theo WHO).
       Nếu bạn vừa vận động hoặc căng thẳng, đây là bình thường. 
       Nếu duy trì > 100 bpm lúc nghỉ, nên tham khảo bác sĩ.
       ⚠️ Thông tin chỉ mang tính tham khảo."
Sources: [Medical_book.pdf]
```

### Developer tra cứu pipeline
```
User: "Pipeline xử lý video upload hoạt động ra sao?"
Bot:  "Video upload đi qua preprocessor.py → face_detector.py (MediaPipe)
       → rppg_engine.py (ONNX) → signal_processor.py (bandpass, FFT peak).
       Kết quả được lưu vào history.db."
Sources: [Medical_book.pdf]
```

---

## 8. Cải tiến tương lai

- [ ] Thêm conversation history (multi-turn chat).
- [ ] Truyền context vitals hiện tại (HR, SNR) vào prompt.
- [ ] Auto-rebuild vectorstore khi tài liệu thay đổi.
- [ ] Rate limiting cho `/chat` endpoint.
- [ ] Chuyển FAISS → Pinecone nếu cần multi-instance deployment.

---

## 9. Lưu ý quan trọng

- **Không trả lời y tế tuyệt đối**: Luôn thêm disclaimer khi reply về sức khỏe.
- **Lazy load**: RAG chain khởi tạo khi có request đầu tiên (không load lúc startup).
- **Rebuild embeddings**: Chạy lại `build_embeddings.py` mỗi khi thay đổi tài liệu.
- **API Key**: Không commit `GEMINI_API_KEY` vào git — đã có trong `.gitignore`.