import { useState, useRef, useEffect, useCallback } from 'react'
import { MessageCircle, X, Send, Bot, User, FileText, Loader2, Globe, BookOpen, ThumbsUp, ThumbsDown } from 'lucide-react'
import { sendChatMessage, sendChatFeedback } from '@/lib/chatApi'
import './ChatBot.css'

const SUGGESTIONS = [
  'Biểu hiện của suy tim mạch là gì?',
  'HRV là gì và nó liên quan gì đến sức khỏe tim mạch?',
  'Các phương pháp sơ cứu cơ bản khi đột quỵ?',
]

let _id = 0
const uid = () => `msg-${Date.now()}-${++_id}`

// Component ChatBot quản lý cửa sổ chat trợ lý ảo
export function ChatBot() {
  // Trạng thái mở/đóng của cửa sổ chat
  const [open, setOpen] = useState(false)
  // Danh sách các tin nhắn trong đoạn chat
  const [messages, setMessages] = useState([])
  // Nội dung người dùng đang nhập
  const [input, setInput] = useState('')
  // Trạng thái chờ phản hồi từ API
  const [loading, setLoading] = useState(false)
  // Tham chiếu đến phần tử chứa danh sách tin nhắn để tự động cuộn xuống
  const scrollRef = useRef(null)
  // Tham chiếu đến ô input để tự động focus
  const inputRef = useRef(null)

  // Tự động cuộn xuống cuối cùng mỗi khi có tin nhắn mới hoặc mở cửa sổ chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, open])

  // Tự động focus vào ô input khi mở cửa sổ chat
  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  // Xử lý việc gửi tin nhắn của người dùng và nhận phản hồi từ bot
  const handleSend = useCallback(async (text) => {
    const question = (text ?? input).trim()
    if (!question || loading) return

    // Thêm tin nhắn của người dùng vào danh sách
    const userMsg = {
      id: uid(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      // Gọi API gửi tin nhắn tới server
      const res = await sendChatMessage(question)
      // Thêm phản hồi của bot vào danh sách
      const botMsg = {
        id: uid(),
        role: 'bot',
        question: question,
        content: res.answer,
        sources: res.sources,
        fromInternalDocs: res.from_internal_docs,
        timestamp: new Date(),
        feedbackStatus: null,
      }
      setMessages((prev) => [...prev, botMsg])
    } catch (err) {
      // Báo lỗi nếu việc gọi API gặp sự cố
      const errMsg = {
        id: uid(),
        role: 'bot',
        content: `⚠️ Lỗi: ${err instanceof Error ? err.message : 'Không thể kết nối đến chatbot. Hãy chắc chắn đã chạy build_embeddings.py và cấu hình GEMINI_API_KEY.'}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errMsg])
    } finally {
      setLoading(false)
    }
  }, [input, loading])

  // Xử lý gửi feedback đánh giá
  const handleFeedback = useCallback(async (msgId, rating) => {
    // Cập nhật trạng thái ngay lập tức để chặn spam click
    setMessages((prev) => 
      prev.map(m => m.id === msgId ? { ...m, feedbackStatus: rating === 1 ? 'liked' : 'disliked' } : m)
    )

    // Lấy thông tin tin nhắn từ state (lưu ý: closure có thể giữ state cũ nên dùng tham chiếu hoặc tìm lại trong ds hiện tại)
    // Nhưng vì chúng ta cần question/answer, chúng ta có thể truyền thẳng thay vì tìm
    setMessages((currentMessages) => {
      const msg = currentMessages.find(m => m.id === msgId)
      if (msg && msg.question) {
        sendChatFeedback({
          question: msg.question,
          answer: msg.content,
          sources: msg.sources || [],
          rating: rating,
        }).catch(e => console.error("Lỗi gửi feedback:", e))
      }
      return currentMessages
    })
  }, [])

  // Lắng nghe sự kiện nhấn Enter để gửi tin nhắn
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Render nút bấm để mở cửa sổ chat nếu đang đóng
  if (!open) {
    return (
      <button
        id="chatbot-toggle"
        onClick={() => setOpen(true)}
        className="chatbot-toggle-btn"
        aria-label="Open chatbot"
      >
        <MessageCircle size={24} />
        <span className="chatbot-toggle-pulse" />
      </button>
    )
  }

  return (
    <div id="chatbot-panel" className="chatbot-panel">
      {/* Header của chatbot chứa tiêu đề và nút đóng */}
      <div className="chatbot-header">
        <div className="chatbot-header-title">
          <div className="chatbot-header-icon">
            <Bot size={20} />
          </div>
          <div>
            <h3 className="chatbot-title">Health Assistant</h3>
            <p className="chatbot-subtitle">rPPG • AI-powered</p>
          </div>
        </div>
        <button
          id="chatbot-close"
          onClick={() => setOpen(false)}
          className="chatbot-close-btn"
          aria-label="Close chatbot"
        >
          <X size={18} />
        </button>
      </div>

      <div ref={scrollRef} className="chatbot-messages">
        {/* Màn hình chào mừng và các câu hỏi gợi ý nếu chưa có tin nhắn */}
        {messages.length === 0 && (
          <div className="chatbot-welcome">
            <div className="chatbot-welcome-icon">
              <Bot size={32} />
            </div>
            <div>
              <p className="chatbot-welcome-title">Xin chào! 👋</p>
              <p className="chatbot-welcome-text">
                Tôi có thể giúp bạn hiểu kết quả đo sức khỏe hoặc tra cứu kiến trúc hệ thống.
              </p>
            </div>
            <div className="chatbot-suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => handleSend(s)} className="chatbot-suggestion-btn">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Hiển thị danh sách các tin nhắn đã gửi/nhận */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chatbot-message-row ${msg.role === 'user' ? 'chatbot-message-user' : 'chatbot-message-bot'}`}
          >
            <div className={`chatbot-avatar ${msg.role === 'user' ? 'chatbot-avatar-user' : 'chatbot-avatar-bot'}`}>
              {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
            </div>
            <div className={`chatbot-bubble ${msg.role === 'user' ? 'chatbot-bubble-user' : 'chatbot-bubble-bot'}`}>
              <p className="chatbot-bubble-content">{msg.content}</p>

              {msg.role === 'bot' && msg.fromInternalDocs !== undefined && (
                <div className="chatbot-source-badge-container">
                  {msg.fromInternalDocs ? (
                    <span className="chatbot-source-internal">
                      <BookOpen size={10} /> 📚 Nguồn: Tài liệu nội bộ
                    </span>
                  ) : (
                    <span className="chatbot-source-external">
                      <Globe size={10} /> 🤖 Nguồn: Gemini AI (không có trong tài liệu nội bộ)
                    </span>
                  )}
                </div>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <div className="chatbot-sources-list">
                  {msg.sources.map((src) => (
                    <span key={src} className="chatbot-source-item">
                      <FileText size={10} /> {src.split(/[/\\]/).pop()}
                    </span>
                  ))}
                </div>
              )}
              
              {/* Cụm nút bấm Feedback (Like/Dislike) */}
              {msg.role === 'bot' && msg.content && !msg.content.startsWith('⚠️') && (
                <div className="chatbot-feedback-actions">
                  <button 
                    onClick={() => handleFeedback(msg.id, 1)}
                    className={`chatbot-feedback-btn ${msg.feedbackStatus === 'liked' ? 'active-like' : ''}`}
                    disabled={msg.feedbackStatus !== null}
                    title="Câu trả lời hữu ích"
                  >
                    <ThumbsUp size={12} />
                  </button>
                  <button 
                    onClick={() => handleFeedback(msg.id, -1)}
                    className={`chatbot-feedback-btn ${msg.feedbackStatus === 'disliked' ? 'active-dislike' : ''}`}
                    disabled={msg.feedbackStatus !== null}
                    title="Câu trả lời chưa tốt"
                  >
                    <ThumbsDown size={12} />
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Hiển thị trạng thái đang xử lý khi đợi API trả về */}
        {loading && (
          <div className="chatbot-message-row chatbot-message-bot">
            <div className="chatbot-avatar chatbot-avatar-bot">
              <Bot size={14} />
            </div>
            <div className="chatbot-bubble chatbot-bubble-bot chatbot-loading-bubble">
              <Loader2 size={14} className="chatbot-spinner" /> Đang suy nghĩ...
            </div>
          </div>
        )}
      </div>

      {/* Khu vực nhập nội dung tin nhắn và nút gửi */}
      <div className="chatbot-input-area">
        <div className="chatbot-input-wrapper">
          <input
            ref={inputRef}
            id="chatbot-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi về sức khỏe hoặc hệ thống..."
            disabled={loading}
            className="chatbot-input-field"
          />
          <button
            id="chatbot-send"
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            className={`chatbot-send-btn ${input.trim() && !loading ? 'chatbot-send-active' : 'chatbot-send-disabled'}`}
            aria-label="Send message"
          >
            <Send size={15} />
          </button>
        </div>
        <p className="chatbot-disclaimer">
          ⚠️ Thông tin sức khỏe chỉ mang tính tham khảo
        </p>
      </div>
    </div>
  )
}
