import { useState, useRef, useEffect, useCallback } from 'react'
import { MessageCircle, X, Send, Bot, User, FileText, Loader2, Globe, BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'
import { sendChatMessage } from '@/lib/chatApi'
import type { ChatMessage } from '@/types/chat'

/* ─── Suggested quick questions ─── */
const SUGGESTIONS = [
  'Nhịp tim bao nhiêu là bình thường?',
  'Pipeline xử lý video hoạt động ra sao?',
  'HRV là gì và ý nghĩa?',
  'SNR thấp có ảnh hưởng gì?',
]

/* ─── Unique ID generator ─── */
let _id = 0
const uid = () => `msg-${Date.now()}-${++_id}`

export function ChatBot() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  /* Auto-scroll to bottom */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, open])

  /* Focus input when opened */
  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const handleSend = useCallback(
    async (text?: string) => {
      const question = (text ?? input).trim()
      if (!question || loading) return

      const userMsg: ChatMessage = {
        id: uid(),
        role: 'user',
        content: question,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      setInput('')
      setLoading(true)

      try {
        const res = await sendChatMessage(question)
        const botMsg: ChatMessage = {
          id: uid(),
          role: 'bot',
          content: res.answer,
          sources: res.sources,
          fromInternalDocs: res.from_internal_docs,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, botMsg])
      } catch (err) {
        const errMsg: ChatMessage = {
          id: uid(),
          role: 'bot',
          content: `⚠️ Lỗi: ${err instanceof Error ? err.message : 'Không thể kết nối đến chatbot. Hãy chắc chắn đã chạy build_embeddings.py và cấu hình GEMINI_API_KEY.'}`,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, errMsg])
      } finally {
        setLoading(false)
      }
    },
    [input, loading],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  /* ─── Floating button (closed state) ─── */
  if (!open) {
    return (
      <button
        id="chatbot-toggle"
        onClick={() => setOpen(true)}
        className={cn(
          'fixed bottom-6 right-6 z-50',
          'flex h-14 w-14 items-center justify-center rounded-full',
          'bg-gradient-to-br from-indigo-500 to-purple-600',
          'text-white shadow-lg shadow-indigo-500/30',
          'transition-all duration-300 hover:scale-110 hover:shadow-xl hover:shadow-indigo-500/40',
          'active:scale-95',
        )}
        aria-label="Open chatbot"
      >
        <MessageCircle size={24} />
        {/* Pulse ring */}
        <span className="absolute inset-0 animate-ping rounded-full bg-indigo-400 opacity-20" />
      </button>
    )
  }

  /* ─── Chat panel (open state) ─── */
  return (
    <div
      id="chatbot-panel"
      className={cn(
        'fixed bottom-6 right-6 z-50',
        'flex flex-col',
        'w-[380px] max-w-[calc(100vw-2rem)] h-[560px] max-h-[calc(100vh-4rem)]',
        'rounded-2xl overflow-hidden',
        'bg-white dark:bg-slate-900',
        'border border-slate-200 dark:border-slate-700',
        'shadow-2xl shadow-slate-900/20 dark:shadow-black/40',
        'animate-slideUp',
      )}
    >
      {/* ── Header ── */}
      <div
        className={cn(
          'flex items-center justify-between px-5 py-4',
          'bg-gradient-to-r from-indigo-600 to-purple-600',
          'text-white',
        )}
      >
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/20 backdrop-blur-sm">
            <Bot size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold leading-tight">Health Assistant</h3>
            <p className="text-[11px] text-indigo-200">rPPG • AI-powered</p>
          </div>
        </div>
        <button
          id="chatbot-close"
          onClick={() => setOpen(false)}
          className="rounded-lg p-1.5 transition-colors hover:bg-white/20"
          aria-label="Close chatbot"
        >
          <X size={18} />
        </button>
      </div>

      {/* ── Messages area ── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scroll-smooth"
      >
        {/* Welcome message if empty */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30">
              <Bot size={32} className="text-indigo-500" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                Xin chào! 👋
              </p>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 max-w-[260px]">
                Tôi có thể giúp bạn hiểu kết quả đo sức khỏe hoặc tra cứu kiến trúc hệ thống.
              </p>
            </div>
            {/* Quick suggestions */}
            <div className="flex flex-wrap justify-center gap-1.5 mt-1">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className={cn(
                    'px-3 py-1.5 text-[11px] rounded-full',
                    'bg-slate-100 dark:bg-slate-800',
                    'text-slate-600 dark:text-slate-300',
                    'hover:bg-indigo-100 hover:text-indigo-700',
                    'dark:hover:bg-indigo-900/30 dark:hover:text-indigo-300',
                    'transition-colors border border-slate-200 dark:border-slate-700',
                  )}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat messages */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              'flex gap-2.5 animate-fadeIn',
              msg.role === 'user' ? 'flex-row-reverse' : 'flex-row',
            )}
          >
            {/* Avatar */}
            <div
              className={cn(
                'flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-white',
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-emerald-400 to-teal-500'
                  : 'bg-gradient-to-br from-indigo-500 to-purple-600',
              )}
            >
              {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
            </div>
            {/* Bubble */}
            <div
              className={cn(
                'max-w-[75%] rounded-2xl px-3.5 py-2.5 text-[13px] leading-relaxed',
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-br-md'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-bl-md',
              )}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {/* Source badge: internal docs vs Gemini */}
              {msg.role === 'bot' && msg.fromInternalDocs !== undefined && (
                <div className="mt-2">
                  {msg.fromInternalDocs ? (
                    <span
                      className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium',
                        'bg-emerald-100/80 dark:bg-emerald-900/40',
                        'text-emerald-700 dark:text-emerald-300',
                      )}
                    >
                      <BookOpen size={10} />
                      📚 Nguồn: Tài liệu nội bộ
                    </span>
                  ) : (
                    <span
                      className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium',
                        'bg-amber-100/80 dark:bg-amber-900/40',
                        'text-amber-700 dark:text-amber-300',
                      )}
                    >
                      <Globe size={10} />
                      🤖 Nguồn: Gemini AI (không có trong tài liệu nội bộ)
                    </span>
                  )}
                </div>
              )}
              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {msg.sources.map((src) => (
                    <span
                      key={src}
                      className={cn(
                        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px]',
                        'bg-indigo-100/80 dark:bg-indigo-900/40',
                        'text-indigo-600 dark:text-indigo-300',
                      )}
                    >
                      <FileText size={10} />
                      {src.split(/[/\\]/).pop()}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex gap-2.5 animate-fadeIn">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 text-white">
              <Bot size={14} />
            </div>
            <div className="rounded-2xl rounded-bl-md bg-slate-100 dark:bg-slate-800 px-4 py-3">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Loader2 size={14} className="animate-spin" />
                Đang suy nghĩ...
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Input area ── */}
      <div className="border-t border-slate-200 dark:border-slate-700 px-4 py-3">
        <div
          className={cn(
            'flex items-center gap-2 rounded-xl',
            'bg-slate-100 dark:bg-slate-800',
            'border border-slate-200 dark:border-slate-700',
            'px-3 py-2',
            'focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-400/20',
            'transition-all',
          )}
        >
          <input
            ref={inputRef}
            id="chatbot-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi về sức khỏe hoặc hệ thống..."
            disabled={loading}
            className={cn(
              'flex-1 bg-transparent text-sm outline-none',
              'text-slate-700 dark:text-slate-200',
              'placeholder:text-slate-400 dark:placeholder:text-slate-500',
              'disabled:opacity-50',
            )}
          />
          <button
            id="chatbot-send"
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-lg',
              'text-white transition-all',
              input.trim() && !loading
                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:shadow-md hover:shadow-indigo-500/30 active:scale-95'
                : 'bg-slate-300 dark:bg-slate-600 cursor-not-allowed',
            )}
            aria-label="Send message"
          >
            <Send size={15} />
          </button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-slate-400 dark:text-slate-500">
          ⚠️ Thông tin sức khỏe chỉ mang tính tham khảo
        </p>
      </div>
    </div>
  )
}
