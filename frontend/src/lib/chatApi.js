const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8001'

// Gửi tin nhắn đến API chatbot (Sử dụng RAG / Gemini)
export async function sendChatMessage(question) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Chat request failed' }))
    throw new Error(err.detail || 'Chat request failed')
  }
  
  return res.json()
}

// Gửi feedback đánh giá câu trả lời của chatbot
export async function sendChatFeedback({ question, answer, sources, rating, comment }) {
  const res = await fetch(`${API_BASE}/chat/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      answer,
      sources: sources || [],
      rating,
      comment
    }),
  })
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Feedback request failed' }))
    throw new Error(err.detail || 'Feedback request failed')
  }
  
  return res.json()
}
