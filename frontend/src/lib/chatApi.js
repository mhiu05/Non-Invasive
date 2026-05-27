import { http } from './api'

// Gửi tin nhắn đến API chatbot (Sử dụng RAG / Gemini)
export async function sendChatMessage(question) {
  try {
    const { data } = await http.post('/chat', { question })
    return data
  } catch (err) {
    throw new Error(err.response?.data?.detail || 'Chat request failed')
  }
}

// Gửi feedback đánh giá câu trả lời của chatbot
export async function sendChatFeedback({ question, answer, sources, rating, comment }) {
  try {
    const { data } = await http.post('/chat/feedback', {
      question,
      answer,
      sources: sources || [],
      rating,
      comment
    })
    return data
  } catch (err) {
    throw new Error(err.response?.data?.detail || 'Feedback request failed')
  }
}

