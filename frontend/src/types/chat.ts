export interface ChatMessage {
  id: string
  role: 'user' | 'bot'
  content: string
  sources?: string[]
  timestamp: Date
}

export interface ChatResponse {
  answer: string
  sources: string[]
}
