export interface ChatMessage {
  id: string
  role: 'user' | 'bot'
  content: string
  sources?: string[]
  /** true = answer from internal RAG docs, false = Gemini general knowledge */
  fromInternalDocs?: boolean
  timestamp: Date
}

export interface ChatResponse {
  answer: string
  sources: string[]
  from_internal_docs: boolean
}
