import type { ChatResponse } from '@/types/chat'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8001'

export async function sendChatMessage(question: string): Promise<ChatResponse> {
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
