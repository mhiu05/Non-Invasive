import axios from 'axios'
import type { HistoryRecord, VideoResult } from '@/types/vitals'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8001',
  timeout: 120_000,
})

export async function uploadVideo(
  file: File,
  age?: number,
  onProgress?: (pct: number) => void,
): Promise<VideoResult> {
  const form = new FormData()
  form.append('file', file)
  if (age != null) {
    form.append('age', age.toString())
  }

  const { data } = await http.post<VideoResult>('/video/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

export async function fetchHistory(): Promise<HistoryRecord[]> {
  const { data } = await http.get<HistoryRecord[]>('/history')
  return data
}

export async function checkHealth() {
  const { data } = await http.get('/health')
  return data as { status: string; model_loaded: boolean; device: string }
}
