import axios from 'axios'
import type { AsyncJobResponse, AsyncJobStatus, HistoryFilters, HistoryRecord, VideoResult } from '@/types/vitals'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8001',
  timeout: 120_000,
})

/* ── Sync upload ── */
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

/* ── Async upload ── */
export async function uploadVideoAsync(
  file: File,
  age?: number,
  onProgress?: (pct: number) => void,
): Promise<AsyncJobResponse> {
  const form = new FormData()
  form.append('file', file)
  if (age != null) {
    form.append('age', age.toString())
  }

  const { data } = await http.post<AsyncJobResponse>('/video/upload-async', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

/* ── Poll job status ── */
export async function getJobStatus(jobId: string): Promise<AsyncJobStatus> {
  const { data } = await http.get<AsyncJobStatus>(`/video/jobs/${jobId}`)
  return data
}

/* ── History with filters ── */
export async function fetchHistory(filters?: HistoryFilters): Promise<HistoryRecord[]> {
  const params: Record<string, string | number> = {}
  if (filters?.type) params.type = filters.type
  if (filters?.start_at) params.start_at = filters.start_at
  if (filters?.end_at) params.end_at = filters.end_at
  if (filters?.limit) params.limit = filters.limit
  if (filters?.offset) params.offset = filters.offset

  const { data } = await http.get<HistoryRecord[]>('/history', { params })
  return data
}

/* ── Fetch single history detail ── */
export async function fetchHistoryDetail(id: string): Promise<HistoryRecord> {
  const { data } = await http.get<HistoryRecord>(`/history/${id}`)
  return data
}

/* ── Health check ── */
export async function checkHealth() {
  const { data } = await http.get('/health')
  return data as { status: string; model_loaded: boolean; device: string }
}
