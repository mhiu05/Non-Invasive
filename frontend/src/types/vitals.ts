export interface FaceMessage {
  type: 'face'
  detected: boolean
  bbox: [number, number, number, number] | null  // [x, y, w, h] in original frame coords
}

export interface VitalsMessage {
  type: 'vitals'
  heart_rate: number | null
  blink_rate: number | null
  snr_db: number | null
  bvp_window: number[]
  buffer_frames?: number
  buffer_needed?: number
  age?: number
  age_group?: string
  bandpass_low_hz?: number
  bandpass_high_hz?: number
  hrv_ms?: number | null
  sdnn_ms?: number | null
  rmssd_ms?: number | null
  pnn50?: number | null
  peak_count?: number | null
}

export interface ErrorMessage {
  type: 'error'
  message: string
}

export type WsMessage = FaceMessage | VitalsMessage | ErrorMessage

export interface VideoResult {
  filename: string
  total_frames: number
  duration_sec: number
  heart_rate: number
  blink_rate: number
  snr_db: number
  bvp_signal: number[]
  age?: number
  age_group?: string
  bandpass_low_hz?: number
  bandpass_high_hz?: number
  hrv_ms?: number | null
  sdnn_ms?: number | null
  rmssd_ms?: number | null
  pnn50?: number | null
  peak_count?: number | null
}

export interface HistoryRecord {
  id: string
  created_at: string
  type: 'video' | 'realtime'
  filename?: string | null
  session_id?: string | null
  duration_sec?: number | null
  heart_rate?: number | null
  blink_rate?: number | null
  snr_db?: number | null
  age?: number | null
  age_group?: string | null
  bandpass_low_hz?: number | null
  bandpass_high_hz?: number | null
  hrv_ms?: number | null
  sdnn_ms?: number | null
  rmssd_ms?: number | null
  pnn50?: number | null
  peak_count?: number | null
  result?: Record<string, unknown> | null
}

/* ── Async job types ── */
export interface AsyncJobResponse {
  job_id: string
  status: 'pending' | 'running' | 'done' | 'failed'
}

export interface AsyncJobStatus {
  job_id: string
  status: 'pending' | 'running' | 'done' | 'failed'
  result: VideoResult | null
  error: string | null
  updated_at: string
}

/* ── History filter params ── */
export interface HistoryFilters {
  type?: string
  start_at?: string
  end_at?: string
  limit?: number
  offset?: number
}

/* ── SNR quality helpers ── */
export type SnrQuality = 'excellent' | 'good' | 'fair' | 'poor'

export function getSnrQuality(snr: number | null | undefined): SnrQuality {
  if (snr == null) return 'poor'
  if (snr >= 5) return 'excellent'
  if (snr >= 2) return 'good'
  if (snr >= 0) return 'fair'
  return 'poor'
}

export function getSnrLabel(q: SnrQuality): string {
  switch (q) {
    case 'excellent': return 'Xuất sắc'
    case 'good':      return 'Tốt'
    case 'fair':      return 'Trung bình'
    case 'poor':      return 'Yếu'
  }
}

export function getSnrColor(q: SnrQuality) {
  switch (q) {
    case 'excellent': return { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-300', ring: 'ring-emerald-500/30' }
    case 'good':      return { bg: 'bg-green-100 dark:bg-green-900/30',     text: 'text-green-700 dark:text-green-300',     ring: 'ring-green-500/30' }
    case 'fair':      return { bg: 'bg-amber-100 dark:bg-amber-900/30',     text: 'text-amber-700 dark:text-amber-300',     ring: 'ring-amber-500/30' }
    case 'poor':      return { bg: 'bg-red-100 dark:bg-red-900/30',         text: 'text-red-700 dark:text-red-300',         ring: 'ring-red-500/30' }
  }
}
