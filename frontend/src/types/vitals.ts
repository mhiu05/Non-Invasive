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
