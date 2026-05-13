export interface FaceMessage {
  type: 'face'
  detected: boolean
  bbox: [number, number, number, number] | null  // [x, y, w, h] in original frame coords
}

export interface VitalsMessage {
  type: 'vitals'
  heart_rate: number
  blink_rate: number
  snr_db: number
  bvp_window: number[]
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
}
