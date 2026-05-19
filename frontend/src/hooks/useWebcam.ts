import { useCallback, useRef, useState } from 'react'

const CAPTURE_FPS = 30  // gửi 30fps để khớp với backend settings.fps = 30

export function useWebcam(onFrame: (base64jpeg: string) => void) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const captureCanvas = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      })
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setActive(true)
      setError(null)

      const interval = Math.round(1000 / CAPTURE_FPS)
      timerRef.current = setInterval(() => {
        const video = videoRef.current
        if (!video || video.readyState < 2) return

        const canvas = captureCanvas.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d')?.drawImage(video, 0, 0)
        // Chất lượng 0.7 để giảm kích thước gửi qua WS
        const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
        onFrame(base64)
      }, interval)
    } catch {
      setError('Không thể truy cập camera. Vui lòng cho phép quyền trong trình duyệt.')
    }
  }, [onFrame])

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }, [])

  return { videoRef, active, error, start, stop }
}
