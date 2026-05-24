import { useCallback, useRef, useState } from 'react'

const CAPTURE_FPS = 30 // Gửi 30fps để khớp với backend settings.fps = 30

// Hook quản lý truy cập webcam và bắt (capture) khung hình để gửi qua backend
export function useWebcam(onFrame) {
  const videoRef = useRef(null)          // Tham chiếu đến thẻ <video>
  const captureCanvas = useRef(document.createElement('canvas')) // Canvas ẩn để vẽ frame
  const streamRef = useRef(null)         // Giữ MediaStream để dễ dàng stop()
  const timerRef = useRef(null)          // Interval timer bắt frame
  const [active, setActive] = useState(false) // Trạng thái webcam đang mở hay đóng
  const [error, setError] = useState(null)    // Lỗi nếu không truy cập được webcam

  // Hàm bắt đầu mở webcam và kích hoạt quá trình chụp frame liên tục
  const start = useCallback(async () => {
    try {
      // Xin quyền truy cập webcam với độ phân giải tiêu chuẩn 640x480
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

      // Bắt đầu interval chụp frame
      const interval = Math.round(1000 / CAPTURE_FPS)
      timerRef.current = setInterval(() => {
        const video = videoRef.current
        if (!video || video.readyState < 2) return

        const canvas = captureCanvas.current
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d')?.drawImage(video, 0, 0)
        
        // Chuyển frame thành base64 JPEG (chất lượng 0.7 để giảm dung lượng mạng)
        const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
        onFrame(base64)
      }, interval)
    } catch {
      setError('Không thể truy cập camera. Vui lòng cho phép quyền trong trình duyệt.')
    }
  }, [onFrame])

  // Hàm dừng webcam và dọn dẹp interval
  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }, [])

  return { videoRef, active, error, start, stop }
}
