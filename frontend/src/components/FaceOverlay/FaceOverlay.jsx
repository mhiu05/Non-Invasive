import { useEffect, useRef } from 'react'
import './FaceOverlay.css'

// Component FaceOverlay vẽ khung (bounding box) quanh khuôn mặt được phát hiện
export function FaceOverlay({ videoRef, bbox, detected, className }) {
  // Tham chiếu tới thẻ canvas để vẽ khung
  const canvasRef = useRef(null)
  // Lưu tọa độ khung hiện tại đang vẽ để tạo hiệu ứng di chuyển mượt mà
  const currentRef = useRef(null)
  // Lưu tọa độ khung mục tiêu từ dữ liệu (bbox)
  const targetRef = useRef(null)
  // Tham chiếu dùng cho requestAnimationFrame
  const rafRef = useRef(null)
  // Biến đếm số lần (frame) không phát hiện được khuôn mặt
  const missRef = useRef(0)
  // Ngưỡng số frame không phát hiện mặt trước khi ẩn khung
  const MISS_FADE = 6

  // Lưu trạng thái phát hiện khuôn mặt để dùng trong hàm draw
  const detectedRef = useRef(detected)

  // Cập nhật tọa độ mục tiêu mỗi khi có dữ liệu khuôn mặt mới
  useEffect(() => {
    detectedRef.current = detected
    if (bbox && detected) {
      targetRef.current = bbox
      missRef.current = 0
    }
  }, [bbox, detected])

  // Vòng lặp chính vẽ khung lên canvas bằng requestAnimationFrame
  useEffect(() => {
    // Hệ số nội suy (Linear Interpolation) để làm mượt chuyển động của khung
    const LERP = 0.25

    function draw() {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) { rafRef.current = requestAnimationFrame(draw); return }

      const ctx = canvas.getContext('2d')
      if (!ctx) { rafRef.current = requestAnimationFrame(draw); return }

      // Đồng bộ kích thước canvas với video hiển thị thực tế
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth
        canvas.height = video.clientHeight
      }

      // Xóa canvas cũ để vẽ frame mới
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Nếu không có khuôn mặt, tăng biến đếm missRef
      if (!detectedRef.current) {
        missRef.current++
      }

      const target = targetRef.current
      const missedTooLong = missRef.current > MISS_FADE

      // Nếu không có tọa độ mục tiêu hoặc đã mất dấu khuôn mặt quá lâu
      if (!target || missedTooLong) {
        currentRef.current = null
        rafRef.current = requestAnimationFrame(draw)
        return
      }

      // Tính toán tạo độ mượt (chuyển dần từ currentRef tới target)
      if (!currentRef.current) {
        currentRef.current = [...target]
      } else {
        currentRef.current = currentRef.current.map(
          (c, i) => c + (target[i] - c) * LERP
        )
      }

      const [x, y, w, h] = currentRef.current
      // Tỷ lệ scale giữa kích thước thực của video gốc và kích thước hiển thị
      const scaleX = canvas.width / (video.videoWidth || 640)
      const scaleY = canvas.height / (video.videoHeight || 480)

      // Chuyển đổi tọa độ khung khớp với kích thước canvas hiển thị
      const cx = x * scaleX
      const cy = y * scaleY
      const cw = w * scaleX
      const ch = h * scaleY

      // Vẽ hình chữ nhật bao quanh khuôn mặt
      ctx.save()
      ctx.strokeStyle = '#6366f1'
      ctx.lineWidth = 2
      ctx.shadowColor = '#818cf8'
      ctx.shadowBlur = 6
      ctx.strokeRect(cx, cy, cw, ch)
      ctx.restore()

      // Vẽ nhãn "Face" phía trên khung
      ctx.fillStyle = '#6366f1'
      ctx.font = '11px system-ui, sans-serif'
      ctx.fillText('Face', cx + 4, cy - 5)

      rafRef.current = requestAnimationFrame(draw)
    }

    // Khởi động vòng lặp vẽ
    rafRef.current = requestAnimationFrame(draw)
    // Dọn dẹp animation frame khi component unmount
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [videoRef])

  // Biến kiểm tra nếu mất dấu quá lâu sẽ hiện thông báo
  const showNoFace = !detected && missRef.current > MISS_FADE

  return (
    <div className={`face-overlay-wrapper ${className || ''}`}>
      {/* Video dùng để hiển thị hình ảnh camera */}
      <video
        ref={videoRef}
        className="face-overlay-video"
        muted
        playsInline
      />
      {/* Canvas đè lên trên Video để vẽ khung bounding box */}
      <canvas
        ref={canvasRef}
        className="face-overlay-canvas"
      />
      {/* Thông báo hiển thị khi không tìm thấy khuôn mặt */}
      {showNoFace && (
        <div className="face-overlay-noface">
          No face detected
        </div>
      )}
    </div>
  )
}
