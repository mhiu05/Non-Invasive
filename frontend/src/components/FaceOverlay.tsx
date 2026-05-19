import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>
  bbox: [number, number, number, number] | null  // coords in original video frame
  detected: boolean
  className?: string
}

/**
 * FaceOverlay — vẽ bbox lên canvas với CSS transition mượt mà.
 * Dùng requestAnimationFrame để interpolate bbox khi mặt di chuyển.
 */
export function FaceOverlay({ videoRef, bbox, detected, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Lưu bbox hiện tại đang render (để interpolate mượt)
  const currentRef = useRef<[number, number, number, number] | null>(null)
  const targetRef = useRef<[number, number, number, number] | null>(null)
  const rafRef = useRef<number | null>(null)
  // Bao nhiêu frame không có mặt → ẩn bbox
  const missRef = useRef(0)
  const MISS_FADE = 6  // ẩn sau 6 animation frames ≈ 100ms

  // Lưu state detected mới nhất để loop draw() có thể đọc được
  const detectedRef = useRef(detected)

  // Cập nhật target bbox mỗi khi server gửi về
  useEffect(() => {
    detectedRef.current = detected
    if (bbox && detected) {
      targetRef.current = bbox
      missRef.current = 0
    }
  }, [bbox, detected])

  // Vòng lặp animation: lerp currentRef → targetRef rồi vẽ
  useEffect(() => {
    const LERP = 0.25  // tốc độ theo bbox: 0 = không move, 1 = snap

    function draw() {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) { rafRef.current = requestAnimationFrame(draw); return }

      const ctx = canvas.getContext('2d')
      if (!ctx) { rafRef.current = requestAnimationFrame(draw); return }

      // Sync canvas size với video display size
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth
        canvas.height = video.clientHeight
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (!detectedRef.current) {
        missRef.current++
      }

      const target = targetRef.current
      const missedTooLong = missRef.current > MISS_FADE

      if (!target || missedTooLong) {
        // Mặt mất thật sự → fade out ngay
        currentRef.current = null
        rafRef.current = requestAnimationFrame(draw)
        return
      }

      // Lerp current → target
      if (!currentRef.current) {
        currentRef.current = [...target]
      } else {
        currentRef.current = currentRef.current.map(
          (c, i) => c + (target[i] - c) * LERP
        ) as [number, number, number, number]
      }

      const [x, y, w, h] = currentRef.current
      const scaleX = canvas.width / (video.videoWidth || 640)
      const scaleY = canvas.height / (video.videoHeight || 480)

      const cx = x * scaleX
      const cy = y * scaleY
      const cw = w * scaleX
      const ch = h * scaleY

      // Vẽ bbox
      ctx.save()
      ctx.strokeStyle = '#6366f1'
      ctx.lineWidth = 2
      ctx.shadowColor = '#818cf8'
      ctx.shadowBlur = 6
      ctx.strokeRect(cx, cy, cw, ch)
      ctx.restore()

      // Label
      ctx.fillStyle = '#6366f1'
      ctx.font = '11px system-ui, sans-serif'
      ctx.fillText('Face', cx + 4, cy - 5)

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [videoRef])

  // Hiển thị "No face" chỉ khi mặt mất lâu (tránh nhấp nháy)
  const showNoFace = !detected && missRef.current > MISS_FADE

  return (
    <div className={cn('relative', className)}>
      <video
        ref={videoRef}
        className="h-full w-full rounded-xl object-cover"
        muted
        playsInline
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 rounded-xl"
      />
      {showNoFace && (
        <div className="absolute bottom-2 left-2 rounded-md bg-black/50 px-2 py-1 text-xs text-white">
          No face detected
        </div>
      )}
    </div>
  )
}
