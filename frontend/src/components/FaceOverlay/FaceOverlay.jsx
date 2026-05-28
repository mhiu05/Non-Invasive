import { useEffect, useRef } from 'react'
import './FaceOverlay.css'

// Vẽ bounding box quanh khuôn mặt được phát hiện trên video webcam
export function FaceOverlay({ videoRef, bbox, detected, className }) {
  const canvasRef = useRef(null)
  const currentRef = useRef(null)
  const targetRef = useRef(null)
  const rafRef = useRef(null)
  const missRef = useRef(0)
  const MISS_FADE = 6

  const detectedRef = useRef(detected)

  useEffect(() => {
    detectedRef.current = detected
    if (bbox && detected) {
      targetRef.current = bbox
      missRef.current = 0
    }
  }, [bbox, detected])

  useEffect(() => {
    const LERP = 0.25

    function draw() {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) { rafRef.current = requestAnimationFrame(draw); return }

      const ctx = canvas.getContext('2d')
      if (!ctx) { rafRef.current = requestAnimationFrame(draw); return }

      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth
        canvas.height = video.clientHeight
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (!detectedRef.current) missRef.current++

      const target = targetRef.current
      const missedTooLong = missRef.current > MISS_FADE

      if (!target || missedTooLong) {
        currentRef.current = null
        rafRef.current = requestAnimationFrame(draw)
        return
      }

      if (!currentRef.current) {
        currentRef.current = [...target]
      } else {
        currentRef.current = currentRef.current.map((c, i) => c + (target[i] - c) * LERP)
      }

      const [x, y, w, h] = currentRef.current
      const scaleX = canvas.width / (video.videoWidth || 640)
      const scaleY = canvas.height / (video.videoHeight || 480)
      const cx = x * scaleX
      const cy = y * scaleY
      const cw = w * scaleX
      const ch = h * scaleY

      // Glow rectangle
      ctx.save()
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)'
      ctx.lineWidth = 1.8
      ctx.shadowColor = '#22d3ee'
      ctx.shadowBlur = 12
      ctx.strokeRect(cx, cy, cw, ch)
      ctx.restore()

      // Corner accents
      const cornerLen = Math.min(cw, ch) * 0.18
      ctx.save()
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2.4
      ctx.lineCap = 'round'
      ctx.shadowColor = '#22d3ee'
      ctx.shadowBlur = 10
      // top-left
      ctx.beginPath()
      ctx.moveTo(cx, cy + cornerLen)
      ctx.lineTo(cx, cy)
      ctx.lineTo(cx + cornerLen, cy)
      // top-right
      ctx.moveTo(cx + cw - cornerLen, cy)
      ctx.lineTo(cx + cw, cy)
      ctx.lineTo(cx + cw, cy + cornerLen)
      // bottom-right
      ctx.moveTo(cx + cw, cy + ch - cornerLen)
      ctx.lineTo(cx + cw, cy + ch)
      ctx.lineTo(cx + cw - cornerLen, cy + ch)
      // bottom-left
      ctx.moveTo(cx + cornerLen, cy + ch)
      ctx.lineTo(cx, cy + ch)
      ctx.lineTo(cx, cy + ch - cornerLen)
      ctx.stroke()
      ctx.restore()

      // Label
      ctx.fillStyle = 'rgba(34, 211, 238, 0.95)'
      ctx.font = '600 10px "JetBrains Mono", ui-monospace, monospace'
      ctx.fillText('FACE · LOCKED', cx + 4, cy - 6)

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [videoRef])

  const showNoFace = !detected && missRef.current > MISS_FADE

  return (
    <div className={`face-overlay-wrapper ${className || ''}`}>
      <video ref={videoRef} className="face-overlay-video" muted playsInline />
      <canvas ref={canvasRef} className="face-overlay-canvas" />
      {showNoFace && (
        <div className="face-overlay-noface">
          <span className="face-overlay-noface-dot" />
          NO FACE DETECTED
        </div>
      )}
    </div>
  )
}
