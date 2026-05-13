import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>
  bbox: [number, number, number, number] | null  // coords in original video frame
  detected: boolean
  className?: string
}

export function FaceOverlay({ videoRef, bbox, detected, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!bbox || !detected) return

    // Scale bbox from video's native resolution to canvas display size
    const scaleX = canvas.width / (video.videoWidth || 640)
    const scaleY = canvas.height / (video.videoHeight || 480)
    const [x, y, w, h] = bbox

    ctx.strokeStyle = '#6366f1'
    ctx.lineWidth = 2
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY)

    // Small label
    ctx.fillStyle = '#6366f1'
    ctx.font = '12px sans-serif'
    ctx.fillText('Face', x * scaleX + 4, y * scaleY - 4)
  }, [bbox, detected, videoRef])

  const syncSize = () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (canvas && video) {
      canvas.width = video.clientWidth
      canvas.height = video.clientHeight
    }
  }

  return (
    <div className={cn('relative', className)}>
      <video
        ref={videoRef}
        className="h-full w-full rounded-xl object-cover"
        muted
        playsInline
        onLoadedMetadata={syncSize}
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 rounded-xl"
      />
      {!detected && (
        <div className="absolute bottom-2 left-2 rounded-md bg-black/50 px-2 py-1 text-xs text-white">
          No face detected
        </div>
      )}
    </div>
  )
}
