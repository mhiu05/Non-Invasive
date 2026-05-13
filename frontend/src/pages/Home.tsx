import { Activity, Eye, Heart, Radio, StopCircle, Video } from 'lucide-react'
import { useCallback, useEffect } from 'react'
import { BVPChart } from '@/components/BVPChart'
import { FaceOverlay } from '@/components/FaceOverlay'
import { VitalSignCard } from '@/components/VitalSignCard'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useWebcam } from '@/hooks/useWebcam'
import { useVitalsStore } from '@/store/vitalsStore'

export function Home() {
  const { heartRate, blinkRate, snrDb, bvpWindow, faceBbox, faceDetected, isConnected, reset } =
    useVitalsStore()

  const { connect, disconnect, sendFrame } = useWebSocket()
  const { videoRef, active, error, start, stop } = useWebcam(sendFrame)

  const handleStart = useCallback(async () => {
    reset()
    connect()
    await start()
  }, [reset, connect, start])

  const handleStop = useCallback(() => {
    stop()
    disconnect()
  }, [stop, disconnect])

  // Cleanup on unmount
  useEffect(() => () => { stop(); disconnect() }, [stop, disconnect])

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">

        {/* ── Left: webcam ── */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Live Analysis
            </h2>
            <span
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                isConnected
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                  : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-slate-400'}`} />
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <FaceOverlay
            videoRef={videoRef}
            bbox={faceBbox}
            detected={faceDetected}
            className="aspect-video w-full bg-slate-900 dark:bg-black rounded-xl"
          />

          {error && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
              {error}
            </p>
          )}

          <div className="flex gap-3">
            {!active ? (
              <button
                onClick={handleStart}
                className="flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-indigo-700 active:scale-95 transition"
              >
                <Video size={16} /> Start Camera
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex items-center gap-2 rounded-xl bg-red-500 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-red-600 active:scale-95 transition"
              >
                <StopCircle size={16} /> Stop
              </button>
            )}
          </div>

          <p className="text-xs text-slate-400">
            Buffer: {bvpWindow.length} / 180 frames &nbsp;·&nbsp; Model needs ~6s to compute HR
          </p>
        </div>

        {/* ── Right: vitals + chart ── */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Vital Signs
          </h2>

          <div className="space-y-3">
            <VitalSignCard icon={Heart}    label="Heart Rate"  value={heartRate}  unit="BPM"     color="red"    />
            <VitalSignCard icon={Eye}      label="Blink Rate"  value={blinkRate}  unit="bl/min"  color="blue"   />
            <VitalSignCard icon={Activity} label="Signal SNR"  value={snrDb}      unit="dB"      color="green"  />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Radio size={14} className="text-indigo-500" />
              <span className="text-xs font-medium text-slate-600 dark:text-slate-400">BVP Signal</span>
            </div>
            <div className="rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200 dark:bg-slate-800 dark:ring-slate-700">
              <BVPChart data={bvpWindow} />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
