import { Activity, Eye, Heart, Radio, StopCircle, Video, History } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { BVPChart } from '@/components/BVPChart'
import { FaceOverlay } from '@/components/FaceOverlay'
import { VitalSignCard } from '@/components/VitalSignCard'
import { HistoryView } from '@/components/HistoryView'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useWebcam } from '@/hooks/useWebcam'
import { useVitalsStore } from '@/store/vitalsStore'

export function Live() {
  const { heartRate, blinkRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded, faceBbox, faceDetected, isConnected, reset } =
    useVitalsStore()
  const [age, setAge] = useState<number | undefined>(undefined)
  const [displayHeartRate, setDisplayHeartRate] = useState<number | null>(null)
  const [displayBlinkRate, setDisplayBlinkRate] = useState<number | null>(null)
  const [displaySnrDb, setDisplaySnrDb] = useState<number | null>(null)
  const [displayHrvMs, setDisplayHrvMs] = useState<number | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const ageRef = useCallback(() => (age != null && !Number.isNaN(age) ? age : null), [age])

  const { connect, disconnect, sendFrame } = useWebSocket(ageRef)
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

  // Smooth numeric transitions for live vitals so the display updates more like a gauge.
  const smoothingIntervalMs = 140
  const smoothingFactor = 0.07
  const smoothingThreshold = 0.25

  useEffect(() => {
    if (heartRate === null) {
      setDisplayHeartRate(null)
      return
    }

    let active = true
    const step = () => {
      setDisplayHeartRate((prev) => {
        if (prev === null) return heartRate
        const next = prev + (heartRate - prev) * smoothingFactor
        if (Math.abs(next - heartRate) < smoothingThreshold) return heartRate
        return next
      })
      if (active) {
        window.setTimeout(step, smoothingIntervalMs)
      }
    }
    step()
    return () => {
      active = false
    }
  }, [heartRate])

  useEffect(() => {
    if (blinkRate === null) {
      setDisplayBlinkRate(null)
      return
    }

    let active = true
    const step = () => {
      setDisplayBlinkRate((prev) => {
        if (prev === null) return blinkRate
        const next = prev + (blinkRate - prev) * smoothingFactor
        if (Math.abs(next - blinkRate) < smoothingThreshold) return blinkRate
        return next
      })
      if (active) {
        window.setTimeout(step, smoothingIntervalMs)
      }
    }
    step()
    return () => {
      active = false
    }
  }, [blinkRate])

  useEffect(() => {
    if (snrDb === null) {
      setDisplaySnrDb(null)
      return
    }

    let active = true
    const step = () => {
      setDisplaySnrDb((prev) => {
        if (prev === null) return snrDb
        const next = prev + (snrDb - prev) * smoothingFactor
        if (Math.abs(next - snrDb) < smoothingThreshold) return snrDb
        return next
      })
      if (active) {
        window.setTimeout(step, smoothingIntervalMs)
      }
    }
    step()
    return () => {
      active = false
    }
  }, [snrDb])

  useEffect(() => {
    if (hrvMs === null) {
      setDisplayHrvMs(null)
      return
    }

    let active = true
    const step = () => {
      setDisplayHrvMs((prev) => {
        if (prev === null) return hrvMs
        const next = prev + (hrvMs - prev) * smoothingFactor
        if (Math.abs(next - hrvMs) < smoothingThreshold) return hrvMs
        return next
      })
      if (active) {
        window.setTimeout(step, smoothingIntervalMs)
      }
    }
    step()
    return () => {
      active = false
    }
  }, [hrvMs])

  // Cleanup on unmount
  useEffect(() => () => { stop(); disconnect() }, [stop, disconnect])

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          {!showHistory ? 'Live Analysis' : 'Lịch sử đo'}
        </h2>
        {!showHistory && (
          <button type="button" onClick={() => setShowHistory(true)} className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-900">
            <History size={16} /> Xem lịch sử
          </button>
        )}
      </div>

      {showHistory ? (
        <HistoryView onBack={() => setShowHistory(false)} />
      ) : (
      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">

        {/* ── Left: webcam ── */}
        <div className="space-y-4">
          <div className="flex items-center justify-end">
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

          <div className="space-y-3">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              Tuổi người dùng
            </label>
            <input
              type="number"
              min="0"
              value={age ?? ''}
              onChange={(event) => {
                const raw = event.target.value
                const value = raw === '' ? undefined : Number(raw)
                setAge(value == null || Number.isNaN(value) ? undefined : value)
              }}
              placeholder="8"
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-indigo-400 dark:focus:ring-indigo-500/20"
            />
            <p className="text-xs text-slate-400">
              Nếu không nhập, mặc định dùng dải cho người lớn (≥ 8 tuổi).
            </p>
          </div>

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
            {heartRate !== null
              ? 'Monitoring live...'
              : bufferNeeded > 0
                ? `Buffer: ${bufferFrames} / ${bufferNeeded} frames · Model needs ~${Math.ceil(bufferNeeded / 30)}s`
                : 'Warming up...'}
          </p>
        </div>

        {/* ── Right: vitals + chart ── */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Vital Signs
          </h2>

          <div className="space-y-3">
            <VitalSignCard icon={Heart}    label="Heart Rate"     value={displayHeartRate}  unit="BPM"     color="red"    />
            <VitalSignCard icon={Eye}      label="Blink Rate"     value={displayBlinkRate}  unit="bl/min"  color="blue"   />
            <VitalSignCard icon={Activity} label="Signal SNR"     value={displaySnrDb}      unit="dB"      color="green"  />
            <VitalSignCard icon={Radio}    label="HRV (RMSSD)"    value={displayHrvMs}      unit="ms"      color="purple" />
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
      )}
    </main>
  )
}
