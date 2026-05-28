import { Activity, Heart, Radio, StopCircle, Video, History, Camera, Zap } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { BVPChart } from '@/components/BVPChart/BVPChart'
import { FaceOverlay } from '@/components/FaceOverlay/FaceOverlay'
import { VitalSignCard } from '@/components/VitalSignCard/VitalSignCard'
import { HistoryView } from '@/components/HistoryView/HistoryView'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useWebcam } from '@/hooks/useWebcam'
import { useSmoothedValue } from '@/hooks/useSmoothedValue'
import { useVitalsStore } from '@/store/vitalsStore'
import { useAuth } from '@/features/auth/AuthProvider'
import './Live.css'

// Đo lường thời gian thực qua webcam — gửi WebSocket sang backend rPPG
export function Live() {
  const {
    heartRate, snrDb, hrvMs, bvpWindow,
    bufferFrames, bufferNeeded,
    faceBbox, faceDetected, isConnected, reset,
  } = useVitalsStore()

  const { session } = useAuth()
  const [age, setAge] = useState(undefined)
  const [showHistory, setShowHistory] = useState(false)

  const displayHeartRate = useSmoothedValue(heartRate)
  const displaySnrDb     = useSmoothedValue(snrDb)
  const displayHrvMs     = useSmoothedValue(hrvMs)

  const ageRef = useCallback(() => (age != null && !Number.isNaN(age) ? age : null), [age])
  const { connect, disconnect, sendFrame } = useWebSocket(ageRef, session?.access_token)
  const { videoRef, active, error, start, stop } = useWebcam(sendFrame)

  const handleStart = useCallback(async () => {
    reset(); connect(); await start()
  }, [reset, connect, start])

  const handleStop = useCallback(() => {
    stop(); disconnect()
  }, [stop, disconnect])

  useEffect(() => () => { stop(); disconnect() }, [stop, disconnect])

  const bufferPct = bufferNeeded > 0 ? Math.min(100, (bufferFrames / bufferNeeded) * 100) : 0

  return (
    <main className="live">
      <header className="live__header">
        <div>
          <span className="eyebrow">{showHistory ? 'Module · history' : 'Module · realtime'}</span>
          <h2 className="live__title">
            {showHistory ? (
              <>Lịch sử <em className="serif">đo lường</em></>
            ) : (
              <>Đo <em className="serif">trực tiếp</em> qua camera</>
            )}
          </h2>
        </div>
        {!showHistory && (
          <button type="button" onClick={() => setShowHistory(true)} className="live__history-btn">
            <History size={15} /> <span>Xem lịch sử</span>
          </button>
        )}
      </header>

      {showHistory ? (
        <HistoryView onBack={() => setShowHistory(false)} />
      ) : (
        <div className="live__layout">
          {/* ── Left: camera & controls ───────────────────────── */}
          <section className="live__camera-col">
            <div className="live__camera-card">
              <div className="live__camera-meta">
                <span className={`live__status ${isConnected ? 'live__status--on' : 'live__status--off'}`}>
                  <span className="live__status-dot" />
                  {isConnected ? 'Online · WebSocket' : 'Offline'}
                </span>
                <span className="live__camera-tag mono">CAM · 01</span>
              </div>

              <div className="live__camera-frame">
                <FaceOverlay
                  videoRef={videoRef}
                  bbox={faceBbox}
                  detected={faceDetected}
                  className="live__video-overlay"
                />
                {!active && (
                  <div className="live__camera-placeholder">
                    <div className="live__camera-placeholder-icon">
                      <Camera size={28} />
                    </div>
                    <p className="live__camera-placeholder-text">
                      Camera chưa kích hoạt.<br />
                      <span className="live__camera-placeholder-hint">Bấm “Start Camera” để bắt đầu phiên đo.</span>
                    </p>
                  </div>
                )}
                <div className="live__camera-corners" aria-hidden="true">
                  <span className="live__corner live__corner--tl" />
                  <span className="live__corner live__corner--tr" />
                  <span className="live__corner live__corner--bl" />
                  <span className="live__corner live__corner--br" />
                </div>
              </div>

              {error && <div className="live__error">{error}</div>}

              <div className="live__camera-controls">
                <label className="live__age">
                  <span className="live__age-label">Tuổi người dùng</span>
                  <input
                    type="number"
                    min="0"
                    value={age ?? ''}
                    onChange={(event) => {
                      const raw = event.target.value
                      const value = raw === '' ? undefined : Number(raw)
                      setAge(value == null || Number.isNaN(value) ? undefined : value)
                    }}
                    placeholder="≥ 8"
                    className="live__age-input mono"
                  />
                </label>

                {!active ? (
                  <button onClick={handleStart} className="live__btn live__btn--start" type="button">
                    <Video size={16} />
                    <span>Start Camera</span>
                  </button>
                ) : (
                  <button onClick={handleStop} className="live__btn live__btn--stop" type="button">
                    <StopCircle size={16} />
                    <span>Stop</span>
                  </button>
                )}
              </div>

              <p className="live__age-hint">
                Nếu để trống, mặc định sử dụng dải băng tần cho người lớn (≥ 8 tuổi).
              </p>

              {/* Buffer progress */}
              <div className="live__buffer">
                <div className="live__buffer-head">
                  <span className="eyebrow">Buffer</span>
                  <span className="live__buffer-text mono">
                    {heartRate !== null
                      ? 'Monitoring · live'
                      : bufferNeeded > 0
                        ? `${bufferFrames} / ${bufferNeeded} frames`
                        : 'Warming up…'}
                  </span>
                </div>
                <div className="live__buffer-track">
                  <div
                    className="live__buffer-bar"
                    style={{
                      width: heartRate !== null ? '100%' : `${bufferPct}%`,
                    }}
                  />
                </div>
                {heartRate === null && bufferNeeded > 0 && (
                  <span className="live__buffer-eta mono">
                    ETA · {Math.ceil((bufferNeeded - bufferFrames) / 30)}s
                  </span>
                )}
              </div>
            </div>
          </section>

          {/* ── Right: vitals & chart ─────────────────────────── */}
          <section className="live__vitals-col">
            <header className="live__vitals-head">
              <span className="eyebrow">Vital signs</span>
              <h3 className="live__vitals-title">Bảng theo dõi</h3>
            </header>

            <div className="live__vitals-stack">
              <VitalSignCard icon={Heart}    label="Heart Rate"  value={displayHeartRate} unit="BPM" color="crimson" />
              <VitalSignCard icon={Activity} label="Signal SNR"  value={displaySnrDb}     unit="dB"  color="cyan"    />
              <VitalSignCard icon={Radio}    label="HRV (RMSSD)" value={displayHrvMs}     unit="ms"  color="violet"  />
            </div>

            <div className="live__chart">
              <div className="live__chart-head">
                <span className="live__chart-title">
                  <Zap size={13} /> BVP Signal
                </span>
                <span className="live__chart-tag mono">live · 30fps</span>
              </div>
              <div className="live__chart-wrap">
                <BVPChart data={bvpWindow} />
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}
