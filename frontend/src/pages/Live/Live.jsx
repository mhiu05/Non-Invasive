import { Activity, Heart, Radio, StopCircle, Video, History } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { BVPChart } from '@/components/BVPChart/BVPChart'
import { FaceOverlay } from '@/components/FaceOverlay/FaceOverlay'
import { VitalSignCard } from '@/components/VitalSignCard/VitalSignCard'
import { HistoryView } from '@/components/HistoryView/HistoryView'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useWebcam } from '@/hooks/useWebcam'
import { useVitalsStore } from '@/store/vitalsStore'
import './Live.css'

// Component Live xử lý việc đo lường các chỉ số sinh tồn trực tiếp từ Webcam
export function Live() {
  // Lấy các state toàn cục (thông số, buffer, face bounding box...) từ zustand store
  const { heartRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded, faceBbox, faceDetected, isConnected, reset } =
    useVitalsStore()
  
  // State quản lý độ tuổi nhập vào và các state hiển thị (đã được làm mượt)
  const [age, setAge] = useState(undefined)
  const [displayHeartRate, setDisplayHeartRate] = useState(null)
  const [displaySnrDb, setDisplaySnrDb] = useState(null)
  const [displayHrvMs, setDisplayHrvMs] = useState(null)
  
  // State ẩn/hiện bảng lịch sử
  const [showHistory, setShowHistory] = useState(false)
  
  // Hàm callback giữ giá trị độ tuổi hợp lệ
  const ageRef = useCallback(() => (age != null && !Number.isNaN(age) ? age : null), [age])

  // Custom hook kết nối WebSocket và gửi frames
  const { connect, disconnect, sendFrame } = useWebSocket(ageRef)
  // Custom hook quản lý Webcam và gửi frame định kỳ vào hàm sendFrame
  const { videoRef, active, error, start, stop } = useWebcam(sendFrame)

  // Hàm bắt đầu quá trình đo: reset store, kết nối websocket, mở camera
  const handleStart = useCallback(async () => {
    reset()
    connect()
    await start()
  }, [reset, connect, start])

  // Hàm kết thúc quá trình đo: tắt camera, ngắt kết nối websocket
  const handleStop = useCallback(() => {
    stop()
    disconnect()
  }, [stop, disconnect])

  // Các thông số dùng để làm mượt hiển thị số liệu (giảm hiệu ứng nhảy số quá nhanh)
  const smoothingIntervalMs = 140
  const smoothingFactor = 0.07
  const smoothingThreshold = 0.25

  // Hook làm mượt hiển thị cho giá trị Heart Rate
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
      if (active) window.setTimeout(step, smoothingIntervalMs)
    }
    step()
    return () => { active = false }
  }, [heartRate])



  // Hook làm mượt hiển thị cho giá trị SNR
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
      if (active) window.setTimeout(step, smoothingIntervalMs)
    }
    step()
    return () => { active = false }
  }, [snrDb])

  // Hook làm mượt hiển thị cho giá trị HRV
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
      if (active) window.setTimeout(step, smoothingIntervalMs)
    }
    step()
    return () => { active = false }
  }, [hrvMs])

  // Dọn dẹp: tự động dừng camera và ngắt kết nối khi component unmount
  useEffect(() => () => { stop(); disconnect() }, [stop, disconnect])

  return (
    <main className="live-main">
      <div className="live-header">
        <h2 className="live-title">
          {!showHistory ? 'Live Analysis' : 'Lịch sử đo'}
        </h2>
        {!showHistory && (
          <button type="button" onClick={() => setShowHistory(true)} className="live-history-btn">
            <History size={16} /> Xem lịch sử
          </button>
        )}
      </div>

      {showHistory ? (
        <HistoryView onBack={() => setShowHistory(false)} />
      ) : (
      <div className="live-content">
        {/* Cột trái: Webcam và các nút điều khiển */}
        <div className="live-webcam-section">
          {/* Badge hiển thị trạng thái kết nối */}
          <div className="live-status-container">
            <span className={`live-status-badge ${isConnected ? 'live-status-connected' : 'live-status-disconnected'}`}>
              <span className={`live-status-dot ${isConnected ? 'live-status-dot-pulse' : ''}`} />
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Component FaceOverlay bọc thẻ video để vẽ khung khuôn mặt */}
          <FaceOverlay
            videoRef={videoRef}
            bbox={faceBbox}
            detected={faceDetected}
            className="live-video-overlay"
          />

          {error && (
            <p className="live-error-msg">
              {error}
            </p>
          )}

          {/* Ô nhập thông tin tuổi để cải thiện thuật toán lọc */}
          <div className="live-age-input-group">
            <label className="live-age-label">Tuổi người dùng</label>
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
              className="live-age-input"
            />
            <p className="live-age-hint">
              Nếu không nhập, mặc định dùng dải cho người lớn (≥ 8 tuổi).
            </p>
          </div>

          {/* Các nút Bắt đầu / Kết thúc đo lường */}
          <div className="live-controls">
            {!active ? (
              <button onClick={handleStart} className="live-start-btn">
                <Video size={16} /> Start Camera
              </button>
            ) : (
              <button onClick={handleStop} className="live-stop-btn">
                <StopCircle size={16} /> Stop
              </button>
            )}
          </div>

          {/* Hiển thị thông tin bộ đệm (số khung hình cần để tính toán) */}
          <p className="live-buffer-info">
            {heartRate !== null
              ? 'Monitoring live...'
              : bufferNeeded > 0
                ? `Buffer: ${bufferFrames} / ${bufferNeeded} frames · Model needs ~${Math.ceil(bufferNeeded / 30)}s`
                : 'Warming up...'}
          </p>
        </div>

        {/* Cột phải: Hiển thị các chỉ số sinh tồn và biểu đồ */}
        <div className="live-vitals-section">
          <h2 className="live-vitals-title">Vital Signs</h2>

          <div className="live-vitals-grid">
            <VitalSignCard icon={Heart}    label="Heart Rate"     value={displayHeartRate}  unit="BPM"     color="red"    />
            <VitalSignCard icon={Activity} label="Signal SNR"     value={displaySnrDb}      unit="dB"      color="green"  />
            <VitalSignCard icon={Radio}    label="HRV (RMSSD)"    value={displayHrvMs}      unit="ms"      color="purple" />
          </div>

          {/* Khu vực hiển thị biểu đồ BVP */}
          <div className="live-chart-container">
            <div className="live-chart-header">
              <Radio size={14} className="live-chart-icon" />
              <span className="live-chart-label">BVP Signal</span>
            </div>
            <div className="live-chart-wrapper">
              <BVPChart data={bvpWindow} />
            </div>
          </div>
        </div>
      </div>
      )}
    </main>
  )
}
