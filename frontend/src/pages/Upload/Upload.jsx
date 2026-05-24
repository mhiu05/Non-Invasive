import { Activity, AlertTriangle, CheckCircle2, Download, Eye, FileVideo, Heart, Loader2, Radio, Upload as UploadIcon } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { BVPChart } from '@/components/BVPChart/BVPChart'
import { VitalSignCard } from '@/components/VitalSignCard/VitalSignCard'
import { SnrBadge } from '@/components/SnrBadge/SnrBadge'
import { HistoryView } from '@/components/HistoryView/HistoryView'
import { uploadVideoAsync, getJobStatus } from '@/lib/api'
import { downloadCSV } from '@/lib/utils'
import { getSnrQuality } from '@/lib/vitals'
import './Upload.css'

// Component Upload xử lý việc tải video lên server để phân tích bất đồng bộ (offline)
export function Upload() {
  // Tham chiếu đến thẻ input file ẩn
  const inputRef = useRef(null)
  
  // Trạng thái của quá trình xử lý: 'idle' | 'uploading' | 'processing' | 'done' | 'error'
  const [state, setState] = useState('idle')
  // Phần trăm tiến trình upload file
  const [progress, setProgress] = useState(0)
  // Kết quả trả về sau khi phân tích xong
  const [result, setResult] = useState(null)
  // Lưu trữ thông báo lỗi nếu có
  const [errMsg, setErrMsg] = useState('')
  // Trạng thái kéo thả file
  const [dragging, setDragging] = useState(false)
  // Độ tuổi (nếu có) để server dùng bộ lọc bandpass tương ứng
  const [age, setAge] = useState(undefined)
  // ID của job đang được server xử lý
  const [jobId, setJobId] = useState(null)

  // Ẩn/hiện màn hình Lịch sử đo
  const [showHistory, setShowHistory] = useState(false)

  // Hàm xử lý file video sau khi người dùng chọn/kéo thả
  const processFile = useCallback(async (file) => {
    // Kiểm tra định dạng file
    if (!file.type.startsWith('video/')) {
      setErrMsg('Chỉ hỗ trợ file video (mp4, avi, mov…)')
      setState('error')
      return
    }
    
    // Reset các state để chuẩn bị quá trình mới
    setState('uploading')
    setProgress(0)
    setResult(null)
    setErrMsg('')
    setJobId(null)
    
    try {
      // 1. Upload video lên server
      const job = await uploadVideoAsync(file, age, setProgress)
      setJobId(job.job_id)
      setState('processing')
      
      // 2. Định kỳ polling (gọi API) kiểm tra trạng thái job
      const poll = setInterval(async () => {
        try {
          const status = await getJobStatus(job.job_id)
          if (status.status === 'done' && status.result) {
            clearInterval(poll)
            setResult(status.result)
            setState('done')
          } else if (status.status === 'failed') {
            clearInterval(poll)
            setErrMsg(status.error || 'Xử lý video thất bại')
            setState('error')
          }
        } catch {
          clearInterval(poll)
          setErrMsg('Mất kết nối khi kiểm tra trạng thái')
          setState('error')
        }
      }, 2000) // Kiểm tra mỗi 2 giây
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : 'Upload thất bại')
      setState('error')
    }
  }, [age])

  // Bắt sự kiện người dùng chọn file từ ô input
  const onFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }
  
  // Bắt sự kiện người dùng kéo thả file vào vùng dropzone
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }
  
  // Hàm xuất dữ liệu tín hiệu BVP ra file CSV
  const handleExport = () => {
    if (!result) return
    downloadCSV(result.bvp_signal.map((v, i) => ({ frame: i, bvp: v })), `bvp_${result.filename}.csv`)
  }

  // Đánh giá xem có nên hiện cảnh báo chất lượng tín hiệu kém không
  const snrWarning = result && getSnrQuality(result.snr_db) === 'poor'

  return (
    <main className="upload-main">
      {!showHistory ? (
        <>
          <div className="upload-header">
            <div>
              <h2 className="upload-title">Offline Video Analysis</h2>
              <p className="upload-subtitle">Upload video (async) để phân tích nhịp tim, blink rate, HRV.</p>
            </div>
            <button type="button" onClick={() => setShowHistory(true)} className="upload-history-btn">
              Xem lịch sử
            </button>
          </div>

          {/* Ô nhập thông tin tuổi để cải thiện thuật toán lọc */}
          <div className="upload-age-group">
            <label className="upload-age-label">Tuổi người dùng</label>
            <input
              type="number"
              min="0"
              value={age ?? ''}
              onChange={e => {
                const v = e.target.value === '' ? undefined : Number(e.target.value)
                setAge(v == null || Number.isNaN(v) ? undefined : v)
              }}
              placeholder="8"
              className="upload-age-input"
            />
          </div>

          {/* Khu vực Drag & Drop để kéo thả file */}
          <div
            className={`upload-dropzone ${dragging ? 'upload-dropzone-dragging' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <FileVideo className="upload-dropzone-icon" size={40} />
            <p className="upload-dropzone-text">Kéo thả video vào đây hoặc <span className="upload-dropzone-link">chọn file</span></p>
            <p className="upload-dropzone-hint">MP4, AVI, MOV · tối đa 100 MB · Async processing</p>
            <input ref={inputRef} type="file" accept="video/*" className="upload-hidden-input" onChange={onFileChange} />
          </div>

          {/* Thanh tiến trình khi đang upload */}
          {state === 'uploading' && (
            <div className="upload-progress-container">
              <div className="upload-progress-text">
                <Loader2 size={15} className="upload-spinner" /> Đang tải lên… {progress}%
              </div>
              <div className="upload-progress-track">
                <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Hiệu ứng chờ khi server đang xử lý video */}
          {state === 'processing' && jobId && (
            <div className="upload-processing-container">
              <div className="upload-processing-header">
                <Loader2 size={20} className="upload-spinner" />
                <div>
                  <p className="upload-processing-title">Đang xử lý video…</p>
                  <p className="upload-processing-subtitle">Job ID: <code className="upload-job-id">{jobId}</code></p>
                </div>
              </div>
              <div className="upload-processing-track">
                <div className="upload-processing-bar" />
              </div>
            </div>
          )}

          {/* Hiển thị lỗi nếu quá trình có vấn đề */}
          {state === 'error' && (
            <p className="upload-error-msg">{errMsg}</p>
          )}

          {/* Phần hiển thị kết quả xử lý thành công */}
          {state === 'done' && result && (
            <div className="upload-results-container">
              <div className="upload-results-header">
                <div>
                  <div className="upload-results-title-group">
                    <CheckCircle2 size={16} className="upload-success-icon" />
                    <p className="upload-filename">{result.filename}</p>
                  </div>
                  <p className="upload-file-stats">{result.total_frames} frames · {result.duration_sec}s{result.age_group ? ` · ${result.age_group}` : ''}</p>
                </div>
                <div className="upload-results-actions">
                  <SnrBadge snr={result.snr_db} />
                  <button onClick={handleExport} className="upload-export-btn">
                    <Download size={13} /> Export CSV
                  </button>
                </div>
              </div>

              {/* Cảnh báo tín hiệu (nếu có) */}
              {snrWarning && (
                <div className="upload-snr-warning">
                  <AlertTriangle size={18} className="upload-warning-icon" />
                  <div>
                    <p className="upload-warning-title">Chất lượng tín hiệu thấp</p>
                    <p className="upload-warning-text">SNR {result.snr_db.toFixed(1)} dB {'<'} 0 dB, bạn nên ngồi yên và trong điều kiện đủ ánh sáng hơn.</p>
                  </div>
                </div>
              )}

              {/* Các thông số cơ bản */}
              <div className="upload-vitals-grid">
                <VitalSignCard icon={Heart}    label="Heart Rate"  value={result.heart_rate} unit="BPM"    color="red" />
                <VitalSignCard icon={Eye}      label="Blink Rate"  value={result.blink_rate} unit="bl/min" color="blue" />
                <VitalSignCard icon={Activity} label="Signal SNR"  value={result.snr_db}     unit="dB"     color="green" />
                <VitalSignCard icon={Radio}    label="HRV (RMSSD)" value={result.hrv_ms ?? result.rmssd_ms ?? null} unit="ms" color="purple" />
              </div>

              {/* Các thông số HRV chuyên sâu (SDNN, pNN50, Peak Count) */}
              {(result.sdnn_ms != null || result.pnn50 != null) && (
                <div className="upload-ministats-grid">
                  <MiniStat label="SDNN" value={result.sdnn_ms} unit="ms" />
                  <MiniStat label="pNN50" value={result.pnn50} unit="%" />
                  <MiniStat label="Peaks" value={result.peak_count} unit="" />
                </div>
              )}

              {/* Biểu đồ tín hiệu BVP */}
              <div className="upload-chart-section">
                <p className="upload-chart-title">
                  <UploadIcon size={13} className="upload-chart-icon" /> BVP Signal
                </p>
                <div className="upload-chart-wrapper">
                  <BVPChart data={result.bvp_signal.slice(-120)} />
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        <HistoryView onBack={() => setShowHistory(false)} />
      )}
    </main>
  )
}

// Component phụ hiển thị một thông số nhỏ gọn trong grid
function MiniStat({ label, value, unit }) {
  return (
    <div className="upload-ministat">
      <p className="upload-ministat-label">{label}</p>
      <p className="upload-ministat-value">{value?.toFixed(1) ?? '—'}<span className="upload-ministat-unit">{unit}</span></p>
    </div>
  )
}
