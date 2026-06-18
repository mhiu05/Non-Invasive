import {
  Activity, AlertTriangle, CheckCircle2, Download, FileVideo,
  Heart, Loader2, Radio, Upload as UploadIcon, History,
} from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { BVPChart } from '@/components/BVPChart/BVPChart'
import { VitalSignCard } from '@/components/VitalSignCard/VitalSignCard'
import { SnrBadge } from '@/components/SnrBadge/SnrBadge'
import { HistoryView } from '@/components/HistoryView/HistoryView'
import { uploadVideo, getJobStatus } from '@/lib/api'
import { downloadCSV } from '@/lib/utils'
import { getSnrQuality } from '@/lib/vitals'
import './Upload.css'

// Upload video để phân tích offline qua server (async polling)
export function Upload() {
  const inputRef = useRef(null)
  const [state, setState] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [errMsg, setErrMsg] = useState('')
  const [dragging, setDragging] = useState(false)
  const [age, setAge] = useState(undefined)
  const [jobId, setJobId] = useState(null)
  const [showHistory, setShowHistory] = useState(false)

  const processFile = useCallback(async (file) => {
    if (!file.type.startsWith('video/')) {
      setErrMsg('Chỉ hỗ trợ file video (mp4, avi, mov…)')
      setState('error')
      return
    }
    setState('uploading'); setProgress(0); setResult(null); setErrMsg(''); setJobId(null)

    try {
      const handleProgress = (pct) => {
        setProgress(pct)
        if (pct >= 100) {
          setState('processing')
        }
      }
      
      const resultData = await uploadVideo(file, age, handleProgress)
      setResult(resultData)
      setState('done')
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : (e?.response?.data?.detail || 'Upload thất bại'))
      setState('error')
    }
  }, [age])

  const onFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }

  const handleExport = () => {
    if (!result) return
    downloadCSV(result.bvp_signal.map((v, i) => ({ frame: i, bvp: v })), `bvp_${result.filename}.csv`)
  }

  const snrWarning = result && getSnrQuality(result.snr_db) === 'poor'

  return (
    <main className="upload">
      {showHistory ? (
        <HistoryView onBack={() => setShowHistory(false)} />
      ) : (
        <>
          <header className="upload__header">
            <div>
              <span className="eyebrow">Module · offline</span>
              <h2 className="upload__title">
                Phân tích <em className="serif">video</em>
              </h2>
              <p className="upload__sub">Upload video (async) để trích xuất nhịp tim, HRV và các chỉ số sinh tồn.</p>
            </div>
            <button type="button" onClick={() => setShowHistory(true)} className="upload__history-btn">
              <History size={15} /> <span>Xem lịch sử</span>
            </button>
          </header>

          <div className="upload__row">
            <label className="upload__age">
              <span className="upload__age-label">Tuổi người dùng</span>
              <input
                type="number"
                min="0"
                value={age ?? ''}
                onChange={e => {
                  const v = e.target.value === '' ? undefined : Number(e.target.value)
                  setAge(v == null || Number.isNaN(v) ? undefined : v)
                }}
                placeholder="≥ 8"
                className="upload__age-input mono"
              />
              <span className="upload__age-hint">
                Nếu để trống, hệ thống mặc định dải băng tần cho người lớn.
              </span>
            </label>
          </div>

          {/* Drop zone */}
          <div
            className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            role="button"
            tabIndex={0}
          >
            <div className="dropzone__inner">
              <span className="dropzone__corner dropzone__corner--tl" />
              <span className="dropzone__corner dropzone__corner--tr" />
              <span className="dropzone__corner dropzone__corner--bl" />
              <span className="dropzone__corner dropzone__corner--br" />

              <span className="dropzone__icon">
                <FileVideo size={32} />
              </span>
              <h3 className="dropzone__title">
                Kéo thả video vào đây
                <br />
                <span className="dropzone__or">hoặc <span className="dropzone__link">chọn file</span></span>
              </h3>
              <p className="dropzone__hint mono">MP4 · AVI · MOV  ·  tối đa 100 MB  ·  async processing</p>

              <input
                ref={inputRef}
                type="file"
                accept="video/*"
                className="dropzone__input"
                onChange={onFileChange}
              />
            </div>
          </div>

          {/* Upload progress */}
          {state === 'uploading' && (
            <div className="upload__progress">
              <div className="upload__progress-text">
                <Loader2 size={14} className="upload__spinner" />
                <span>Đang tải lên</span>
                <span className="mono upload__progress-pct">{progress}%</span>
              </div>
              <div className="upload__progress-track">
                <div className="upload__progress-bar" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Processing */}
          {state === 'processing' && jobId && (
            <div className="upload__processing">
              <Loader2 size={20} className="upload__spinner" />
              <div className="upload__processing-text">
                <p className="upload__processing-title">Đang xử lý video…</p>
                <p className="upload__processing-sub">
                  Job ID <code className="upload__job-id mono">{jobId}</code>
                </p>
              </div>
              <div className="upload__processing-track">
                <div className="upload__processing-bar" />
              </div>
            </div>
          )}

          {state === 'error' && (
            <div className="upload__error">
              <AlertTriangle size={16} />
              <span>{errMsg}</span>
            </div>
          )}

          {/* Results */}
          {state === 'done' && result && (
            <section className="results">
              <header className="results__head">
                <div className="results__head-info">
                  <div className="results__head-title">
                    <span className="results__success">
                      <CheckCircle2 size={14} /> SUCCESS
                    </span>
                    <h3 className="results__filename">{result.filename}</h3>
                  </div>
                  <p className="results__stats mono">
                    {result.total_frames} frames · {result.duration_sec}s
                    {result.age_group ? ` · ${result.age_group}` : ''}
                  </p>
                </div>
                <div className="results__actions">
                  <SnrBadge snr={result.snr_db} />
                  <button onClick={handleExport} className="results__export-btn" type="button">
                    <Download size={13} /> Export CSV
                  </button>
                </div>
              </header>

              {snrWarning && (
                <div className="results__warning">
                  <span className="results__warning-icon">
                    <AlertTriangle size={18} />
                  </span>
                  <div className="results__warning-text">
                    <p className="results__warning-title">Chất lượng tín hiệu thấp</p>
                    <p className="results__warning-desc">
                      SNR <span className="mono">{result.snr_db.toFixed(1)} dB</span> &lt; 0 dB, hãy ngồi yên và trong điều kiện đủ ánh sáng hơn.
                    </p>
                  </div>
                </div>
              )}

              <div className="results__vitals">
                <VitalSignCard icon={Heart}    label="Heart Rate"  value={result.heart_rate}                                  unit="BPM" color="crimson" />
                <VitalSignCard icon={Activity} label="Signal SNR"  value={result.snr_db}                                      unit="dB"  color="cyan" />
                <VitalSignCard icon={Radio}    label="HRV (RMSSD)" value={result.hrv_ms ?? result.rmssd_ms ?? null}           unit="ms"  color="violet" />
              </div>

              {(result.sdnn_ms != null || result.pnn50 != null) && (
                <div className="results__mini">
                  <MiniStat label="SDNN" value={result.sdnn_ms} unit="ms" />
                  <MiniStat label="pNN50" value={result.pnn50} unit="%" />
                  <MiniStat label="Peaks" value={result.peak_count} unit="" />
                </div>
              )}

              <div className="results__chart">
                <header className="results__chart-head">
                  <span className="results__chart-title">
                    <UploadIcon size={13} /> BVP Signal
                  </span>
                  <span className="results__chart-tag mono">last 120 samples</span>
                </header>
                <div className="results__chart-wrap">
                  <BVPChart data={result.bvp_signal.slice(-120)} />
                </div>
              </div>
            </section>
          )}
        </>
      )}
    </main>
  )
}

function MiniStat({ label, value, unit }) {
  return (
    <div className="ministat">
      <span className="ministat__label">{label}</span>
      <span className="ministat__value">
        <span className="ministat__num">{value?.toFixed(1) ?? '—'}</span>
        <span className="ministat__unit">{unit}</span>
      </span>
    </div>
  )
}
