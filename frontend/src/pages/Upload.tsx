import { Activity, AlertTriangle, ArrowLeft, CheckCircle2, Clock3, Download, Eye, FileVideo, Filter, Heart, Loader2, Radio, RefreshCw, Upload as UploadIcon, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { BVPChart } from '@/components/BVPChart'
import { VitalSignCard } from '@/components/VitalSignCard'
import { SnrBadge } from '@/components/SnrBadge'
import { fetchHistory, uploadVideoAsync, getJobStatus } from '@/lib/api'
import { downloadCSV } from '@/lib/utils'
import type { HistoryRecord, VideoResult, HistoryFilters, AsyncJobStatus } from '@/types/vitals'
import { getSnrQuality } from '@/types/vitals'

type UploadState = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export function Upload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [state, setState] = useState<UploadState>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<VideoResult | null>(null)
  const [errMsg, setErrMsg] = useState('')
  const [dragging, setDragging] = useState(false)
  const [age, setAge] = useState<number | undefined>(undefined)
  const [jobId, setJobId] = useState<string | null>(null)

  // History state
  const [showHistory, setShowHistory] = useState(false)
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null)

  // History filters
  const [filterType, setFilterType] = useState<string>('')
  const [filterStart, setFilterStart] = useState('')
  const [filterEnd, setFilterEnd] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const filters: HistoryFilters = {}
      if (filterType) filters.type = filterType
      if (filterStart) filters.start_at = new Date(filterStart).toISOString()
      if (filterEnd) filters.end_at = new Date(filterEnd).toISOString()
      const data = await fetchHistory(filters)
      setHistoryRecords(data)
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Không thể tải lịch sử')
    } finally {
      setHistoryLoading(false)
    }
  }, [filterType, filterStart, filterEnd])

  useEffect(() => {
    if (showHistory) loadHistory()
  }, [showHistory, loadHistory])

  // Async upload + polling
  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setErrMsg('Chỉ hỗ trợ file video (mp4, avi, mov…)')
      setState('error')
      return
    }
    setState('uploading')
    setProgress(0)
    setResult(null)
    setErrMsg('')
    setJobId(null)
    try {
      const job = await uploadVideoAsync(file, age, setProgress)
      setJobId(job.job_id)
      setState('processing')
      // Start polling
      const poll = setInterval(async () => {
        try {
          const status: AsyncJobStatus = await getJobStatus(job.job_id)
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
      }, 2000)
    } catch (e: unknown) {
      setErrMsg(e instanceof Error ? e.message : 'Upload thất bại')
      setState('error')
    }
  }, [age])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }
  const handleExport = () => {
    if (!result) return
    downloadCSV(result.bvp_signal.map((v, i) => ({ frame: i, bvp: v })), `bvp_${result.filename}.csv`)
  }

  const snrWarning = result && getSnrQuality(result.snr_db) === 'poor'

  return (
    <main className="mx-auto max-w-3xl px-4 py-6 space-y-6">
      {/* ── Detail Modal ── */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setSelectedRecord(null)}>
          <div className="relative m-4 w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl dark:bg-slate-900" onClick={e => e.stopPropagation()}>
            <button onClick={() => setSelectedRecord(null)} className="absolute right-4 top-4 rounded-lg p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"><X size={18} /></button>
            <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-1">Chi tiết phiên đo</h3>
            <p className="text-xs text-slate-500 mb-4">{new Date(selectedRecord.created_at).toLocaleString('vi-VN')}</p>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <DetailRow label="Loại" value={selectedRecord.type === 'video' ? 'Video' : 'Realtime'} />
              <DetailRow label="File" value={selectedRecord.filename ?? '—'} />
              <DetailRow label="Thời lượng" value={selectedRecord.duration_sec ? `${selectedRecord.duration_sec}s` : '—'} />
              <DetailRow label="Heart Rate" value={selectedRecord.heart_rate?.toFixed(1) ?? '—'} unit="BPM" />
              <DetailRow label="Blink Rate" value={selectedRecord.blink_rate?.toFixed(1) ?? '—'} unit="bl/min" />
              <DetailRow label="SNR" value={selectedRecord.snr_db?.toFixed(1) ?? '—'} unit="dB" badge={<SnrBadge snr={selectedRecord.snr_db} />} />
              <DetailRow label="Nhóm tuổi" value={selectedRecord.age_group ?? '—'} />
              <DetailRow label="Tuổi" value={selectedRecord.age?.toString() ?? '—'} />
              <DetailRow label="Bandpass" value={selectedRecord.bandpass_low_hz != null ? `${selectedRecord.bandpass_low_hz} – ${selectedRecord.bandpass_high_hz} Hz` : '—'} />
              <DetailRow label="HRV (RMSSD)" value={selectedRecord.rmssd_ms?.toFixed(1) ?? '—'} unit="ms" />
              <DetailRow label="SDNN" value={selectedRecord.sdnn_ms?.toFixed(1) ?? '—'} unit="ms" />
              <DetailRow label="pNN50" value={selectedRecord.pnn50?.toFixed(1) ?? '—'} unit="%" />
              <DetailRow label="Peak Count" value={selectedRecord.peak_count?.toString() ?? '—'} />
            </div>
          </div>
        </div>
      )}

      {!showHistory ? (
        <>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Offline Video Analysis</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Upload video (async) để phân tích nhịp tim, blink rate, HRV.</p>
            </div>
            <button type="button" onClick={() => setShowHistory(true)} className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-900">
              Xem lịch sử
            </button>
          </div>

          <div className="grid gap-3 sm:grid-cols-[1fr_120px]">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Tuổi người dùng</label>
            <input type="number" min="0" value={age ?? ''} onChange={e => { const v = e.target.value === '' ? undefined : Number(e.target.value); setAge(v == null || Number.isNaN(v) ? undefined : v) }} placeholder="8" className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-indigo-400 dark:focus:ring-indigo-500/20" />
          </div>

          {/* Drop zone */}
          <div className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${dragging ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20' : 'border-slate-300 hover:border-indigo-400 dark:border-slate-600'}`} onClick={() => inputRef.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <FileVideo className="mx-auto mb-3 text-slate-400" size={40} />
            <p className="font-medium text-slate-700 dark:text-slate-200">Kéo thả video vào đây hoặc <span className="text-indigo-600 underline">chọn file</span></p>
            <p className="mt-1 text-xs text-slate-400">MP4, AVI, MOV · tối đa 100 MB · Async processing</p>
            <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={onFileChange} />
          </div>

          {/* Upload progress */}
          {state === 'uploading' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <Loader2 size={15} className="animate-spin text-indigo-500" /> Đang tải lên… {progress}%
              </div>
              <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700">
                <div className="h-2 rounded-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Processing status */}
          {state === 'processing' && jobId && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4 dark:border-indigo-800 dark:bg-indigo-950/30">
              <div className="flex items-center gap-3">
                <Loader2 size={20} className="animate-spin text-indigo-500" />
                <div>
                  <p className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Đang xử lý video…</p>
                  <p className="text-xs text-indigo-500 dark:text-indigo-400 mt-0.5">Job ID: <code className="rounded bg-indigo-100 px-1.5 py-0.5 font-mono text-[10px] dark:bg-indigo-900/50">{jobId}</code></p>
                </div>
              </div>
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-indigo-200 dark:bg-indigo-800">
                <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-500" />
              </div>
            </div>
          )}

          {/* Error */}
          {state === 'error' && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">{errMsg}</p>
          )}

          {/* Results */}
          {state === 'done' && result && (
            <div className="space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500" />
                    <p className="font-medium text-slate-800 dark:text-slate-100">{result.filename}</p>
                  </div>
                  <p className="text-xs text-slate-500">{result.total_frames} frames · {result.duration_sec}s{result.age_group ? ` · ${result.age_group}` : ''}</p>
                </div>
                <div className="flex items-center gap-2">
                  <SnrBadge snr={result.snr_db} />
                  <button onClick={handleExport} className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800">
                    <Download size={13} /> Export CSV
                  </button>
                </div>
              </div>

              {/* SNR Warning */}
              {snrWarning && (
                <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-950/30">
                  <AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-500" />
                  <div>
                    <p className="text-sm font-medium text-amber-700 dark:text-amber-300">Chất lượng tín hiệu thấp</p>
                    <p className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">SNR {result.snr_db.toFixed(1)} dB {'<'} 3 dB. Kết quả có thể không chính xác. Thử ngồi yên, đủ ánh sáng, hoặc dùng model FactorizePhys/PhysFormer.</p>
                  </div>
                </div>
              )}

              <div className="grid gap-3 sm:grid-cols-4">
                <VitalSignCard icon={Heart}    label="Heart Rate"  value={result.heart_rate} unit="BPM"    color="red" />
                <VitalSignCard icon={Eye}      label="Blink Rate"  value={result.blink_rate} unit="bl/min" color="blue" />
                <VitalSignCard icon={Activity} label="Signal SNR"  value={result.snr_db}     unit="dB"     color="green" />
                <VitalSignCard icon={Radio}    label="HRV (RMSSD)" value={result.hrv_ms ?? result.rmssd_ms ?? null} unit="ms" color="purple" />
              </div>

              {/* Extra HRV details */}
              {(result.sdnn_ms != null || result.pnn50 != null) && (
                <div className="grid grid-cols-3 gap-3">
                  <MiniStat label="SDNN" value={result.sdnn_ms} unit="ms" />
                  <MiniStat label="pNN50" value={result.pnn50} unit="%" />
                  <MiniStat label="Peaks" value={result.peak_count} unit="" />
                </div>
              )}

              <div className="space-y-2">
                <p className="text-xs font-medium text-slate-600 dark:text-slate-400 flex items-center gap-1.5">
                  <UploadIcon size={13} className="text-indigo-500" /> BVP Signal
                </p>
                <div className="rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200 dark:bg-slate-800 dark:ring-slate-700">
                  <BVPChart data={result.bvp_signal.slice(-120)} />
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        /* ── History View ── */
        <div className="space-y-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Measurement History</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Danh sách phiên đo đã lưu.</p>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={() => setShowFilters(!showFilters)} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
                <Filter size={14} /> Lọc
              </button>
              <button type="button" onClick={() => setShowHistory(false)} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
                <ArrowLeft size={14} /> Back
              </button>
              <button type="button" onClick={loadHistory} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          {/* Filter panel */}
          {showFilters && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
              <div className="grid gap-3 sm:grid-cols-3">
                <div>
                  <label className="text-xs font-medium text-slate-500 dark:text-slate-400">Loại</label>
                  <select value={filterType} onChange={e => setFilterType(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
                    <option value="">Tất cả</option>
                    <option value="video">Video</option>
                    <option value="realtime">Realtime</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-500 dark:text-slate-400">Từ ngày</label>
                  <input type="date" value={filterStart} onChange={e => setFilterStart(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                </div>
                <div>
                  <label className="text-xs font-medium text-slate-500 dark:text-slate-400">Đến ngày</label>
                  <input type="date" value={filterEnd} onChange={e => setFilterEnd(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
                </div>
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <button onClick={() => { setFilterType(''); setFilterStart(''); setFilterEnd('') }} className="text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400">Xóa bộ lọc</button>
                <button onClick={loadHistory} className="rounded-lg bg-indigo-500 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-600">Áp dụng</button>
              </div>
            </div>
          )}

          {historyLoading ? (
            <div className="rounded-2xl border border-slate-200 bg-white px-6 py-10 text-center text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">Đang tải lịch sử…</div>
          ) : historyError ? (
            <div className="rounded-2xl border border-red-200 bg-red-50 px-6 py-10 text-center text-red-700 shadow-sm dark:border-red-900/50 dark:bg-red-950/50 dark:text-red-300">{historyError}</div>
          ) : historyRecords.length === 0 ? (
            <div className="rounded-2xl border border-slate-200 bg-white px-6 py-10 text-center text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">Chưa có phiên đo nào.</div>
          ) : (
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-950">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 text-sm dark:divide-slate-700">
                  <thead className="bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                    <tr>
                      <th className="px-4 py-3">Thời gian</th>
                      <th className="px-4 py-3">Loại</th>
                      <th className="px-4 py-3">File</th>
                      <th className="px-4 py-3">HR</th>
                      <th className="px-4 py-3">Blink</th>
                      <th className="px-4 py-3">SNR</th>
                      <th className="px-4 py-3">HRV</th>
                      <th className="px-4 py-3">Thời lượng</th>
                      <th className="px-4 py-3">Tuổi</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 bg-white dark:divide-slate-700 dark:bg-slate-950">
                    {historyRecords.map((r) => (
                      <tr key={r.id} className="cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900/80 transition-colors" onClick={() => setSelectedRecord(r)}>
                        <td className="px-4 py-3 text-xs text-slate-600 dark:text-slate-300">
                          <div className="flex items-center gap-1.5"><Clock3 size={12} />{new Date(r.created_at).toLocaleString('vi-VN')}</div>
                        </td>
                        <td className="px-4 py-3 font-medium text-slate-800 dark:text-slate-100">{r.type === 'video' ? 'Video' : 'Realtime'}</td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300 max-w-[120px] truncate">{r.filename ?? r.session_id ?? '—'}</td>
                        <td className="px-4 py-3 text-slate-800 dark:text-slate-100">{r.heart_rate?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-3 text-slate-800 dark:text-slate-100">{r.blink_rate?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-3"><SnrBadge snr={r.snr_db} /></td>
                        <td className="px-4 py-3 text-slate-800 dark:text-slate-100">{r.hrv_ms?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-3 text-slate-800 dark:text-slate-100">{r.duration_sec ? `${r.duration_sec}s` : '—'}</td>
                        <td className="px-4 py-3 text-slate-800 dark:text-slate-100">{r.age_group ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  )
}

/* ── Helper components ── */
function DetailRow({ label, value, unit, badge }: { label: string; value: string; unit?: string; badge?: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/50">
      <p className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500">{label}</p>
      {badge || <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{value}{unit ? <span className="ml-1 text-xs text-slate-400">{unit}</span> : null}</p>}
    </div>
  )
}

function MiniStat({ label, value, unit }: { label: string; value: number | null | undefined; unit: string }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3 text-center dark:bg-slate-800/50">
      <p className="text-[10px] uppercase tracking-wider text-slate-400">{label}</p>
      <p className="text-lg font-bold text-slate-700 dark:text-slate-200">{value?.toFixed(1) ?? '—'}<span className="ml-1 text-xs font-normal text-slate-400">{unit}</span></p>
    </div>
  )
}
