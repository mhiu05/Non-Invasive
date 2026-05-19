import { Activity, ArrowLeft, Clock3, Download, Eye, FileVideo, Heart, Loader2, Radio, RefreshCw, Upload as UploadIcon } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { BVPChart } from '@/components/BVPChart'
import { VitalSignCard } from '@/components/VitalSignCard'
import { fetchHistory, uploadVideo } from '@/lib/api'
import { downloadCSV } from '@/lib/utils'
import type { HistoryRecord, VideoResult } from '@/types/vitals'

type State = 'idle' | 'uploading' | 'done' | 'error'

export function Upload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [state, setState] = useState<State>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<VideoResult | null>(null)
  const [errMsg, setErrMsg] = useState('')
  const [dragging, setDragging] = useState(false)
  const [age, setAge] = useState<number | undefined>(undefined)
  const [showHistory, setShowHistory] = useState(false)
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError('')

    try {
      const data = await fetchHistory()
      setHistoryRecords(data)
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Không thể tải lịch sử')
    } finally {
      setHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    if (showHistory) {
      loadHistory()
    }
  }, [showHistory, loadHistory])

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
    try {
      const res = await uploadVideo(file, age, setProgress)
      setResult(res)
      setState('done')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload thất bại'
      setErrMsg(msg)
      setState('error')
    }
  }, [])

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
    downloadCSV(
      result.bvp_signal.map((v, i) => ({ frame: i, bvp: v })),
      `bvp_${result.filename}.csv`,
    )
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-6 space-y-6">
      {!showHistory ? (
        <>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                Offline Video Analysis
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Upload video và xem kết quả phân tích nhịp tim, blink rate, HRV.</p>
            </div>
            <button
              type="button"
              onClick={() => setShowHistory(true)}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-900"
            >
              Xem lịch sử
            </button>
          </div>

          <div className="grid gap-3 sm:grid-cols-[1fr_120px]">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
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
              className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-indigo-400 dark:focus:ring-indigo-500/20"
            />
          </div>
          {/* Drop zone */}
          <div
            className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
              dragging
                ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
                : 'border-slate-300 hover:border-indigo-400 dark:border-slate-600'
            }`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <FileVideo className="mx-auto mb-3 text-slate-400" size={40} />
            <p className="font-medium text-slate-700 dark:text-slate-200">
              Kéo thả video vào đây hoặc{' '}
              <span className="text-indigo-600 underline">chọn file</span>
            </p>
            <p className="mt-1 text-xs text-slate-400">MP4, AVI, MOV · tối đa 100 MB</p>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={onFileChange}
            />
          </div>

          {/* Progress */}
          {state === 'uploading' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <Loader2 size={15} className="animate-spin text-indigo-500" />
                Đang xử lý… {progress}%
              </div>
              <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700">
                <div
                  className="h-2 rounded-full bg-indigo-500 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error */}
          {state === 'error' && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
              {errMsg}
            </p>
          )}

          {/* Results */}
          {state === 'done' && result && (
            <div className="space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-slate-800 dark:text-slate-100">{result.filename}</p>
                  <p className="text-xs text-slate-500">
                    {result.total_frames} frames · {result.duration_sec}s
                  </p>
                </div>
                <button
                  onClick={handleExport}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
                >
                  <Download size={13} /> Export CSV
                </button>
              </div>

              <div className="grid gap-3 sm:grid-cols-4">
                <VitalSignCard icon={Heart}    label="Heart Rate"  value={result.heart_rate} unit="BPM"    color="red"    />
                <VitalSignCard icon={Eye}      label="Blink Rate"  value={result.blink_rate} unit="bl/min" color="blue"   />
                <VitalSignCard icon={Activity} label="Signal SNR"  value={result.snr_db}     unit="dB"     color="green"  />
                <VitalSignCard icon={Radio}    label="HRV (RMSSD)" value={result.hrv_ms ?? result.rmssd_ms ?? null} unit="ms" color="purple" />
              </div>

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
        <div className="space-y-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Measurement History</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Danh sách phiên đo đã lưu từ video upload.</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowHistory(false)}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-900"
              >
                <ArrowLeft size={16} /> Back
              </button>
              <button
                type="button"
                onClick={loadHistory}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-900"
              >
                <RefreshCw size={16} /> Refresh
              </button>
            </div>
          </div>

          {historyLoading ? (
            <div className="mt-8 rounded-2xl border border-slate-200 bg-white px-6 py-10 text-center text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
              Đang tải lịch sử…
            </div>
          ) : historyError ? (
            <div className="mt-8 rounded-2xl border border-red-200 bg-red-50 px-6 py-10 text-center text-red-700 shadow-sm dark:border-red-900/50 dark:bg-red-950/50 dark:text-red-300">
              {historyError}
            </div>
          ) : historyRecords.length === 0 ? (
            <div className="mt-8 rounded-2xl border border-slate-200 bg-white px-6 py-10 text-center text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-950 dark:text-slate-400">
              Chưa có phiên đo nào. Thử upload video hoặc dùng live camera.
            </div>
          ) : (
            <div className="mt-8 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-950">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 text-sm dark:divide-slate-700">
                  <thead className="bg-slate-50 text-left text-xs uppercase tracking-wider text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                    <tr>
                      <th className="px-4 py-3">Thời gian</th>
                      <th className="px-4 py-3">Loại</th>
                      <th className="px-4 py-3">File / Session</th>
                      <th className="px-4 py-3">HR</th>
                      <th className="px-4 py-3">Blink</th>
                      <th className="px-4 py-3">SNR</th>
                      <th className="px-4 py-3">HRV</th>
                      <th className="px-4 py-3">Age</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 bg-white dark:divide-slate-700 dark:bg-slate-950">
                    {historyRecords.map((record) => (
                      <tr key={record.id} className="hover:bg-slate-50 dark:hover:bg-slate-900/80">
                        <td className="px-4 py-4 text-xs text-slate-600 dark:text-slate-300">
                          <div className="flex items-center gap-2">
                            <Clock3 size={14} />
                            <span>{new Date(record.created_at).toLocaleString('vi-VN')}</span>
                          </div>
                        </td>
                        <td className="px-4 py-4 font-medium text-slate-800 dark:text-slate-100">{record.type === 'video' ? 'Video' : 'Realtime'}</td>
                        <td className="px-4 py-4 text-slate-600 dark:text-slate-300">{record.filename ?? record.session_id ?? '—'}</td>
                        <td className="px-4 py-4 text-slate-800 dark:text-slate-100">{record.heart_rate?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-4 text-slate-800 dark:text-slate-100">{record.blink_rate?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-4 text-slate-800 dark:text-slate-100">{record.snr_db?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-4 text-slate-800 dark:text-slate-100">{record.hrv_ms?.toFixed(1) ?? '—'}</td>
                        <td className="px-4 py-4 text-slate-800 dark:text-slate-100">{record.age_group ?? '—'}</td>
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
