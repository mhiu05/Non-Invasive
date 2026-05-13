import { Activity, Download, Eye, FileVideo, Heart, Loader2, Upload as UploadIcon } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { BVPChart } from '@/components/BVPChart'
import { VitalSignCard } from '@/components/VitalSignCard'
import { uploadVideo } from '@/lib/api'
import { downloadCSV } from '@/lib/utils'
import type { VideoResult } from '@/types/vitals'

type State = 'idle' | 'uploading' | 'done' | 'error'

export function Upload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [state, setState] = useState<State>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<VideoResult | null>(null)
  const [errMsg, setErrMsg] = useState('')
  const [dragging, setDragging] = useState(false)

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
      const res = await uploadVideo(file, setProgress)
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
      <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
        Offline Video Analysis
      </h2>

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

          <div className="grid gap-3 sm:grid-cols-3">
            <VitalSignCard icon={Heart}    label="Heart Rate" value={result.heart_rate} unit="BPM"    color="red"   />
            <VitalSignCard icon={Eye}      label="Blink Rate" value={result.blink_rate} unit="bl/min" color="blue"  />
            <VitalSignCard icon={Activity} label="Signal SNR" value={result.snr_db}     unit="dB"     color="green" />
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
    </main>
  )
}
