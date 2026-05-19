import { useCallback, useEffect, useState } from 'react'
import { ArrowLeft, Clock3, Filter, RefreshCw, X } from 'lucide-react'
import { fetchHistory } from '@/lib/api'
import type { HistoryRecord, HistoryFilters } from '@/types/vitals'
import { SnrBadge } from '@/components/SnrBadge'

interface HistoryViewProps {
  onBack: () => void
}

export function HistoryView({ onBack }: HistoryViewProps) {
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null)

  // Filters
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
    loadHistory()
  }, [loadHistory])

  return (
    <div className="space-y-4">
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

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Measurement History</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">Danh sách phiên đo đã lưu (video & realtime).</p>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => setShowFilters(!showFilters)} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
            <Filter size={14} /> Lọc
          </button>
          <button type="button" onClick={onBack} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
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
                  <th className="px-4 py-3">File / Session</th>
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
  )
}

function DetailRow({ label, value, unit, badge }: { label: string; value: string; unit?: string; badge?: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/50">
      <p className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500">{label}</p>
      {badge || <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{value}{unit ? <span className="ml-1 text-xs text-slate-400">{unit}</span> : null}</p>}
    </div>
  )
}
