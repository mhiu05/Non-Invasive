import { useCallback, useEffect, useState } from 'react'
import { ArrowLeft, Clock3, Filter, RefreshCw, X } from 'lucide-react'
import { fetchHistory } from '@/lib/api'
import { SnrBadge } from '@/components/SnrBadge/SnrBadge'
import './HistoryView.css'

// Component HistoryView dùng để hiển thị danh sách và chi tiết các phiên đo
export function HistoryView({ onBack }) {
  // Các state lưu trữ dữ liệu danh sách lịch sử và trạng thái tải
  const [historyRecords, setHistoryRecords] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  // State lưu phiên đo đang được chọn để xem chi tiết
  const [selectedRecord, setSelectedRecord] = useState(null)

  // Các state dành cho tính năng lọc danh sách
  const [filterType, setFilterType] = useState('') // Lọc theo loại (video/realtime)
  const [filterStart, setFilterStart] = useState('') // Lọc từ ngày
  const [filterEnd, setFilterEnd] = useState('') // Lọc đến ngày
  const [showFilters, setShowFilters] = useState(false) // Đóng/mở khung bộ lọc

  // Hàm gọi API để lấy danh sách lịch sử đo dựa theo bộ lọc
  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const filters = {}
      if (filterType) filters.type = filterType
      if (filterStart) filters.start_at = new Date(filterStart).toISOString()
      if (filterEnd) filters.end_at = new Date(filterEnd).toISOString()
      const data = await fetchHistory(filters)
      setHistoryRecords(data)
    } catch (err) {
      if (err.response?.status === 401) {
        setHistoryError('Bạn hãy đăng nhập để lưu và xem lịch sử đo')
      } else {
        setHistoryError(err instanceof Error ? err.message : 'Không thể tải lịch sử')
      }
    } finally {
      setHistoryLoading(false)
    }
  }, [filterType, filterStart, filterEnd])

  // Tự động tải dữ liệu khi component được render lần đầu hoặc khi bộ lọc thay đổi
  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  return (
    <div className="history-view-container">
      {/* Modal hiển thị chi tiết của một phiên đo khi click vào hàng trong bảng */}
      {selectedRecord && (
        <div className="history-modal-overlay" onClick={() => setSelectedRecord(null)}>
          <div className="history-modal-content" onClick={e => e.stopPropagation()}>
            <button onClick={() => setSelectedRecord(null)} className="history-modal-close"><X size={18} /></button>
            <h3 className="history-modal-title">Chi tiết phiên đo</h3>
            <p className="history-modal-time">{new Date(selectedRecord.created_at).toLocaleString('vi-VN')}</p>
            <div className="history-detail-grid">
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

      {/* Header và các nút hành động (lọc, quay lại, làm mới) */}
      <div className="history-header">
        <div>
          <h2 className="history-title">Measurement History</h2>
          <p className="history-subtitle">Danh sách phiên đo đã lưu (video & realtime).</p>
        </div>
        <div className="history-actions">
          <button type="button" onClick={() => setShowFilters(!showFilters)} className="history-btn">
            <Filter size={14} /> Lọc
          </button>
          <button type="button" onClick={onBack} className="history-btn">
            <ArrowLeft size={14} /> Back
          </button>
          <button type="button" onClick={loadHistory} className="history-btn">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Panel bộ lọc */}
      {showFilters && (
        <div className="history-filters-panel">
          <div className="history-filters-grid">
            <div>
              <label className="history-filter-label">Loại</label>
              <select value={filterType} onChange={e => setFilterType(e.target.value)} className="history-filter-input">
                <option value="">Tất cả</option>
                <option value="video">Video</option>
                <option value="realtime">Realtime</option>
              </select>
            </div>
            <div>
              <label className="history-filter-label">Từ ngày</label>
              <input type="date" value={filterStart} onChange={e => setFilterStart(e.target.value)} className="history-filter-input" />
            </div>
            <div>
              <label className="history-filter-label">Đến ngày</label>
              <input type="date" value={filterEnd} onChange={e => setFilterEnd(e.target.value)} className="history-filter-input" />
            </div>
          </div>
          <div className="history-filters-actions">
            <button onClick={() => { setFilterType(''); setFilterStart(''); setFilterEnd('') }} className="history-filter-clear">Xóa bộ lọc</button>
            <button onClick={loadHistory} className="history-filter-apply">Áp dụng</button>
          </div>
        </div>
      )}

      {/* Phần hiển thị danh sách hoặc trạng thái đang tải/lỗi */}
      {historyLoading ? (
        <div className="history-state-card">Đang tải lịch sử…</div>
      ) : historyError ? (
        <div className="history-error-card">{historyError}</div>
      ) : historyRecords.length === 0 ? (
        <div className="history-state-card">Chưa có phiên đo nào.</div>
      ) : (
        <div className="history-table-container">
          {/* Bảng hiển thị danh sách các phiên đo */}
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead className="history-thead">
                <tr>
                  <th>Thời gian</th>
                  <th>Loại</th>
                  <th>File / Session</th>
                  <th>HR</th>
                  <th>Blink</th>
                  <th>SNR</th>
                  <th>HRV</th>
                  <th>Thời lượng</th>
                  <th>Tuổi</th>
                </tr>
              </thead>
              <tbody className="history-tbody">
                {historyRecords.map((r) => (
                  <tr key={r.id} className="history-tr" onClick={() => setSelectedRecord(r)}>
                    <td className="history-td history-td-time">
                      <div className="history-time-wrap"><Clock3 size={12} />{new Date(r.created_at).toLocaleString('vi-VN')}</div>
                    </td>
                    <td className="history-td history-td-type">{r.type === 'video' ? 'Video' : 'Realtime'}</td>
                    <td className="history-td history-td-file">{r.filename ?? r.session_id ?? '—'}</td>
                    <td className="history-td history-td-value">{r.heart_rate?.toFixed(1) ?? '—'}</td>
                    <td className="history-td history-td-value">{r.blink_rate?.toFixed(1) ?? '—'}</td>
                    <td className="history-td"><SnrBadge snr={r.snr_db} /></td>
                    <td className="history-td history-td-value">{r.hrv_ms?.toFixed(1) ?? '—'}</td>
                    <td className="history-td history-td-value">{r.duration_sec ? `${r.duration_sec}s` : '—'}</td>
                    <td className="history-td history-td-value">{r.age_group ?? '—'}</td>
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

// Component phụ để render từng dòng thông tin chi tiết trong modal
function DetailRow({ label, value, unit, badge }) {
  return (
    <div className="history-detail-row">
      <p className="history-detail-label">{label}</p>
      {badge || <p className="history-detail-value">{value}{unit ? <span className="history-detail-unit">{unit}</span> : null}</p>}
    </div>
  )
}
