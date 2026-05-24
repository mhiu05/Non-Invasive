import { useState, useEffect } from 'react'
import { fetchHistory } from '@/lib/api'
import { useAuthStore } from '@/store/authStore'
import { Activity, Heart, Eye, Activity as Signal, User, Clock, Calendar } from 'lucide-react'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, Legend
} from 'recharts'
import './Profile.css'

export function Profile() {
  const user = useAuthStore((state) => state.user)
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const LIMIT = 10

  const fetchAndFormatData = async (currentOffset) => {
    const data = await fetchHistory({ limit: LIMIT, offset: currentOffset })
    const formattedData = data.map(item => {
      const date = new Date(item.created_at)
      return {
        ...item,
        formattedDate: `${date.getDate()}/${date.getMonth()+1} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`,
        heartRate: Math.round(item.heart_rate || 0),
        hrv: Math.round(item.sdnn_ms || 0),
        snr: Math.round(item.snr_db || 0)
      }
    })
    return { data, formattedData }
  }

  useEffect(() => {
    const initData = async () => {
      try {
        const { formattedData } = await fetchAndFormatData(0)
        // Sort chronologically for charts
        const sortedData = [...formattedData].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
        setHistory(sortedData)
        if (formattedData.length < LIMIT) setHasMore(false)
      } catch (err) {
        console.error("Failed to fetch history:", err)
      } finally {
        setIsLoading(false)
      }
    }
    initData()
  }, [])

  const loadMore = async () => {
    if (isLoadingMore || !hasMore) return
    setIsLoadingMore(true)
    const nextOffset = offset + LIMIT
    try {
      const { formattedData } = await fetchAndFormatData(nextOffset)
      if (formattedData.length < LIMIT) {
        setHasMore(false)
      }
      
      if (formattedData.length > 0) {
        // Append older data to the beginning of the sorted array so charts stay chronological
        // The newer items are at the end. The newly fetched items are older, so they should go at the beginning.
        const sortedNew = [...formattedData].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
        setHistory(prev => [...sortedNew, ...prev])
        setOffset(nextOffset)
      }
    } catch (err) {
      console.error("Failed to load more:", err)
    } finally {
      setIsLoadingMore(false)
    }
  }

  const latestStats = history.length > 0 ? history[history.length - 1] : null

  return (
    <div className="profile-container page-transition">
      <div className="profile-header glass-panel">
        <div className="user-info">
          <div className="avatar">
            <User size={40} />
          </div>
          <div>
            <h2>Chào, {user?.username || 'Người dùng'}</h2>
            <p>{user?.email}</p>
          </div>
        </div>
        <div className="summary-badges">
          <div className="badge">
            <Activity size={18} />
            <span>{history.length} lần đo</span>
          </div>
          {latestStats && (
            <div className="badge highlight">
              <Clock size={18} />
              <span>Đo lần cuối: {latestStats.formattedDate}</span>
            </div>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="loading-state glass-panel">Đang tải dữ liệu...</div>
      ) : history.length === 0 ? (
        <div className="empty-state glass-panel">
          <Activity size={48} className="empty-icon" />
          <h3>Chưa có dữ liệu</h3>
          <p>Hãy thực hiện đo qua Webcam hoặc Tải video lên để xem thống kê tại đây.</p>
        </div>
      ) : (
        <div className="dashboard-grid">
          {/* Quick Stats */}
          <div className="stat-card glass-panel">
            <div className="stat-header">
              <Heart className="stat-icon hr-icon" />
              <h3>Nhịp tim (gần nhất)</h3>
            </div>
            <div className="stat-value">
              {latestStats.heartRate} <span>BPM</span>
            </div>
          </div>
          
          <div className="stat-card glass-panel">
            <div className="stat-header">
              <Activity className="stat-icon hrv-icon" />
              <h3>HRV (SDNN)</h3>
            </div>
            <div className="stat-value">
              {latestStats.hrv} <span>ms</span>
            </div>
          </div>

          <div className="stat-card glass-panel">
            <div className="stat-header">
              <Signal className="stat-icon snr-icon" />
              <h3>Chất lượng (SNR)</h3>
            </div>
            <div className="stat-value">
              {latestStats.snr} <span>dB</span>
            </div>
          </div>

          <div className="stat-card glass-panel">
            <div className="stat-header">
              <Eye className="stat-icon blink-icon" />
              <h3>Chớp mắt</h3>
            </div>
            <div className="stat-value">
              {Math.round(latestStats.blink_rate)} <span>lần/p</span>
            </div>
          </div>

          {/* Charts */}
          <div className="chart-section glass-panel">
            <h3 className="section-title">Xu hướng nhịp tim (BPM)</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorHr" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="formattedDate" stroke="#94a3b8" fontSize={12} tickMargin={10} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={['dataMin - 5', 'dataMax + 5']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#ef4444' }}
                  />
                  <Area type="monotone" dataKey="heartRate" name="Nhịp tim" stroke="#ef4444" strokeWidth={3} fillOpacity={1} fill="url(#colorHr)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="chart-section glass-panel">
            <h3 className="section-title">Xu hướng HRV (SDNN)</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="formattedDate" stroke="#94a3b8" fontSize={12} tickMargin={10} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#3b82f6' }}
                  />
                  <Line type="monotone" dataKey="hrv" name="HRV (ms)" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: '#1e293b', strokeWidth: 2 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* History Table */}
          <div className="table-section glass-panel">
            <h3 className="section-title">Lịch sử chi tiết</h3>
            <div className="table-responsive">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Thời gian</th>
                    <th>Nguồn</th>
                    <th>Nhịp tim</th>
                    <th>HRV</th>
                    <th>SNR</th>
                    <th>Độ dài</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history].reverse().map((item) => (
                    <tr key={item.id}>
                      <td>
                        <div className="flex-cell">
                          <Calendar size={14} />
                          {new Date(item.created_at).toLocaleString('vi-VN')}
                        </div>
                      </td>
                      <td>
                        <span className={`source-badge ${item.type}`}>
                          {item.type === 'realtime' ? 'Webcam' : 'Video'}
                        </span>
                      </td>
                      <td className="highlight-cell">{item.heartRate} bpm</td>
                      <td>{item.hrv} ms</td>
                      <td>{item.snr} dB</td>
                      <td>{item.duration_sec}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {hasMore && (
              <div className="load-more-container">
                <button 
                  className="load-more-btn" 
                  onClick={loadMore} 
                  disabled={isLoadingMore}
                >
                  {isLoadingMore ? 'Đang tải...' : 'Tải thêm'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
