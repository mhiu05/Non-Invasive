import { useState, useEffect } from 'react'
import { fetchHistory } from '@/lib/api'
import { useAuth } from '@/features/auth/AuthProvider'
import { Activity, Heart, User, Clock, Calendar, TrendingUp, Sparkles } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area,
} from 'recharts'
import './Profile.css'

export function Profile() {
  const { user } = useAuth()
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const LIMIT = 10

  const fetchAndFormatData = async (currentOffset) => {
    const data = await fetchHistory({ limit: LIMIT, offset: currentOffset })
    const formattedData = data.map((item) => {
      const date = new Date(item.created_at)
      return {
        ...item,
        formattedDate: `${date.getDate()}/${date.getMonth() + 1} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`,
        heartRate: Math.round(item.heart_rate || 0),
        hrv: Math.round(item.sdnn_ms || 0),
        snr: Math.round(item.snr_db || 0),
      }
    })
    return { data, formattedData }
  }

  useEffect(() => {
    const initData = async () => {
      try {
        const { formattedData } = await fetchAndFormatData(0)
        const sortedData = [...formattedData].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
        setHistory(sortedData)
        if (formattedData.length < LIMIT) setHasMore(false)
      } catch (err) {
        console.error('Failed to fetch history:', err)
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
      if (formattedData.length < LIMIT) setHasMore(false)
      if (formattedData.length > 0) {
        const sortedNew = [...formattedData].sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
        setHistory((prev) => [...sortedNew, ...prev])
        setOffset(nextOffset)
      }
    } catch (err) {
      console.error('Failed to load more:', err)
    } finally {
      setIsLoadingMore(false)
    }
  }

  const latestStats = history.length > 0 ? history[history.length - 1] : null
  const displayName =
    user?.user_metadata?.full_name ||
    user?.user_metadata?.username ||
    user?.email?.split('@')[0] ||
    'Người dùng'
  const initial = (displayName || 'N').trim().charAt(0).toUpperCase()

  return (
    <div className="profile">
      {/* ── Hero header ──────────────────────────────────── */}
      <header className="profile__hero">
        <div className="profile__hero-bg" aria-hidden="true" />

        <div className="profile__identity">
          <div className="profile__avatar">
            <span className="profile__avatar-initial">{initial}</span>
            <span className="profile__avatar-ring" aria-hidden="true" />
          </div>

          <div className="profile__identity-text">
            <span className="eyebrow">Patient · profile</span>
            <h2 className="profile__name">{displayName}</h2>
            <p className="profile__email">{user?.email}</p>
            {(user?.user_metadata?.gender || user?.user_metadata?.dob) && (
              <p className="profile__meta">
                {user?.user_metadata?.gender === 'nam'
                  ? 'Nam'
                  : user?.user_metadata?.gender === 'nu'
                  ? 'Nữ'
                  : 'Khác'}
                {user?.user_metadata?.dob && (
                  <>
                    {' · '}
                    <span className="mono">Sinh {new Date(user.user_metadata.dob).getFullYear()}</span>
                  </>
                )}
              </p>
            )}
          </div>
        </div>

        <div className="profile__hero-stats">
          <div className="profile__hero-stat">
            <span className="profile__hero-stat-icon"><Activity size={15} /></span>
            <div>
              <span className="profile__hero-stat-num mono">{history.length}</span>
              <span className="profile__hero-stat-label">lần đo</span>
            </div>
          </div>
          {latestStats && (
            <div className="profile__hero-stat profile__hero-stat--accent">
              <span className="profile__hero-stat-icon"><Clock size={15} /></span>
              <div>
                <span className="profile__hero-stat-num mono">{latestStats.formattedDate}</span>
                <span className="profile__hero-stat-label">đo gần nhất</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* ── Content ──────────────────────────────────────── */}
      {isLoading ? (
        <div className="profile__state">
          <span className="profile__state-spinner" />
          <p>Đang tải dữ liệu…</p>
        </div>
      ) : history.length === 0 ? (
        <div className="profile__empty">
          <div className="profile__empty-icon"><Activity size={36} /></div>
          <h3>Chưa có dữ liệu</h3>
          <p>Hãy thực hiện đo qua Webcam hoặc Upload video để xem thống kê chi tiết tại đây.</p>
        </div>
      ) : (
        <div className="profile__grid">
          {/* Stat cards */}
          <StatCard
            tone="crimson"
            icon={<Heart size={18} />}
            label="Nhịp tim · gần nhất"
            value={latestStats.heartRate}
            unit="BPM"
            trend={trendOf(history, 'heartRate')}
          />
          <StatCard
            tone="cyan"
            icon={<Activity size={18} />}
            label="HRV · SDNN"
            value={latestStats.hrv}
            unit="ms"
            trend={trendOf(history, 'hrv')}
          />
          <StatCard
            tone="mint"
            icon={<Sparkles size={18} />}
            label="Chất lượng tín hiệu · SNR"
            value={latestStats.snr}
            unit="dB"
            trend={trendOf(history, 'snr')}
          />

          {/* Charts */}
          <section className="chartcard chartcard--wide">
            <header className="chartcard__head">
              <div>
                <span className="eyebrow">Trend · BPM</span>
                <h3 className="chartcard__title">Xu hướng <em className="serif">nhịp tim</em></h3>
              </div>
              <TrendingUp size={18} className="chartcard__icon" />
            </header>
            <div className="chartcard__body">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history} margin={{ top: 10, right: 12, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="hrFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.45" />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,175,220,0.10)" vertical={false} />
                  <XAxis dataKey="formattedDate" stroke="#7b87a8" fontSize={11} tickMargin={8} axisLine={false} tickLine={false} />
                  <YAxis stroke="#7b87a8" fontSize={11} domain={['dataMin - 5', 'dataMax + 5']} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    itemStyle={{ color: '#f43f5e' }}
                    cursor={{ stroke: 'rgba(244, 63, 94, 0.4)', strokeWidth: 1, strokeDasharray: '3 3' }}
                  />
                  <Area type="monotone" dataKey="heartRate" name="Nhịp tim" stroke="#f43f5e" strokeWidth={2.4} fill="url(#hrFill)" dot={false} activeDot={{ r: 5, fill: '#f43f5e', strokeWidth: 2, stroke: '#08101e' }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="chartcard chartcard--wide">
            <header className="chartcard__head">
              <div>
                <span className="eyebrow">Trend · HRV</span>
                <h3 className="chartcard__title">Xu hướng <em className="serif">SDNN</em></h3>
              </div>
              <TrendingUp size={18} className="chartcard__icon" />
            </header>
            <div className="chartcard__body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 10, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,175,220,0.10)" vertical={false} />
                  <XAxis dataKey="formattedDate" stroke="#7b87a8" fontSize={11} tickMargin={8} axisLine={false} tickLine={false} />
                  <YAxis stroke="#7b87a8" fontSize={11} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    itemStyle={{ color: '#22d3ee' }}
                    cursor={{ stroke: 'rgba(34, 211, 238, 0.4)', strokeWidth: 1, strokeDasharray: '3 3' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="hrv"
                    name="HRV (ms)"
                    stroke="#22d3ee"
                    strokeWidth={2.4}
                    dot={{ r: 3, fill: '#0a1224', strokeWidth: 2, stroke: '#22d3ee' }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* History table */}
          <section className="historyblock">
            <header className="historyblock__head">
              <div>
                <span className="eyebrow">Records · all sessions</span>
                <h3 className="chartcard__title">Lịch sử <em className="serif">chi tiết</em></h3>
              </div>
            </header>
            <div className="historyblock__scroll">
              <table className="ptable">
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
                        <span className="ptable__time">
                          <Calendar size={12} />
                          {new Date(item.created_at).toLocaleString('vi-VN')}
                        </span>
                      </td>
                      <td>
                        <span className={`ptable__src ptable__src--${item.type}`}>
                          {item.type === 'realtime' ? 'Webcam' : 'Video'}
                        </span>
                      </td>
                      <td className="ptable__hr mono">{item.heartRate} <span>bpm</span></td>
                      <td className="mono">{item.hrv} <span>ms</span></td>
                      <td className="mono">{item.snr} <span>dB</span></td>
                      <td className="mono">{item.duration_sec}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {hasMore && (
              <div className="historyblock__footer">
                <button className="profile__loadmore" onClick={loadMore} disabled={isLoadingMore}>
                  {isLoadingMore ? 'Đang tải…' : 'Tải thêm'}
                </button>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  )
}

/* ── Sub-components ────────────────────────────────────── */

function StatCard({ tone, icon, label, value, unit, trend }) {
  return (
    <article className={`statcard statcard--${tone}`}>
      <span className="statcard__glow" aria-hidden="true" />
      <header className="statcard__head">
        <span className="statcard__icon">{icon}</span>
        <span className="statcard__label">{label}</span>
      </header>
      <div className="statcard__value">
        <span className="statcard__num">{value}</span>
        <span className="statcard__unit">{unit}</span>
      </div>
      {trend && (
        <span className={`statcard__trend ${trend.direction === 'up' ? 'statcard__trend--up' : trend.direction === 'down' ? 'statcard__trend--down' : ''}`}>
          {trend.text}
        </span>
      )}
    </article>
  )
}

/* ── Helpers ──────────────────────────────────────────── */

function trendOf(history, key) {
  if (history.length < 2) return null
  const a = history[history.length - 1][key]
  const b = history[history.length - 2][key]
  const diff = a - b
  if (!isFinite(diff) || diff === 0) return { direction: 'flat', text: '→ ổn định' }
  const sign = diff > 0 ? '↑' : '↓'
  return {
    direction: diff > 0 ? 'up' : 'down',
    text: `${sign} ${Math.abs(diff).toFixed(0)} so với lần trước`,
  }
}

const tooltipStyle = {
  backgroundColor: 'rgba(8, 12, 22, 0.92)',
  border: '1px solid rgba(34, 211, 238, 0.30)',
  borderRadius: 8,
  color: '#e5f2ff',
  fontFamily: 'JetBrains Mono, ui-monospace, monospace',
  fontSize: 11,
  boxShadow: '0 8px 24px -10px rgba(0,0,0,0.6)',
}
