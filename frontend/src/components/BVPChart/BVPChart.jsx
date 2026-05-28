import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip, ReferenceLine } from 'recharts'
import './BVPChart.css'

// Biểu đồ tín hiệu BVP (Blood Volume Pulse) theo thời gian thực
export function BVPChart({ data }) {
  if (!data.length) {
    return (
      <div className="bvp-empty">
        <span className="bvp-empty__dot" />
        <p className="bvp-empty__text">Đang chờ tín hiệu…</p>
      </div>
    )
  }

  const points = data.map((v, i) => ({ i, v }))

  return (
    <div className="bvp-container">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 6, right: 4, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="bvpGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"  stopColor="#22d3ee" />
              <stop offset="50%" stopColor="#67e8f9" />
              <stop offset="100%" stopColor="#a78bfa" />
            </linearGradient>
          </defs>
          <YAxis domain={['auto', 'auto']} hide />
          <Tooltip
            formatter={(val) => [val.toFixed(3), 'BVP']}
            labelFormatter={() => ''}
            contentStyle={{
              fontSize: 11,
              backgroundColor: 'rgba(8, 12, 22, 0.92)',
              border: '1px solid rgba(34, 211, 238, 0.30)',
              borderRadius: 8,
              color: '#e5f2ff',
              boxShadow: '0 8px 24px -10px rgba(0,0,0,0.6)',
            }}
            cursor={{ stroke: 'rgba(34, 211, 238, 0.45)', strokeWidth: 1, strokeDasharray: '3 3' }}
          />
          <ReferenceLine y={0} stroke="rgba(148, 175, 220, 0.25)" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="v"
            stroke="url(#bvpGrad)"
            strokeWidth={1.8}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
