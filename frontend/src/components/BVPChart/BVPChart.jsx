import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip, ReferenceLine } from 'recharts'
import './BVPChart.css'

// Component BVPChart hiển thị biểu đồ tín hiệu BVP (Blood Volume Pulse) theo thời gian thực
export function BVPChart({ data }) {
  // Nếu chưa có dữ liệu, hiển thị thông báo chờ
  if (!data.length) {
    return (
      <div className="bvp-empty">
        <p className="bvp-empty-text">Waiting for signal…</p>
      </div>
    )
  }

  // Chuyển đổi mảng dữ liệu thành format phù hợp cho Recharts [{i: index, v: value}]
  const points = data.map((v, i) => ({ i, v }))

  return (
    <div className="bvp-container">
      {/* ResponsiveContainer giúp biểu đồ tự động co giãn theo kích thước của thẻ cha */}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <YAxis domain={['auto', 'auto']} hide />
          {/* Tooltip hiện giá trị khi hover vào biểu đồ */}
          <Tooltip
            formatter={(val) => [val.toFixed(3), 'BVP']}
            labelFormatter={() => ''}
            contentStyle={{ fontSize: 11 }}
          />
          {/* ReferenceLine vẽ đường nét đứt ngang tại giá trị 0 */}
          <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
          {/* Line vẽ đường cong nối các điểm dữ liệu */}
          <Line
            type="monotone"
            dataKey="v"
            stroke="#6366f1"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false} // Tắt animation để biểu đồ cập nhật mượt hơn với real-time data
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
