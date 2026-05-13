import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip, ReferenceLine } from 'recharts'

interface Props {
  data: number[]
}

export function BVPChart({ data }: Props) {
  if (!data.length) {
    return (
      <div className="flex h-32 items-center justify-center rounded-xl border border-dashed border-slate-300 dark:border-slate-700">
        <p className="text-sm text-slate-400">Waiting for signal…</p>
      </div>
    )
  }

  const points = data.map((v, i) => ({ i, v }))

  return (
    <div className="h-32">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <YAxis domain={['auto', 'auto']} hide />
          <Tooltip
            formatter={(val: number) => [val.toFixed(3), 'BVP']}
            labelFormatter={() => ''}
            contentStyle={{ fontSize: 11 }}
          />
          <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="v"
            stroke="#6366f1"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
