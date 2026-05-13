import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'

interface Props {
  icon: LucideIcon
  label: string
  value: number | null
  unit: string
  color: 'red' | 'blue' | 'green' | 'purple'
}

const colorMap = {
  red:    { bg: 'bg-red-50 dark:bg-red-900/20',    text: 'text-red-600 dark:text-red-400',    icon: 'text-red-500' },
  blue:   { bg: 'bg-blue-50 dark:bg-blue-900/20',  text: 'text-blue-600 dark:text-blue-400',  icon: 'text-blue-500' },
  green:  { bg: 'bg-green-50 dark:bg-green-900/20',text: 'text-green-600 dark:text-green-400',icon: 'text-green-500' },
  purple: { bg: 'bg-purple-50 dark:bg-purple-900/20',text: 'text-purple-600 dark:text-purple-400',icon: 'text-purple-500' },
}

export function VitalSignCard({ icon: Icon, label, value, unit, color }: Props) {
  const c = colorMap[color]
  return (
    <div className={cn('flex items-center gap-3 rounded-xl p-4', c.bg)}>
      <div className={cn('shrink-0', c.icon)}>
        <Icon size={22} />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
        {value !== null ? (
          <p className={cn('text-2xl font-bold leading-tight', c.text)}>
            {value.toFixed(1)}
            <span className="ml-1 text-sm font-normal text-slate-500">{unit}</span>
          </p>
        ) : (
          <p className="text-sm text-slate-400 animate-pulse_slow">Detecting…</p>
        )}
      </div>
    </div>
  )
}
