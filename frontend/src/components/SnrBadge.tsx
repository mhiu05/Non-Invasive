import { cn } from '@/lib/utils'
import { Signal } from 'lucide-react'
import { getSnrQuality, getSnrLabel, getSnrColor } from '@/types/vitals'

export function SnrBadge({ snr }: { snr: number | null | undefined }) {
  const q = getSnrQuality(snr)
  const c = getSnrColor(q)
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset', c.bg, c.text, c.ring)}>
      <Signal size={12} />
      {snr != null ? `${snr.toFixed(1)} dB` : '—'} · {getSnrLabel(q)}
    </span>
  )
}
