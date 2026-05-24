import { Signal } from 'lucide-react'
import { getSnrQuality, getSnrLabel, getSnrColor } from '@/lib/vitals'
import './SnrBadge.css'

// Component SnrBadge hiển thị chỉ số Tỷ lệ Tín hiệu/Nhiễu (SNR)
export function SnrBadge({ snr }) {
  // Lấy đánh giá chất lượng (Tốt, Khá, Kém) từ giá trị SNR
  const q = getSnrQuality(snr)
  // Lấy màu sắc tương ứng với chất lượng
  const c = getSnrColor(q)
  
  return (
    <span className={`snr-badge ${c}`}>
      <Signal size={12} />
      {/* Hiển thị giá trị SNR (làm tròn 1 chữ số thập phân) và nhãn chất lượng */}
      {snr != null ? `${snr.toFixed(1)} dB` : '—'} · {getSnrLabel(q)}
    </span>
  )
}
