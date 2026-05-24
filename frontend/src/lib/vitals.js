// Phân loại chất lượng tín hiệu (SNR) dựa trên ngưỡng (dB)
export function getSnrQuality(snr) {
  if (snr == null) return 'poor'
  if (snr >= 5) return 'excellent'
  if (snr >= 2) return 'good'
  if (snr >= 0) return 'fair'
  return 'poor'
}

// Lấy nhãn tiếng Việt tương ứng cho chất lượng tín hiệu
export function getSnrLabel(q) {
  switch (q) {
    case 'excellent': return 'Xuất sắc'
    case 'good':      return 'Tốt'
    case 'fair':      return 'Trung bình'
    case 'poor':      return 'Yếu'
  }
}

// Lấy class CSS màu sắc tương ứng cho chất lượng tín hiệu
export function getSnrColor(q) {
  switch (q) {
    case 'excellent': return 'snr-excellent'
    case 'good':      return 'snr-good'
    case 'fair':      return 'snr-fair'
    case 'poor':      return 'snr-poor'
  }
}
