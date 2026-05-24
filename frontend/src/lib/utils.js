// Tiện ích nối các class CSS (bỏ qua các giá trị falsy như false, null, undefined)
export function cn(...inputs) {
  return inputs.filter(Boolean).join(' ')
}

// Làm tròn số thập phân (mặc định 1 chữ số thập phân)
export function round(val, dec = 1) {
  return Math.round(val * 10 ** dec) / 10 ** dec
}

// Hàm tải xuống dữ liệu mảng đối tượng dưới định dạng file CSV
export function downloadCSV(data, filename) {
  if (!data.length) return
  const headers = Object.keys(data[0])
  const rows = data.map((r) => headers.map((h) => r[h]).join(','))
  const csv = [headers.join(','), ...rows].join('\n')
  
  // Tạo blob chứa dữ liệu CSV và tự động click tải về
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
