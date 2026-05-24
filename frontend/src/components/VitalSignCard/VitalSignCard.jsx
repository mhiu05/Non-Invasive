import './VitalSignCard.css'

// Component VitalSignCard hiển thị một thẻ chỉ số sinh tồn (như Nhịp tim, HRV, Blink rate)
export function VitalSignCard({ icon: Icon, label, value, unit, color }) {
  return (
    <div className={`vital-card vital-card-${color}`}>
      {/* Vùng chứa Icon của chỉ số */}
      <div className={`vital-icon vital-icon-${color}`}>
        <Icon size={22} />
      </div>
      {/* Vùng chứa nội dung: tên chỉ số và giá trị */}
      <div className="vital-content">
        <p className="vital-label">{label}</p>
        {/* Nếu đã tính được giá trị thì hiển thị, chưa thì báo đang nhận diện */}
        {value !== null ? (
          <p className={`vital-value vital-text-${color}`}>
            {value.toFixed(1)}
            <span className="vital-unit">{unit}</span>
          </p>
        ) : (
          <p className="vital-detecting">Detecting…</p>
        )}
      </div>
    </div>
  )
}
