import './VitalSignCard.css'

// Thẻ hiển thị một chỉ số sinh tồn — nhịp tim, SNR, HRV…
export function VitalSignCard({ icon: Icon, label, value, unit, color = 'cyan' }) {
  return (
    <article className={`vcard vcard--${color}`}>
      <span className="vcard__glow" aria-hidden="true" />
      <div className="vcard__icon"><Icon size={20} /></div>

      <div className="vcard__body">
        <span className="vcard__label">{label}</span>
        {value !== null ? (
          <span className="vcard__value">
            <span className="vcard__num">{value.toFixed(1)}</span>
            <span className="vcard__unit">{unit}</span>
          </span>
        ) : (
          <span className="vcard__detecting">
            <span className="vcard__detecting-dot" />
            Detecting…
          </span>
        )}
      </div>

      <span className="vcard__decor" aria-hidden="true">
        <svg viewBox="0 0 60 20" preserveAspectRatio="none">
          <polyline
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
            points="0,10 8,10 12,5 16,15 24,10 30,2 36,18 42,10 50,10 56,7 60,10"
          />
        </svg>
      </span>
    </article>
  )
}
