import {
  Activity, Camera, CheckCircle2, FileVideo, Info, MonitorPlay,
  ShieldCheck, Video, Github, Mail, Facebook, Phone, Heart, Sparkles, ArrowRight,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import './Home.css'

// Trang chủ — hero editorial, giới thiệu hệ thống đo lường rPPG
export function Home() {
  return (
    <div className="home">
      {/* ─── HERO ─────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__grid">
          <div className="hero__main">
            <span className="hero__eyebrow">
              <span className="hero__eyebrow-dot" />
              <span>Non-Invasive Vital Signs · rPPG</span>
            </span>

            <h1 className="hero__title">
              Đo nhịp tim qua{' '}
              <span className="hero__title-italic">camera</span>
              <br />
              không cần chạm — chỉ cần
              <br />
              <span className="hero__title-flash">ánh sáng</span> &amp;{' '}
              <span className="hero__title-flash hero__title-flash--mint">khuôn mặt</span>.
            </h1>

            <p className="hero__lede">
              NIHealth phân tích những thay đổi mạch máu siêu nhỏ trên da mặt từ video webcam,
              trả về <strong>nhịp tim, HRV, SNR</strong> theo thời gian thực — hoàn toàn không xâm lấn.
            </p>

            <div className="hero__cta">
              <Link to="/live" className="btn btn--primary">
                <span className="btn__glow" aria-hidden="true" />
                <Camera size={17} />
                <span>Bắt đầu đo Live</span>
                <ArrowRight size={14} className="btn__arrow" />
              </Link>
              <Link to="/upload" className="btn btn--ghost">
                <FileVideo size={17} />
                <span>Phân tích video offline</span>
              </Link>
            </div>

            <ul className="hero__stats">
              <li>
                <span className="hero__stat-num">30<span className="hero__stat-unit">fps</span></span>
                <span className="hero__stat-label">Pipeline thời gian thực</span>
              </li>
              <li>
                <span className="hero__stat-num">&lt;15<span className="hero__stat-unit">s</span></span>
                <span className="hero__stat-label">Warm-up buffer</span>
              </li>
              <li>
                <span className="hero__stat-num">0<span className="hero__stat-unit">tiếp xúc</span></span>
                <span className="hero__stat-label">Phi xâm lấn</span>
              </li>
            </ul>
          </div>

          {/* Visual decorative module */}
          <aside className="hero__visual" aria-hidden="true">
            <div className="hero__visual-card">
              <div className="hero__visual-header">
                <span className="hero__visual-dot hero__visual-dot--red" />
                <span className="hero__visual-dot hero__visual-dot--amber" />
                <span className="hero__visual-dot hero__visual-dot--green" />
                <span className="hero__visual-label">live · session 01</span>
              </div>

              <div className="hero__pulse">
                <Heart className="hero__pulse-icon" size={28} />
                <div className="hero__pulse-readout">
                  <span className="hero__pulse-num">72.4</span>
                  <span className="hero__pulse-unit">bpm</span>
                </div>
                <span className="hero__pulse-trend">↗ +3 trong 30s qua</span>
              </div>

              <svg className="hero__waveform" viewBox="0 0 320 80" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="waveGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%"  stopColor="#22d3ee" stopOpacity="0.2" />
                    <stop offset="50%" stopColor="#22d3ee" stopOpacity="1" />
                    <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.7" />
                  </linearGradient>
                </defs>
                <polyline
                  fill="none"
                  stroke="url(#waveGrad)"
                  strokeWidth="2"
                  points="0,40 18,40 26,30 34,40 48,40 58,12 66,60 74,40 90,40 110,40 116,32 122,40 138,40 148,18 156,52 164,40 180,40 196,40 206,28 214,40 230,40 240,16 248,58 256,40 272,40 290,40 300,30 308,40 320,40"
                />
              </svg>

              <div className="hero__visual-grid">
                <div className="hero__vchip hero__vchip--cyan">
                  <span className="hero__vchip-label">SNR</span>
                  <span className="hero__vchip-value mono">+8.4 dB</span>
                </div>
                <div className="hero__vchip hero__vchip--mint">
                  <span className="hero__vchip-label">HRV</span>
                  <span className="hero__vchip-value mono">42.8 ms</span>
                </div>
                <div className="hero__vchip hero__vchip--violet">
                  <span className="hero__vchip-label">SDNN</span>
                  <span className="hero__vchip-value mono">51.2 ms</span>
                </div>
              </div>
            </div>

            <div className="hero__visual-ambient" />
          </aside>
        </div>
      </section>

      {/* ─── FEATURES ─────────────────────────────────────────────── */}
      <section className="section">
        <div className="section__header">
          <span className="eyebrow">01 · Capabilities</span>
          <h2 className="section__title">
            Ba module, một <em className="serif">pipeline</em>
          </h2>
          <p className="section__sub">
            Từ webcam thời gian thực đến phân tích video offline, mọi luồng đều dùng chung
            engine rPPG đã được hiệu chỉnh.
          </p>
        </div>

        <div className="feature-grid">
          <FeatureCard
            number="01"
            tone="cyan"
            icon={<MonitorPlay size={20} />}
            title="Đo lường Realtime"
            desc="Webcam → nhịp tim, HRV, SNR liên tục. Cảnh báo chất lượng tín hiệu và bộ đệm thông minh."
            tags={['WebSocket', 'Face Tracking', '30 fps']}
          />
          <FeatureCard
            number="02"
            tone="violet"
            icon={<Video size={20} />}
            title="Phân tích Video"
            desc="Upload video sẵn. Hệ thống xử lý async, trả báo cáo chi tiết với SDNN, RMSSD, pNN50."
            tags={['Async Job', 'CSV Export', 'HRV ngoài lề']}
          />
          <FeatureCard
            number="03"
            tone="mint"
            icon={<ShieldCheck size={20} />}
            title="Trợ lý AI · Lịch sử"
            desc="Chatbot tra cứu kiến thức rPPG, mô tả triệu chứng. Lịch sử lưu trữ mỗi phiên đo."
            tags={['RAG', 'Gemini', 'Supabase']}
          />
        </div>
      </section>

      {/* ─── WORKFLOW ────────────────────────────────────────────── */}
      <section className="section workflow">
        <div className="section__header">
          <span className="eyebrow">02 · Workflow</span>
          <h2 className="section__title">Bốn bước, dưới một phút</h2>
        </div>

        <ol className="steps">
          <li className="step">
            <span className="step__num">01</span>
            <h4 className="step__title">Ánh sáng</h4>
            <p className="step__desc">Ngồi đối diện nguồn sáng. Tránh ngược sáng. Tháo kính râm và khẩu trang.</p>
          </li>
          <li className="step">
            <span className="step__num">02</span>
            <h4 className="step__title">Khai báo tuổi</h4>
            <p className="step__desc">Cung cấp độ tuổi để chọn dải Bandpass Filter sinh lý phù hợp.</p>
          </li>
          <li className="step">
            <span className="step__num">03</span>
            <h4 className="step__title">Giữ yên 10–15s</h4>
            <p className="step__desc">Nhìn thẳng vào camera đến khi buffer đầy. Hệ thống bắt đầu tính toán.</p>
          </li>
          <li className="step step--final">
            <span className="step__num">04</span>
            <h4 className="step__title">Đọc kết quả</h4>
            <p className="step__desc">SNR &gt; 5 dB nghĩa là tín hiệu tốt. Dưới 0 dB nên cải thiện ánh sáng.</p>
          </li>
        </ol>
      </section>

      {/* ─── METRICS DICTIONARY ─────────────────────────────────── */}
      <section className="section">
        <div className="section__header">
          <span className="eyebrow">03 · Dictionary</span>
          <h2 className="section__title">Ý nghĩa các chỉ số</h2>
        </div>

        <div className="metric-grid">
          <MetricCard
            tone="crimson"
            title="Heart Rate"
            unit="BPM"
            range="60 – 100"
            desc="Nhịp tim trung bình. Người lớn lúc nghỉ ngơi thường trong khoảng 60–100 nhịp/phút."
          />
          <MetricCard
            tone="cyan"
            title="Signal SNR"
            unit="dB"
            range="≥ +5 dB"
            desc="Tỷ lệ Tín hiệu/Nhiễu. SNR > 5: xuất sắc · 0–5: khá · < 0: kém, nên đo lại trong điều kiện sáng tốt."
          />
          <MetricCard
            tone="violet"
            title="HRV — RMSSD"
            unit="ms"
            range="20 – 80"
            desc="Biến thiên nhịp tim ngắn hạn. Phản ánh hoạt động phó giao cảm (parasympathetic recovery)."
          />
          <MetricCard
            tone="mint"
            title="SDNN"
            unit="ms"
            range="30 – 100"
            desc="Tổng độ biến thiên nhịp tim. Càng cao thường gắn với hệ thần kinh tự chủ khoẻ mạnh."
          />
        </div>
      </section>

      {/* ─── CONTACT ─────────────────────────────────────────────── */}
      <section className="section contact">
        <div className="section__header">
          <span className="eyebrow">04 · Get in touch</span>
          <h2 className="section__title">Liên hệ tác giả</h2>
          <p className="section__sub">Mọi đóng góp, báo lỗi hoặc cộng tác đều được hoan nghênh.</p>
        </div>

        <div className="contact__grid">
          <ContactItem
            href="https://github.com/mhiu05/Non-Invasive"
            icon={<Github size={18} />}
            label="GitHub"
            value="mhiu05/Non-Invasive"
          />
          <ContactItem
            href="mailto:minhhieuhh2k5@gmail.com"
            icon={<Mail size={18} />}
            label="Email"
            value="minhhieuhh2k5@gmail.com"
          />
          <ContactItem
            href="https://www.facebook.com/minhhieuhh2k5"
            icon={<Facebook size={18} />}
            label="Facebook"
            value="@minhhieuhh2k5"
          />
          <ContactItem
            icon={<Phone size={18} />}
            label="Phone"
            value="0375 049 906"
          />
        </div>
      </section>

      <footer className="footer">
        <span className="footer__mark"><Sparkles size={12} /></span>
        <span>
          NIHealth · <span className="serif">built with research</span> ·{' '}
          <span className="mono">v1.0</span>
        </span>
      </footer>
    </div>
  )
}

/* ────────────────────────── Sub-components ──────────────────────── */

function FeatureCard({ number, tone, icon, title, desc, tags }) {
  return (
    <article className={`feature feature--${tone}`}>
      <span className="feature__corner" aria-hidden="true" />
      <header className="feature__head">
        <span className="feature__num">{number}</span>
        <span className="feature__icon">{icon}</span>
      </header>
      <h3 className="feature__title">{title}</h3>
      <p className="feature__desc">{desc}</p>
      <ul className="feature__tags">
        {tags.map(t => <li key={t} className="feature__tag">{t}</li>)}
      </ul>
    </article>
  )
}

function MetricCard({ tone, title, unit, range, desc }) {
  return (
    <article className={`metric metric--${tone}`}>
      <div className="metric__head">
        <h3 className="metric__title">{title}</h3>
        <span className="metric__unit">{unit}</span>
      </div>
      <p className="metric__range mono">Khoảng tham chiếu · {range}</p>
      <p className="metric__desc">{desc}</p>
    </article>
  )
}

function ContactItem({ href, icon, label, value }) {
  const Wrapper = href ? 'a' : 'div'
  const props = href ? { href, target: href.startsWith('http') ? '_blank' : undefined, rel: 'noreferrer' } : {}
  return (
    <Wrapper className="contact__item" {...props}>
      <span className="contact__icon">{icon}</span>
      <span className="contact__text">
        <span className="contact__label">{label}</span>
        <span className="contact__value">{value}</span>
      </span>
      {href && <ArrowRight size={14} className="contact__arrow" />}
    </Wrapper>
  )
}
