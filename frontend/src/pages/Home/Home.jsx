import { Activity, Camera, CheckCircle2, FileVideo, Info, MonitorPlay, ShieldCheck, Video, Github, Mail, Facebook, Phone } from 'lucide-react'
import { Link } from 'react-router-dom'
import './Home.css'

// Component Home hiển thị trang chủ giới thiệu ứng dụng
export function Home() {
  return (
    <div className="home-container">
      {/* Phần Hero: Thông điệp chính và các nút điều hướng */}
      <section className="hero-section">
        <div className="hero-badge">
          <Activity size={16} />
          <span>Non-Invasive Health Monitoring</span>
        </div>
        <h1 className="hero-title">
          Đo lường chỉ số sức khỏe <br className="hero-title-break" />
          <span className="hero-title-gradient">
            qua Camera (rPPG)
          </span>
        </h1>
        <p className="hero-subtitle">
          Hệ thống phân tích video khuôn mặt để trích xuất các chỉ số sinh tồn như nhịp tim, nhịp chớp mắt và biến thiên nhịp tim (HRV) hoàn toàn không xâm lấn.
        </p>
        <div className="hero-actions">
          {/* Nút chuyển đến trang đo lường Live */}
          <Link to="/live" className="btn btn-primary">
            <Camera size={18} /> Bắt đầu đo Live
          </Link>
          {/* Nút chuyển đến trang Upload video */}
          <Link to="/upload" className="btn btn-secondary">
            <FileVideo size={18} /> Phân tích Video Offline
          </Link>
        </div>
      </section>

      {/* Phần Features: Các tính năng chính của hệ thống */}
      <section className="features-section">
        <h2 className="section-title">
          <CheckCircle2 className="icon-indigo" /> Các chức năng chính
        </h2>
        <div className="features-grid">
          <FeatureCard
            icon={<MonitorPlay size={24} className="icon-indigo" />}
            title="Đo lường thời gian thực (Live)"
            desc="Sử dụng Webcam để theo dõi và trích xuất nhịp tim, HRV liên tục. Hệ thống tự động phân tích và hiển thị kết quả ngay lập tức."
          />
          <FeatureCard
            icon={<Video size={24} className="icon-cyan" />}
            title="Phân tích Video (Offline)"
            desc="Tải lên các đoạn video đã quay sẵn để hệ thống xử lý bất đồng bộ (async). Nhận lại báo cáo chi tiết về chất lượng tín hiệu và các chỉ số."
          />
          <FeatureCard
            icon={<ShieldCheck size={24} className="icon-emerald" />}
            title="Trợ lý AI & Lịch sử"
            desc="Chatbot AI tích hợp sẵn để giải đáp thắc mắc về sức khỏe. Lưu trữ lịch sử các phiên đo chi tiết để theo dõi tiến triển theo thời gian."
          />
        </div>
      </section>

      {/* Phần Hướng dẫn quy trình sử dụng */}
      <section className="how-to-section">
        <h2 className="section-title">Quy trình sử dụng</h2>
        <div className="steps-grid">
          <div className="steps-line" />
          <Step
            num={1}
            title="Chuẩn bị"
            desc="Ngồi ở nơi có ánh sáng tốt (đèn hắt từ phía trước). Tránh ngược sáng. Cởi bỏ khẩu trang và kính râm."
          />
          <Step
            num={2}
            title="Nhập thông tin"
            desc="Cung cấp độ tuổi để hệ thống chọn dải băng tần lọc nhiễu (Bandpass Filter) phù hợp nhất với bạn."
          />
          <Step
            num={3}
            title="Đo lường"
            desc="Nhìn thẳng vào camera, giữ đầu tĩnh tại trong ít nhất 10-15 giây để hệ thống thu thập đủ số khung hình (Buffer)."
          />
          <Step
            num={4}
            title="Xem kết quả"
            desc="Kiểm tra các chỉ số sức khỏe hiển thị trên màn hình. Nếu chỉ số SNR < 0 dB, bạn nên cải thiện ánh sáng và đo lại."
          />
        </div>
      </section>

      {/* Phần Ý nghĩa các chỉ số: Giải thích các thông số đo được */}
      <section className="metrics-section">
        <h2 className="section-title">
          <Info className="icon-indigo" /> Ý nghĩa các thông số sức khỏe
        </h2>
        <div className="metrics-grid">
          <MetricCard
            title="Heart Rate (BPM)"
            unit="Nhịp/phút"
            desc="Nhịp tim trung bình. Người trưởng thành bình thường lúc nghỉ ngơi thường có nhịp tim từ 60 đến 100 nhịp/phút."
            color="red"
          />
          <MetricCard
            title="Blink Rate"
            unit="Lần/phút"
            desc="Tần suất chớp mắt. Chớp mắt giúp làm ẩm và bảo vệ mắt. Trung bình người trưởng thành chớp mắt 10-20 lần mỗi phút."
            color="blue"
          />
          <MetricCard
            title="Signal SNR (dB)"
            unit="Decibel"
            desc="Tỷ lệ tín hiệu trên nhiễu (Signal-to-Noise Ratio). Dùng để đánh giá độ tin cậy của thuật toán rPPG. SNR > 5 dB là Xuất sắc, < 0 dB là Yếu (nên đo lại)."
            color="green"
          />
          <MetricCard
            title="HRV (RMSSD / SDNN)"
            unit="Mili-giây (ms)"
            desc="Biến thiên nhịp tim. RMSSD đo mức độ hoạt động của hệ thần kinh phó giao cảm (phục hồi). SDNN đại diện cho tổng mức độ biến thiên."
            color="purple"
          />
        </div>
      </section>

      {/* Phần Thông tin liên hệ */}
      <section className="contact-section">
        <h2 className="section-title">Thông tin liên hệ</h2>
        <div className="contact-grid">
          <a href="https://github.com/mhiu05/Non-Invasive" target="_blank" rel="noreferrer" className="contact-item">
            <Github size={20} className="contact-icon" />
            <span>github.com/mhiu05/Non-Invasive</span>
          </a>
          <a href="mailto:minhhieuhh2k5@gmail.com" className="contact-item">
            <Mail size={20} className="contact-icon" />
            <span>minhhieuhh2k5@gmail.com</span>
          </a>
          <a href="https://www.facebook.com/minhhieuhh2k5" target="_blank" rel="noreferrer" className="contact-item">
            <Facebook size={20} className="contact-icon" />
            <span>facebook.com/minhhieuhh2k5</span>
          </a>
          <div className="contact-item">
            <Phone size={20} className="contact-icon" />
            <span>0375049906</span>
          </div>
        </div>
      </section>
    </div>
  )
}

// Component phụ hiển thị một tính năng
function FeatureCard({ icon, title, desc }) {
  return (
    <div className="feature-card">
      <div className="feature-icon-wrapper">
        {icon}
      </div>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-desc">{desc}</p>
    </div>
  )
}

// Component phụ hiển thị một bước trong quy trình
function Step({ num, title, desc }) {
  return (
    <div className="step-item">
      <div className="step-num">
        {num}
      </div>
      <h4 className="step-title">{title}</h4>
      <p className="step-desc">{desc}</p>
    </div>
  )
}

// Component phụ hiển thị ý nghĩa một chỉ số
function MetricCard({ title, desc, unit, color }) {
  return (
    <div className="metric-card">
      <div className="metric-header">
        <h3 className="metric-title">{title}</h3>
        <span className={`metric-unit metric-unit-${color}`}>{unit}</span>
      </div>
      <p className="metric-desc">{desc}</p>
    </div>
  )
}
