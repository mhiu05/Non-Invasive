import { Activity, Camera, CheckCircle2, FileVideo, Info, MonitorPlay, ShieldCheck, Video } from 'lucide-react'
import { Link } from 'react-router-dom'

export function Home() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:py-12 space-y-16">
      {/* Hero Section */}
      <section className="text-center space-y-6">
        <div className="mx-auto inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-600 ring-1 ring-inset ring-indigo-500/20 dark:bg-indigo-900/30 dark:text-indigo-300">
          <Activity size={16} />
          <span>Non-Invasive Health Monitoring</span>
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl dark:text-white">
          Đo lường chỉ số sức khỏe <br className="hidden sm:block" />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-cyan-500">
            qua Camera (rPPG)
          </span>
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-slate-600 dark:text-slate-400">
          Hệ thống phân tích video khuôn mặt để trích xuất các chỉ số sinh tồn như nhịp tim, nhịp chớp mắt và biến thiên nhịp tim (HRV) hoàn toàn không xâm lấn.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-4">
          <Link
            to="/live"
            className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-indigo-700 active:scale-95"
          >
            <Camera size={18} /> Bắt đầu đo Live
          </Link>
          <Link
            to="/upload"
            className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl bg-white px-6 py-3 text-sm font-semibold text-slate-700 shadow-sm ring-1 ring-inset ring-slate-300 transition-all hover:bg-slate-50 active:scale-95 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-700 dark:hover:bg-slate-700"
          >
            <FileVideo size={18} /> Phân tích Video Offline
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <CheckCircle2 className="text-indigo-500" /> Các chức năng chính
        </h2>
        <div className="grid gap-6 sm:grid-cols-3">
          <FeatureCard
            icon={<MonitorPlay size={24} className="text-indigo-500" />}
            title="Đo lường thời gian thực (Live)"
            desc="Sử dụng Webcam để theo dõi và trích xuất nhịp tim, HRV liên tục. Hệ thống tự động phân tích và hiển thị kết quả ngay lập tức."
          />
          <FeatureCard
            icon={<Video size={24} className="text-cyan-500" />}
            title="Phân tích Video (Offline)"
            desc="Tải lên các đoạn video đã quay sẵn để hệ thống xử lý bất đồng bộ (async). Nhận lại báo cáo chi tiết về chất lượng tín hiệu và các chỉ số."
          />
          <FeatureCard
            icon={<ShieldCheck size={24} className="text-emerald-500" />}
            title="Trợ lý AI & Lịch sử"
            desc="Chatbot AI tích hợp sẵn để giải đáp thắc mắc về sức khỏe. Lưu trữ lịch sử các phiên đo chi tiết để theo dõi tiến triển theo thời gian."
          />
        </div>
      </section>

      {/* How to use */}
      <section className="rounded-3xl bg-slate-100 p-8 dark:bg-slate-800/50 space-y-8">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Quy trình sử dụng</h2>
        <div className="grid gap-6 md:grid-cols-4 relative">
          <div className="hidden md:block absolute top-6 left-[10%] right-[10%] h-0.5 bg-slate-300 dark:bg-slate-700" />
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

      {/* Metrics Explanation */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Info className="text-indigo-500" /> Ý nghĩa các thông số sức khỏe
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
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
    </div>
  )
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode; title: string; desc: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-slate-50 dark:bg-slate-800">
        {icon}
      </div>
      <h3 className="mb-2 text-lg font-semibold text-slate-800 dark:text-slate-100">{title}</h3>
      <p className="text-sm leading-relaxed text-slate-500 dark:text-slate-400">{desc}</p>
    </div>
  )
}

function Step({ num, title, desc }: { num: number; title: string; desc: string }) {
  return (
    <div className="relative flex flex-col items-center text-center space-y-3">
      <div className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full border-4 border-slate-100 bg-indigo-600 text-lg font-bold text-white dark:border-slate-800">
        {num}
      </div>
      <h4 className="font-semibold text-slate-900 dark:text-slate-100">{title}</h4>
      <p className="text-sm text-slate-500 dark:text-slate-400">{desc}</p>
    </div>
  )
}

function MetricCard({ title, desc, unit, color }: { title: string; desc: string; unit: string; color: 'red' | 'blue' | 'green' | 'purple' }) {
  const colorMap = {
    red:    { bg: 'bg-red-50 dark:bg-red-900/20',    text: 'text-red-700 dark:text-red-300' },
    blue:   { bg: 'bg-blue-50 dark:bg-blue-900/20',  text: 'text-blue-700 dark:text-blue-300' },
    green:  { bg: 'bg-emerald-50 dark:bg-emerald-900/20',text: 'text-emerald-700 dark:text-emerald-300' },
    purple: { bg: 'bg-purple-50 dark:bg-purple-900/20',text: 'text-purple-700 dark:text-purple-300' },
  }
  const c = colorMap[color]

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold text-slate-800 dark:text-slate-100">{title}</h3>
        <span className={`rounded-md px-2 py-1 text-xs font-medium ${c.bg} ${c.text}`}>{unit}</span>
      </div>
      <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">{desc}</p>
    </div>
  )
}
