import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { User, Lock, Mail, Calendar, Hash, Type, ArrowRight, Activity, Sparkles } from 'lucide-react'
import { supabase } from '@/lib/supabase'
import './Register.css'

export function Register() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [gender, setGender] = useState('nam')
  const [dob, setDob] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { username, full_name: fullName, gender, dob } },
      })
      if (error) { setError(error.message); return }
      navigate('/login', {
        state: { message: 'Đăng ký thành công! Vui lòng kiểm tra hộp thư Email để xác nhận tài khoản trước khi đăng nhập.' },
      })
    } catch (err) {
      setError('Đã có lỗi xảy ra khi đăng ký')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="auth">
      <div className="auth__shell">
        <aside className="auth__aside" aria-hidden="true">
          <div className="auth__aside-inner">
            <span className="auth__brand">
              <Activity size={16} />
              <span>NIHealth</span>
            </span>
            <p className="auth__quote serif">
              “Một tài khoản — toàn bộ lịch sử nhịp tim của bạn, lưu lại để theo dõi tiến triển.”
            </p>
            <span className="auth__quote-author">— Personal vitals timeline</span>
            <div className="auth__aside-stat">
              <Sparkles size={14} />
              <span>Free · End-to-end encrypted · GDPR</span>
            </div>
          </div>
          <div className="auth__aside-glow" />
        </aside>

        <main className="auth__form-wrap">
          <header className="auth__header">
            <span className="auth__eyebrow">Create account</span>
            <h2 className="auth__title">Tạo tài khoản</h2>
            <p className="auth__sub">Tham gia để lưu trữ và theo dõi sức khỏe theo thời gian.</p>
          </header>

          {error && <div className="auth__alert auth__alert--error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth__form">
            <Field icon={<User size={17} />}  label="Tên người dùng" value={username} onChange={setUsername} required />
            <Field icon={<Type size={17} />}  label="Họ và Tên"      value={fullName} onChange={setFullName} required />

            <div className="input-row">
              <Field
                icon={<Calendar size={17} />} label="Ngày sinh"
                value={dob} onChange={setDob} type="date" required halfWidth
              />
              <label className="field field--half">
                <span className="field__icon"><Hash size={17} /></span>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="field__select"
                  required
                >
                  <option value="nam">Nam</option>
                  <option value="nu">Nữ</option>
                  <option value="khac">Khác</option>
                </select>
                <span className="field__label field__label--static">Giới tính</span>
              </label>
            </div>

            <Field icon={<Mail size={17} />} label="Email" value={email}    onChange={setEmail}    type="email"    required />
            <Field icon={<Lock size={17} />} label="Mật khẩu" value={password} onChange={setPassword} type="password" required />

            <button type="submit" className="auth__submit" disabled={isLoading}>
              <span>{isLoading ? 'Đang đăng ký…' : 'Đăng ký'}</span>
              <ArrowRight size={16} />
            </button>
          </form>

          <footer className="auth__footer">
            Đã có tài khoản?{' '}
            <Link to="/login" className="auth__link">Đăng nhập</Link>
          </footer>
        </main>
      </div>
    </div>
  )
}

function Field({ icon, label, value, onChange, type = 'text', required, halfWidth }) {
  return (
    <label className={`field${halfWidth ? ' field--half' : ''}`}>
      <span className="field__icon">{icon}</span>
      <input
        className="field__input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        placeholder=" "
      />
      <span className="field__label">{label}</span>
    </label>
  )
}
