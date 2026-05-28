import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Lock, User, Activity, Sparkles, ArrowRight } from 'lucide-react'
import { supabase } from '@/lib/supabase'
import './Login.css'

export function Login() {
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    if (location.state?.message) setMessage(location.state.message)
  }, [location])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setMessage(''); setIsLoading(true)

    try {
      let authError = null

      if (identifier.includes('@')) {
        const { error } = await supabase.auth.signInWithPassword({ email: identifier, password })
        authError = error
      } else {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001'
        const res = await fetch(`${apiUrl}/api/auth/login-username`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: identifier, password }),
        })
        if (!res.ok) {
          const errorData = await res.json()
          throw new Error(errorData.detail || 'Tên đăng nhập hoặc mật khẩu không đúng')
        }
        const data = await res.json()
        const { error } = await supabase.auth.setSession({
          access_token: data.access_token,
          refresh_token: data.refresh_token,
        })
        authError = error
      }
      if (authError) throw authError
      navigate('/profile')
    } catch (err) {
      setError(err.message || 'Đã có lỗi xảy ra')
    } finally {
      setIsLoading(false)
    }
  }

  const handleGoogleLogin = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({ provider: 'google' })
      if (error) throw error
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="auth">
      <div className="auth__shell">
        {/* Cột bên trái — cinematic */}
        <aside className="auth__aside" aria-hidden="true">
          <div className="auth__aside-inner">
            <span className="auth__brand">
              <Activity size={16} />
              <span>NIHealth</span>
            </span>
            <p className="auth__quote serif">
              “Nhịp tim — bản nhạc thầm lặng nhất của sự sống. Chúng ta lắng nghe nó qua một khung hình.”
            </p>
            <span className="auth__quote-author">— rPPG Research Team</span>

            <div className="auth__aside-stat">
              <Sparkles size={14} />
              <span>30 fps · non-invasive · &lt;15s warm-up</span>
            </div>
          </div>
          <div className="auth__aside-glow" />
        </aside>

        {/* Cột bên phải — form */}
        <main className="auth__form-wrap">
          <header className="auth__header">
            <span className="auth__eyebrow">Welcome back</span>
            <h2 className="auth__title">Đăng nhập</h2>
            <p className="auth__sub">Tiếp tục theo dõi các chỉ số sức khỏe của bạn.</p>
          </header>

          {message && <div className="auth__alert auth__alert--success">{message}</div>}
          {error   && <div className="auth__alert auth__alert--error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth__form">
            <Field
              icon={<User size={17} />}
              label="Tên đăng nhập hoặc Email"
              value={identifier}
              onChange={setIdentifier}
              type="text"
              required
            />
            <Field
              icon={<Lock size={17} />}
              label="Mật khẩu"
              value={password}
              onChange={setPassword}
              type="password"
              required
            />

            <button type="submit" className="auth__submit" disabled={isLoading}>
              <span>{isLoading ? 'Đang đăng nhập…' : 'Đăng nhập'}</span>
              <ArrowRight size={16} />
            </button>
          </form>

          <div className="auth__divider"><span>hoặc tiếp tục với</span></div>

          <button type="button" className="auth__google" onClick={handleGoogleLogin}>
            <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="" />
            <span>Google</span>
          </button>

          <footer className="auth__footer">
            <Link to="/forgot-password" className="auth__link auth__link--mute">
              Quên mật khẩu?
            </Link>
            <span className="auth__footer-sep">·</span>
            <span>
              Chưa có tài khoản?{' '}
              <Link to="/register" className="auth__link">Đăng ký ngay</Link>
            </span>
          </footer>
        </main>
      </div>
    </div>
  )
}

function Field({ icon, label, value, onChange, type = 'text', required, autoComplete, minLength }) {
  return (
    <label className="field">
      <span className="field__icon">{icon}</span>
      <input
        className="field__input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        autoComplete={autoComplete}
        minLength={minLength}
        placeholder=" "
      />
      <span className="field__label">{label}</span>
    </label>
  )
}
