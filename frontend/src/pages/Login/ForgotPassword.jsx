import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Mail, ArrowLeft, Activity, Sparkles, ArrowRight } from 'lucide-react'
import { supabase } from '@/lib/supabase'
import './Login.css'

export function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setMessage(''); setIsLoading(true)
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      })
      if (error) { setError(error.message); return }
      setMessage('Đã gửi liên kết khôi phục. Vui lòng kiểm tra email của bạn.')
    } catch (err) {
      setError('Đã có lỗi xảy ra')
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
              “Quên là điều bình thường. Chúng tôi sẽ gửi giúp bạn một con đường quay lại — qua email.”
            </p>
            <span className="auth__quote-author">— Account recovery</span>
            <div className="auth__aside-stat">
              <Sparkles size={14} />
              <span>Liên kết tự huỷ sau 60 phút</span>
            </div>
          </div>
          <div className="auth__aside-glow" />
        </aside>

        <main className="auth__form-wrap">
          <header className="auth__header">
            <span className="auth__eyebrow">Recover access</span>
            <h2 className="auth__title">Quên mật khẩu</h2>
            <p className="auth__sub">Nhập email tài khoản để nhận liên kết đặt lại.</p>
          </header>

          {message && <div className="auth__alert auth__alert--success">{message}</div>}
          {error   && <div className="auth__alert auth__alert--error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth__form">
            <label className="field">
              <span className="field__icon"><Mail size={17} /></span>
              <input
                className="field__input"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder=" "
              />
              <span className="field__label">Email của bạn</span>
            </label>

            <button type="submit" className="auth__submit" disabled={isLoading}>
              <span>{isLoading ? 'Đang gửi…' : 'Gửi liên kết khôi phục'}</span>
              <ArrowRight size={16} />
            </button>
          </form>

          <footer className="auth__footer">
            <Link to="/login" className="auth__link">
              <ArrowLeft size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} />
              Quay lại đăng nhập
            </Link>
          </footer>
        </main>
      </div>
    </div>
  )
}
