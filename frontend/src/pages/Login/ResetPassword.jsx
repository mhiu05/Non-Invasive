import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Lock, ArrowRight, Activity, Sparkles } from 'lucide-react'
import { supabase } from '@/lib/supabase'
import './Login.css'

export function ResetPassword() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    supabase.auth.onAuthStateChange(async (event) => {
      if (event === 'PASSWORD_RECOVERY') {
        // Session tạm thời đã có; hiển thị form đổi mật khẩu
      }
    })
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setMessage('')
    if (password !== confirmPassword) { setError('Mật khẩu không khớp'); return }
    setIsLoading(true)
    try {
      const { error } = await supabase.auth.updateUser({ password })
      if (error) { setError(error.message); return }
      setMessage('Đổi mật khẩu thành công! Đang chuyển hướng…')
      setTimeout(() => navigate('/login'), 2000)
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
              “Một mật khẩu tốt là một mật khẩu chỉ bạn nhớ — và máy chủ không bao giờ thấy nó dưới dạng rõ.”
            </p>
            <span className="auth__quote-author">— Security note</span>
            <div className="auth__aside-stat">
              <Sparkles size={14} />
              <span>Tối thiểu 6 ký tự · Khuyến nghị 12+</span>
            </div>
          </div>
          <div className="auth__aside-glow" />
        </aside>

        <main className="auth__form-wrap">
          <header className="auth__header">
            <span className="auth__eyebrow">Set new password</span>
            <h2 className="auth__title">Mật khẩu mới</h2>
            <p className="auth__sub">Nhập mật khẩu mới cho tài khoản của bạn.</p>
          </header>

          {message && <div className="auth__alert auth__alert--success">{message}</div>}
          {error   && <div className="auth__alert auth__alert--error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth__form">
            <label className="field">
              <span className="field__icon"><Lock size={17} /></span>
              <input
                className="field__input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                placeholder=" "
              />
              <span className="field__label">Mật khẩu mới</span>
            </label>

            <label className="field">
              <span className="field__icon"><Lock size={17} /></span>
              <input
                className="field__input"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength={6}
                placeholder=" "
              />
              <span className="field__label">Xác nhận mật khẩu</span>
            </label>

            <button type="submit" className="auth__submit" disabled={isLoading}>
              <span>{isLoading ? 'Đang lưu…' : 'Lưu mật khẩu mới'}</span>
              <ArrowRight size={16} />
            </button>
          </form>
        </main>
      </div>
    </div>
  )
}
