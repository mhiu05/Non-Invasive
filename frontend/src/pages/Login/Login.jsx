import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { LogIn, Mail, Lock, User } from 'lucide-react'
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
    if (location.state?.message) {
      setMessage(location.state.message)
    }
  }, [location])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setMessage('')
    setIsLoading(true)
    
    try {
      let authError = null

      if (identifier.includes('@')) {
        // Login by Email
        const { error } = await supabase.auth.signInWithPassword({
          email: identifier,
          password,
        })
        authError = error
      } else {
        // Login by Username using Backend API
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001'
        const res = await fetch(`${apiUrl}/api/auth/login-username`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: identifier, password })
        })
        
        if (!res.ok) {
          const errorData = await res.json()
          throw new Error(errorData.detail || 'Tên đăng nhập hoặc mật khẩu không đúng')
        }
        
        const data = await res.json()
        // Manually set session in Supabase client
        const { error } = await supabase.auth.setSession({
          access_token: data.access_token,
          refresh_token: data.refresh_token,
        })
        authError = error
      }

      if (authError) {
        throw authError
      }

      navigate('/profile')
    } catch (err) {
      setError(err.message || 'Đã có lỗi xảy ra')
    } finally {
      setIsLoading(false)
    }
  }

  const handleGoogleLogin = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
      })
      if (error) throw error
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Đăng nhập</h2>
          <p>Chào mừng bạn quay trở lại</p>
        </div>
        
        {message && <div className="auth-message">{message}</div>}
        {error && <div className="auth-error">{error}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <User size={20} className="input-icon" />
            <input 
              type="text" 
              placeholder="Username hoặc Email" 
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              required
            />
          </div>
          
          <div className="input-group">
            <Lock size={20} className="input-icon" />
            <input 
              type="password" 
              placeholder="Mật khẩu" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          
          <button type="submit" className="auth-button" disabled={isLoading}>
            {isLoading ? 'Đang đăng nhập...' : 'Đăng nhập'}
          </button>
        </form>

        <div className="auth-divider">
          <span>hoặc</span>
        </div>

        <button type="button" className="google-btn" onClick={handleGoogleLogin}>
          <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" className="google-icon" />
          Tiếp tục với Google
        </button>
        
        <div className="auth-footer" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div>
            <Link to="/forgot-password" style={{ fontSize: '0.9rem', color: '#94a3b8' }}>Quên mật khẩu?</Link>
          </div>
          <div>
            Chưa có tài khoản? <Link to="/register">Đăng ký ngay</Link>
          </div>
        </div>
      </div>
    </div>
  )
}
