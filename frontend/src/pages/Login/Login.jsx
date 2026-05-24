import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { LogIn, Mail, Lock } from 'lucide-react'
import { login, getMe } from '@/lib/api'
import { useAuthStore } from '@/store/authStore'
import './Login.css'

export function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()
  const setAuth = useAuthStore((state) => state.setAuth)

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
      const data = await login(email, password)
      // Save to store temporarily to let axios pick it up for getMe
      useAuthStore.getState().setAuth(null, data.access_token)
      
      // Fetch user profile
      const user = await getMe()
      setAuth(user, data.access_token)
      
      navigate('/profile')
    } catch (err) {
      setError(err.response?.data?.detail || 'Email hoặc mật khẩu không đúng')
    } finally {
      setIsLoading(false)
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
            <Mail size={20} className="input-icon" />
            <input 
              type="email" 
              placeholder="Email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
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
        
        <div className="auth-footer">
          Chưa có tài khoản? <Link to="/register">Đăng ký ngay</Link>
        </div>
      </div>
    </div>
  )
}
