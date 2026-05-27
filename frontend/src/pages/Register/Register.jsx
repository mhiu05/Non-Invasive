import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { LogIn, User, Lock, Mail, Calendar, Hash, Type } from 'lucide-react'
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
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            username,
            full_name: fullName,
            gender,
            dob
          }
        }
      })

      if (error) {
        setError(error.message)
        return
      }

      navigate('/login', { state: { message: 'Đăng ký thành công! Vui lòng kiểm tra hộp thư Email để xác nhận tài khoản trước khi đăng nhập.' } })
    } catch (err) {
      setError('Đã có lỗi xảy ra khi đăng ký')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Tạo tài khoản</h2>
          <p>Tham gia để lưu trữ và theo dõi sức khỏe của bạn</p>
        </div>
        
        {error && <div className="auth-error">{error}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <User size={20} className="input-icon" />
            <input 
              type="text" 
              placeholder="Tên người dùng" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <Type size={20} className="input-icon" />
            <input 
              type="text" 
              placeholder="Họ và Tên" 
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          </div>

          <div className="input-row">
            <div className="input-group half">
              <Calendar size={20} className="input-icon" />
              <input 
                type="date" 
                value={dob}
                onChange={(e) => setDob(e.target.value)}
                required
              />
            </div>
            
            <div className="input-group half">
              <Hash size={20} className="input-icon" />
              <select value={gender} onChange={(e) => setGender(e.target.value)} required className="gender-select">
                <option value="nam">Nam</option>
                <option value="nu">Nữ</option>
                <option value="khac">Khác</option>
              </select>
            </div>
          </div>
          
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
            {isLoading ? 'Đang đăng ký...' : 'Đăng ký'}
          </button>
        </form>
        
        <div className="auth-footer">
          Đã có tài khoản? <Link to="/login">Đăng nhập</Link>
        </div>
      </div>
    </div>
  )
}
