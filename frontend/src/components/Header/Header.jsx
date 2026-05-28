import { Activity, Home, Upload, Video, LogIn, UserPlus, User as UserIcon, LogOut } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '@/features/auth/AuthProvider'
import './Header.css'

// Thanh điều hướng (Header) hiển thị logo và các tab điều hướng chính
export function Header() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const { session, signOut } = useAuth()
  const isAuthenticated = !!session

  const handleLogout = async () => {
    await signOut()
    navigate('/login')
  }

  const navLink = (to, icon, label) => (
    <Link to={to} className={`nav-link ${pathname === to ? 'active' : ''}`}>
      <span className="nav-link__icon">{icon}</span>
      <span className="nav-link__label">{label}</span>
    </Link>
  )

  return (
    <header className="header">
      <div className="header__inner">
        {/* Logo + tên ứng dụng */}
        <button className="brand" onClick={() => navigate('/')} type="button" aria-label="Về trang chủ">
          <span className="brand__mark">
            <Activity size={18} strokeWidth={2.5} />
            <span className="brand__mark-pulse" />
          </span>
          <span className="brand__text">
            <span className="brand__primary">NIHealth</span>
            <span className="brand__tag">rPPG monitoring</span>
          </span>
        </button>

        {/* Các tab điều hướng */}
        <nav className="nav">
          {navLink('/',       <Home size={15} />,   'Trang chủ')}
          {navLink('/live',   <Video size={15} />,  'Đo Live')}
          {navLink('/upload', <Upload size={15} />, 'Upload')}
          <span className="nav__divider" aria-hidden="true" />
          {isAuthenticated ? (
            <>
              {navLink('/profile', <UserIcon size={15} />, 'Hồ sơ')}
              <button className="nav-link nav-link--logout" onClick={handleLogout} type="button">
                <span className="nav-link__icon"><LogOut size={15} /></span>
                <span className="nav-link__label">Thoát</span>
              </button>
            </>
          ) : (
            <>
              {navLink('/login',    <LogIn size={15} />,    'Đăng nhập')}
              {navLink('/register', <UserPlus size={15} />, 'Đăng ký')}
            </>
          )}
        </nav>
      </div>
    </header>
  )
}
