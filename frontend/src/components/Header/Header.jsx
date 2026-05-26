import { Activity, Home, Upload, Video, LogIn, UserPlus, User as UserIcon, LogOut } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '@/features/auth/AuthProvider'
import './Header.css'

// Component Header hiển thị thanh điều hướng phía trên của ứng dụng
export function Header() {
  // Lấy đường dẫn hiện tại để xác định tab nào đang active
  const { pathname } = useLocation()
  const navigate = useNavigate()
  
  const { session, signOut } = useAuth()
  const isAuthenticated = !!session

  const handleLogout = async () => {
    await signOut()
    navigate('/login')
  }

  // Hàm tạo liên kết điều hướng (nav link) với icon và nhãn tương ứng
  const navLink = (to, icon, label) => (
    <Link
      to={to}
      className={`nav-link ${pathname === to ? 'active' : ''}`}
    >
      {icon}
      <span className="nav-label">{label}</span>
    </Link>
  )

  return (
    <header className="header">
      <div className="header-container">
        {/* Logo và tên ứng dụng */}
        <div className="brand" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
          <Activity size={20} />
          <span className="brand-desktop">Non-Invasive Health</span>
          <span className="brand-mobile">NIHealth</span>
        </div>

        {/* Các liên kết điều hướng */}
        <nav className="nav">
          {navLink('/', <Home size={15} />, 'Trang chủ')}
          {navLink('/live', <Video size={15} />, 'Live')}
          {navLink('/upload', <Upload size={15} />, 'Upload')}
          <div className="nav-divider"></div>
          {isAuthenticated ? (
            <>
              {navLink('/profile', <UserIcon size={15} />, 'Hồ sơ')}
              <button className="nav-link logout-btn" onClick={handleLogout}>
                <LogOut size={15} />
                <span className="nav-label">Thoát</span>
              </button>
            </>
          ) : (
            <>
              {navLink('/login', <LogIn size={15} />, 'Đăng nhập')}
              {navLink('/register', <UserPlus size={15} />, 'Đăng ký')}
            </>
          )}
        </nav>
      </div>
    </header>
  )
}
