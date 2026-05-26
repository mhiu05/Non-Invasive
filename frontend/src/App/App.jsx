import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header/Header'
import { Home } from '@/pages/Home/Home'
import { Live } from '@/pages/Live/Live'
import { Upload } from '@/pages/Upload/Upload'
import { Login } from '@/pages/Login/Login'
import { Register } from '@/pages/Register/Register'
import { Profile } from '@/pages/Profile/Profile'
import { ProtectedRoute } from '@/components/ProtectedRoute/ProtectedRoute'
import { ChatBot } from '@/components/ChatBot/ChatBot'
import { AuthProvider } from '@/features/auth/AuthProvider'
import './App.css'

// Component gốc của ứng dụng, thiết lập cấu trúc Layout (Header, ChatBot) và định tuyến (Routes)
export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        {/* Container chính bao bọc toàn bộ giao diện */}
        <div className="app-container">
          <Header />
        
        {/* Khai báo các route (trang) trong hệ thống */}
        <Routes>
          <Route path="/"       element={<Home />} />
          <Route path="/live"   element={<Live />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/login"  element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        </Routes>
        
        {/* Widget ChatBot luôn hiển thị ở góc màn hình trên mọi trang */}
        <ChatBot />
      </div>
      </AuthProvider>
    </BrowserRouter>
  )
}
