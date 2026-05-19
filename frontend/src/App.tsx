import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header'
import { Home } from '@/pages/Home'
import { Live } from '@/pages/Live'
import { Upload } from '@/pages/Upload'
import { ChatBot } from '@/components/ChatBot'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
        <Header />
        <Routes>
          <Route path="/"       element={<Home />} />
          <Route path="/live"   element={<Live />} />
          <Route path="/upload" element={<Upload />} />
        </Routes>
        <ChatBot />
      </div>
    </BrowserRouter>
  )
}
