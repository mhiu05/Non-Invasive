import { Activity, Moon, Sun, Upload, Video } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'

export function Header() {
  const { pathname } = useLocation()
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  const navLink = (to: string, icon: React.ReactNode, label: string) => (
    <Link
      to={to}
      className={cn(
        'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
        pathname === to
          ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300'
          : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800',
      )}
    >
      {icon}
      {label}
    </Link>
  )

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-700 dark:bg-slate-900/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2 font-semibold text-indigo-600 dark:text-indigo-400">
          <Activity size={20} />
          <span className="hidden sm:inline">Non-Invasive Health</span>
          <span className="sm:hidden">NIHealth</span>
        </div>

        <nav className="flex items-center gap-1">
          {navLink('/', <Video size={15} />, 'Live')}
          {navLink('/upload', <Upload size={15} />, 'Upload')}
        </nav>

        <button
          onClick={() => setDark((d) => !d)}
          className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
          aria-label="toggle dark mode"
        >
          {dark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  )
}
