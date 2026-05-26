import { Navigate } from 'react-router-dom'
import { useAuth } from '@/features/auth/AuthProvider'

export function ProtectedRoute({ children }) {
  const { session, loading } = useAuth()

  if (loading) return <div>Loading...</div>

  if (!session) {
    return <Navigate to="/login" replace />
  }

  return children
}
