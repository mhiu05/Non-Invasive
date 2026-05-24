import axios from 'axios'
import { useAuthStore } from '../store/authStore'

// Cấu hình HTTP client với axios
export const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8001',
  timeout: 120_000, // Timeout 2 phút cho các request xử lý lâu
})

http.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
    }
    return Promise.reject(error)
  }
)

export async function login(email, password) {
  const { data } = await http.post('/auth/login', { email, password })
  return data
}

export async function register(username, email, password) {
  const { data } = await http.post('/auth/register', { username, email, password })
  return data
}

export async function getMe() {
  const { data } = await http.get('/auth/me')
  return data
}


// Upload video và xử lý đồng bộ (đợi kết quả ngay)
export async function uploadVideo(file, age, onProgress) {
  const form = new FormData()
  form.append('file', file)
  if (age != null) {
    form.append('age', age.toString())
  }

  const { data } = await http.post('/video/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

// Upload video bất đồng bộ (nhận về job_id để polling)
export async function uploadVideoAsync(file, age, onProgress) {
  const form = new FormData()
  form.append('file', file)
  if (age != null) {
    form.append('age', age.toString())
  }

  const { data } = await http.post('/video/upload-async', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

// Lấy trạng thái của job phân tích video (polling)
export async function getJobStatus(jobId) {
  const { data } = await http.get(`/video/jobs/${jobId}`)
  return data
}

// Lấy danh sách lịch sử phiên đo (có hỗ trợ filter)
export async function fetchHistory(filters) {
  const params = {}
  if (filters?.type) params.type = filters.type
  if (filters?.start_at) params.start_at = filters.start_at
  if (filters?.end_at) params.end_at = filters.end_at
  if (filters?.limit) params.limit = filters.limit
  if (filters?.offset) params.offset = filters.offset

  const { data } = await http.get('/history', { params })
  return data
}

// Lấy chi tiết một phiên đo lịch sử cụ thể
export async function fetchHistoryDetail(id) {
  const { data } = await http.get(`/history/${id}`)
  return data
}

// Kiểm tra trạng thái hoạt động của backend (health check)
export async function checkHealth() {
  const { data } = await http.get('/health')
  return data
}
