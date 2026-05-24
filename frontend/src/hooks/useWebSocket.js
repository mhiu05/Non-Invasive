import { useCallback, useEffect, useRef } from 'react'
import { useVitalsStore } from '@/store/vitalsStore'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8001/ws/stream'
const RECONNECT_MS = 3000

// Hook quản lý kết nối WebSocket hai chiều với backend để stream dữ liệu
export function useWebSocket(getAge) {
  const ws = useRef(null)
  const reconnectTimer = useRef(null)
  const shouldReconnect = useRef(false)
  const { setVitals, setFace, setConnected } = useVitalsStore()

  // Khởi tạo kết nối WebSocket
  const connect = useCallback(() => {
    shouldReconnect.current = true
    ws.current = new WebSocket(WS_URL)

    ws.current.onopen = () => {
      setConnected(true)
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }

    // Lắng nghe thông điệp trả về từ backend
    ws.current.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      // Nếu là dữ liệu nhịp tim, BVP, SNR
      if (msg.type === 'vitals') {
        setVitals(
          msg.heart_rate,
          msg.blink_rate,
          msg.snr_db,
          msg.hrv_ms ?? null,
          msg.bvp_window,
          msg.buffer_frames ?? msg.bvp_window.length,
          msg.buffer_needed ?? 181,
        )
      } 
      // Nếu là dữ liệu tọa độ khuôn mặt
      else if (msg.type === 'face') {
        setFace(msg.detected, msg.bbox)
      }
    }

    // Xử lý mất kết nối và tự động kết nối lại
    ws.current.onclose = () => {
      setConnected(false)
      if (shouldReconnect.current) {
        reconnectTimer.current = setTimeout(connect, RECONNECT_MS)
      }
    }

    ws.current.onerror = () => ws.current?.close()
  }, [setVitals, setFace, setConnected])

  // Ngắt kết nối thủ công
  const disconnect = useCallback(() => {
    shouldReconnect.current = false
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    ws.current?.close()
    ws.current = null
    setConnected(false)
  }, [setConnected])

  // Gửi khung hình base64 từ webcam lên backend
  const sendFrame = useCallback((base64jpeg) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      const age = getAge?.() ?? null
      const payload = { type: 'frame', data: base64jpeg }
      
      // Đính kèm độ tuổi (nếu có) để backend tinh chỉnh bộ lọc bandpass
      if (typeof age === 'number' && !Number.isNaN(age) && age >= 0) {
        payload.age = age
      }
      ws.current.send(JSON.stringify(payload))
    }
  }, [getAge])

  // Dọn dẹp kết nối khi component bị unmount
  useEffect(() => () => { disconnect() }, [disconnect])

  return { connect, disconnect, sendFrame }
}
