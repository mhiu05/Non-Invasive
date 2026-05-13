import { useCallback, useEffect, useRef } from 'react'
import { useVitalsStore } from '@/store/vitalsStore'
import type { WsMessage } from '@/types/vitals'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8001/ws/stream'
const RECONNECT_MS = 3000

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const shouldReconnect = useRef(false)
  const { setVitals, setFace, setConnected } = useVitalsStore()

  const connect = useCallback(() => {
    shouldReconnect.current = true
    ws.current = new WebSocket(WS_URL)

    ws.current.onopen = () => {
      setConnected(true)
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }

    ws.current.onmessage = (e: MessageEvent) => {
      const msg = JSON.parse(e.data as string) as WsMessage
      if (msg.type === 'vitals') {
        setVitals(msg.heart_rate, msg.blink_rate, msg.snr_db, msg.bvp_window)
      } else if (msg.type === 'face') {
        setFace(msg.detected, msg.bbox)
      }
    }

    ws.current.onclose = () => {
      setConnected(false)
      if (shouldReconnect.current) {
        reconnectTimer.current = setTimeout(connect, RECONNECT_MS)
      }
    }

    ws.current.onerror = () => ws.current?.close()
  }, [setVitals, setFace, setConnected])

  const disconnect = useCallback(() => {
    shouldReconnect.current = false
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    ws.current?.close()
    ws.current = null
    setConnected(false)
  }, [setConnected])

  const sendFrame = useCallback((base64jpeg: string) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'frame', data: base64jpeg }))
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => () => { disconnect() }, [disconnect])

  return { connect, disconnect, sendFrame }
}
