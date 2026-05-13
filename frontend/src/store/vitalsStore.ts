import { create } from 'zustand'

interface VitalsState {
  heartRate: number | null
  blinkRate: number | null
  snrDb: number | null
  bvpWindow: number[]
  faceBbox: [number, number, number, number] | null
  faceDetected: boolean
  isConnected: boolean

  setVitals: (hr: number, br: number, snr: number, bvp: number[]) => void
  setFace: (detected: boolean, bbox: [number, number, number, number] | null) => void
  setConnected: (v: boolean) => void
  reset: () => void
}

export const useVitalsStore = create<VitalsState>((set) => ({
  heartRate: null,
  blinkRate: null,
  snrDb: null,
  bvpWindow: [],
  faceBbox: null,
  faceDetected: false,
  isConnected: false,

  setVitals: (heartRate, blinkRate, snrDb, bvpWindow) =>
    set({ heartRate, blinkRate, snrDb, bvpWindow }),
  setFace: (faceDetected, faceBbox) => set({ faceDetected, faceBbox }),
  setConnected: (isConnected) => set({ isConnected }),
  reset: () =>
    set({
      heartRate: null,
      blinkRate: null,
      snrDb: null,
      bvpWindow: [],
      faceBbox: null,
      faceDetected: false,
    }),
}))
