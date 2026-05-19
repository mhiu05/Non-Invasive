import { create } from 'zustand'

interface VitalsState {
  heartRate: number | null
  blinkRate: number | null
  snrDb: number | null
  hrvMs: number | null
  bvpWindow: number[]
  bufferFrames: number
  bufferNeeded: number
  faceBbox: [number, number, number, number] | null
  faceDetected: boolean
  isConnected: boolean

  setVitals: (
    hr: number | null,
    br: number | null,
    snr: number | null,
    hrv: number | null,
    bvp: number[],
    bf: number,
    bn: number,
  ) => void
  setFace: (detected: boolean, bbox: [number, number, number, number] | null) => void
  setConnected: (v: boolean) => void
  reset: () => void
}

export const useVitalsStore = create<VitalsState>((set) => ({
  heartRate: null,
  blinkRate: null,
  snrDb: null,
  hrvMs: null,
  bvpWindow: [],
  bufferFrames: 0,
  bufferNeeded: 181,
  faceBbox: null,
  faceDetected: false,
  isConnected: false,

  setVitals: (heartRate, blinkRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded) =>
    set({ heartRate, blinkRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded }),
  setFace: (faceDetected, faceBbox) => set({ faceDetected, faceBbox }),
  setConnected: (isConnected) => set({ isConnected }),
  reset: () =>
    set({
      heartRate: null,
      blinkRate: null,
      snrDb: null,
      hrvMs: null,
      bvpWindow: [],
      bufferFrames: 0,
      bufferNeeded: 181,
      faceBbox: null,
      faceDetected: false,
    }),
}))
