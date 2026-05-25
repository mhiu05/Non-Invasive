import { create } from 'zustand'

// Store quản lý trạng thái toàn cục (global state) cho các chỉ số sức khỏe realtime
export const useVitalsStore = create((set) => ({
  heartRate: null,
  snrDb: null,
  hrvMs: null,
  bvpWindow: [],      // Cửa sổ tín hiệu BVP để vẽ biểu đồ
  bufferFrames: 0,    // Số frame hiện tại đã thu thập
  bufferNeeded: 181,  // Số frame cần thiết để model phân tích
  faceBbox: null,     // Tọa độ bounding box khuôn mặt
  faceDetected: false,// Trạng thái có phát hiện khuôn mặt không
  isConnected: false, // Trạng thái kết nối WebSocket

  // Cập nhật các chỉ số sinh tồn chính
  setVitals: (heartRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded) =>
    set({ heartRate, snrDb, hrvMs, bvpWindow, bufferFrames, bufferNeeded }),
    
  // Cập nhật trạng thái nhận diện khuôn mặt
  setFace: (faceDetected, faceBbox) => set({ faceDetected, faceBbox }),
  
  // Cập nhật trạng thái kết nối
  setConnected: (isConnected) => set({ isConnected }),
  
  // Reset toàn bộ chỉ số về giá trị ban đầu (thường dùng khi bắt đầu phiên đo mới)
  reset: () =>
    set({
      heartRate: null,
      snrDb: null,
      hrvMs: null,
      bvpWindow: [],
      bufferFrames: 0,
      bufferNeeded: 181,
      faceBbox: null,
      faceDetected: false,
    }),
}))
