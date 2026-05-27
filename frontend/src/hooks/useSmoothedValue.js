import { useState, useEffect } from 'react'

export function useSmoothedValue(rawValue, smoothingFactor = 0.07, threshold = 0.25, intervalMs = 140) {
  const [displayValue, setDisplayValue] = useState(null)

  useEffect(() => {
    if (rawValue === null) {
      setDisplayValue(null)
      return
    }

    let active = true
    const step = () => {
      setDisplayValue((prev) => {
        if (prev === null) return rawValue
        const next = prev + (rawValue - prev) * smoothingFactor
        if (Math.abs(next - rawValue) < threshold) return rawValue
        return next
      })
      
      if (active) window.setTimeout(step, intervalMs)
    }
    
    step()
    return () => { active = false }
  }, [rawValue, smoothingFactor, threshold, intervalMs])

  return displayValue
}
