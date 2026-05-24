import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App/App'
import './index.css'

// Điểm khởi đầu của ứng dụng React, mount component App vào thẻ có id 'root'
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
