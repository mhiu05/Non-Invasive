from flask import Blueprint, request, Response, jsonify # type: ignore
import os
import tempfile
import threading
import queue
import json
from services.video_processing import (
    # --- THAY ĐỔI ---
    # Import lại các tên hàm và biến chính xác
    # có tồn tại bên trong file video_processing.py
    set_kalman_defaults,     # Dùng hàm này thay vì reset_processing_state
    broadcast_rgb_data,      # Đổi tên từ 'hr' sang 'rgb'
    generate_gray_frames,
    rgb_clients,             # Đổi tên từ 'hr' sang 'rgb'
    rgb_data_lock,           # Đổi tên từ 'hr' sang 'rgb'
    current_rgb_data         # Đổi tên từ 'hr' sang 'rgb'
    # ------------------
)

video_bp = Blueprint('video', __name__)
video_path = ""

# --- THAY ĐỔI ---: Đổi tên route và hàm cho khớp
@video_bp.route('/reset_state', methods=['POST'])
def reset_state():
    """
    Endpoint này dùng để reset lại trạng thái xử lý.
    """
    data = request.get_json()
    variable = data.get('variable', None)
    print(f"Variable received: {variable}")
    
    # Gọi đúng tên hàm đã import
    set_kalman_defaults()  
    
    return jsonify({'status': 'Processing state reset successfully'})

@video_bp.route('/upload', methods=['POST'])
def upload():
    """
    Endpoint này không đổi.
    """
    global video_path
    video_file = request.files['video']
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(video_path)
    return "success"

@video_bp.route('/video_feed')
def video_feed():
    """
    Endpoint này không đổi.
    """
    global video_path
    if video_path:
        return Response(generate_gray_frames(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video", 404

# --- THAY ĐỔI ---: Đổi tên route và hàm cho khớp
@video_bp.route('/rgb_stream')
def rgb_stream():
    """
    Stream dữ liệu (hiện tại là HR) qua Server-Sent Events (SSE).
    """
    def event_stream():
        client_queue = queue.Queue(maxsize=10)
        rgb_clients.append(client_queue) # Dùng đúng tên biến đã import
        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield "data: {\"heartbeat\": true}\n\n"
        finally:
            if client_queue in rgb_clients:
                rgb_clients.remove(client_queue) # Dùng đúng tên biến đã import
    return Response(event_stream(), mimetype='text/event-stream')

# --- THAY ĐỔI ---: Đổi tên route và hàm cho khớp
@video_bp.route('/get_rgb_data')
def get_rgb_data():
    """
    Lấy dữ liệu hiện tại.
    """
    with rgb_data_lock: # Dùng đúng tên biến đã import
        return jsonify(current_rgb_data) # Dùng đúng tên biến đã import