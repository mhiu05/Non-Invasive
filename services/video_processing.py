import cv2
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import spdiags
from collections import deque
import threading
import time
import queue
import logging
import torch

from collections import deque, OrderedDict
from rPPG_Toolbox.neural_methods.model.DeepPhys import DeepPhys
from rPPG_Toolbox.dataset.data_loader.face_detector.YOLO5Face import YOLO5Face

# === Cấu hình Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Hằng số (Constants) ===
### TODO: 1. ĐIỀN ĐƯỜNG DẪN VÀ THIẾT LẬP MODEL ###
MODEL_PATH = r"rPPG_Toolbox/final_model_release/PURE_DeepPhys.pth" 
MODEL_INPUT_SIZE = (72, 72) # Kích thước (H, W) model được train (ví dụ: 72, 72)

# Thiết lập xử lý
FFT_BUFFER_SIZE = 300   # Buffer cho tín hiệu PPG (đầu vào FFT)
FPS = 30                # Giả định FPS (cần khớp với khi train)
LOW_PASS = 0.75         # Tần số thấp (khuyến nghị 0.75)
HIGH_PASS = 2.5         # Tần số cao (khuyến nghị 2.5)

# === Tải Model (Model Loading) ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEEPPHYS_MODEL = None
YOLO_FACE_DETECTOR = None # --- THAY ĐỔI ---: Thêm global cho YOLO
logger.info(f"Sử dụng thiết bị: {DEVICE}")

def load_deepphys_model():
    """
    Tải model DeepPhys đã huấn luyện.
    """
    global DEEPPHYS_MODEL
    try:
        logger.info(f"Đang tải model DeepPhys từ: {MODEL_PATH}")
        model = DeepPhys(img_size=MODEL_INPUT_SIZE[0])
        
        # --- BẮT ĐẦU SỬA LỖI ---
        # Tải trọng số (state_dict) vào một biến tạm
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Tạo một state_dict mới (OrderedDict) để chứa các key đã sửa
        new_state_dict = OrderedDict()
        
        # Lặp qua tất cả các key trong file model đã tải
        for k, v in state_dict.items():
            name = k
            if k.startswith('module.'):
                name = k[7:] # Xóa bỏ 7 ký tự "module." ở đầu
            new_state_dict[name] = v
            
        # Tải state_dict MỚI đã được làm sạch
        model.load_state_dict(new_state_dict)
        # --- KẾT THÚC SỬA LỖI ---
        
        model.to(DEVICE)
        model.eval() 
        DEEPPHYS_MODEL = model
        logger.info("Tải model DeepPhys thành công!")
        
    except Exception as e:
        logger.error(f"LỖI khi tải model DeepPhys: {e}")
        logger.error("Vui lòng kiểm tra lại MODEL_PATH và kiến trúc model.")


def load_yolo_detector():
    """
    Tải model YOLO5Face.
    """
    global YOLO_FACE_DETECTOR
    try:
        logger.info("Đang tải model YOLO5Face...")
        # Chuyển torch.device thành string (ví dụ: "cpu" hoặc "cuda:0")
        # vì class YOLO5Face của bạn mong đợi một string
        device_str = str(DEVICE)
        YOLO_FACE_DETECTOR = YOLO5Face(device=device_str)
        logger.info(f"Tải model YOLO5Face thành công! (Device: {device_str})")
    except Exception as e:
        logger.error(f"LỖI khi tải YOLO5Face: {e}")
        logger.error("Đảm bảo file YOLO5Face.py và các file phụ thuộc (ckpts, model...) nằm đúng vị trí.")


# === Khởi tạo Biến Toàn cục (Global Variables) ===
pulse_signal_buffer = deque(maxlen=FFT_BUFFER_SIZE)
previous_processed_face = None
rgb_clients = []
current_rgb_data = {'heart_rate': 0.0}
rgb_data_lock = threading.Lock()    


def standardized_data(data):
    """Z-score standardization (từ BaseLoader.py)"""
    if data.shape[2] == 0: return data
    for i in range(data.shape[2]):
        channel = data[..., i]
        mean = np.mean(channel)
        std = np.std(channel) + 1e-7 
        data[..., i] = (channel - mean) / std
    data[np.isnan(data)] = 0
    return data

# === HÀM TỪ 'post_process.py' (Hậu xử lý) ===
def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value=100):
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def post_process_signal(signal):
    processed_signal = np.cumsum(signal)
    processed_signal = _detrend(processed_signal, 100)
    [b, a] = butter(1, [LOW_PASS / FPS * 2, HIGH_PASS / FPS * 2], btype='bandpass')
    processed_signal = filtfilt(b, a, np.double(processed_signal))
    return processed_signal

def _calculate_fft_hr(ppg_signal, fs=FPS, low_pass=LOW_PASS, high_pass=HIGH_PASS):
    if len(ppg_signal) < 60: 
        return 0.0
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    if fmask_ppg.size == 0:
        logger.debug("Không tìm thấy tần số trong dải hợp lệ.")
        return 0.0
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

# === Hàm Tiền xử lý Frame (Frame Preprocessing) ===

def detect_and_crop_face(image):
    """
    Phát hiện mặt bằng YOLO5Face, cắt (crop) và resize 
    về kích thước model yêu cầu.
    """
    if YOLO_FACE_DETECTOR is None:
        logger.warning("YOLO_FACE_DETECTOR chưa được tải.")
        return None, None

    # 1. Phát hiện mặt
    # File YOLO5Face.py trả về [x1, y1, x2, y2] hoặc None
    face_coords = YOLO_FACE_DETECTOR.detect_face(image)

    if face_coords is None:
        logger.debug("Không phát hiện thấy mặt.")
        return None, None 

    # 2. Giải nén tọa độ
    x_min, y_min, x_max, y_max = [int(coord) for coord in face_coords]

    # 3. Đảm bảo box hợp lệ
    h, w, _ = image.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)
    
    if x_min >= x_max or y_min >= y_max:
        logger.debug("Bounding box không hợp lệ.")
        return None, None

    # 4. Cắt (crop) mặt
    cropped_face = image[y_min:y_max, x_min:x_max]
    
    # 5. Resize về kích thước chuẩn của model
    resized_face = cv2.resize(cropped_face, MODEL_INPUT_SIZE, 
                              interpolation=cv2.INTER_AREA)

    bbox = (x_min, y_min, x_max, y_max)
    return resized_face, bbox

# === Hàm Điều khiển (Control Functions) ===
def reset_processing_state():
    """Xóa buffers và reset trạng thái (quan trọng khi mất mặt)."""
    global previous_processed_face
    logger.debug("Đang reset trạng thái xử lý (mất mặt hoặc bắt đầu).")
    pulse_signal_buffer.clear()
    previous_processed_face = None

def set_kalman_defaults(): # Được gọi từ video_routes.py
    reset_processing_state()

def broadcast_rgb_data(data):
    with rgb_data_lock:
        global current_rgb_data
        current_rgb_data = data
        disconnected_clients = []
        for client_queue in rgb_clients:
            try:
                client_queue.put(data, timeout=0.1)
            except queue.Full:
                disconnected_clients.append(client_queue)
        for client in disconnected_clients:
            rgb_clients.remove(client)

# === HÀM CHÍNH (Main Generator Function) ===
def generate_gray_frames(path):
    """
    Generator chính (đã được cập nhật để tải và sử dụng YOLO)
    """
    global previous_processed_face
    
    # --- THAY ĐỔI ---: Tải cả hai model khi bắt đầu
    if DEEPPHYS_MODEL is None:
        load_deepphys_model()
    if YOLO_FACE_DETECTOR is None:
        load_yolo_detector()

    # Kiểm tra lại sau khi tải
    if DEEPPHYS_MODEL is None or YOLO_FACE_DETECTOR is None:
        logger.error("KHÔNG THỂ CHẠY: Một trong các model chưa được tải.")
        return

    cap = cv2.VideoCapture(path)
    reset_processing_state() # Xóa buffer cũ
    
    heart_rate = 0.0
    hr_display = "..."

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.info("Kết thúc video hoặc lỗi đọc frame.")
                break

            # 2. Tiền xử lý (Phát hiện, Cắt) - Hàm này đã được thay bằng YOLO
            resized_face, bbox = detect_and_crop_face(frame)
            
            if resized_face is None:
                # --- Không phát hiện thấy mặt ---
                reset_processing_state() 
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                hr_display = "..."
                heart_rate = 0.0
            
            else:
                # --- Đã phát hiện thấy mặt ---
                # 2b. Chuẩn hóa (Standardize)
                current_processed_face = standardized_data(resized_face.astype(np.float32))

                # 3. Tạo input 6 kênh (chỉ khi có frame trước đó)
                if previous_processed_face is not None:
                    diff_frame = current_processed_face - previous_processed_face
                    raw_frame = current_processed_face
                    
                    diff_tensor = torch.from_numpy(diff_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    raw_tensor = torch.from_numpy(raw_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    model_input = torch.cat((diff_tensor, raw_tensor), dim=1).float()

                    # 4. Dự đoán bằng DeepPhys
                    with torch.no_grad():
                        pulse_value = DEEPPHYS_MODEL(model_input).item()
                    
                    # 5. Thêm vào Signal Buffer
                    pulse_signal_buffer.append(pulse_value)

                    # 6. Kiểm tra buffer tín hiệu đã đủ để FFT chưa
                    if len(pulse_signal_buffer) < FFT_BUFFER_SIZE:
                        hr_display = "Detecting..."
                        heart_rate = 0.0
                    else:
                        # 7. Hậu xử lý & Tính toán HR
                        signal_raw = np.array(pulse_signal_buffer)
                        signal_clean = post_process_signal(signal_raw)
                        heart_rate = _calculate_fft_hr(signal_clean)
                        
                        if heart_rate == 0.0:
                            hr_display = "Detecting..."
                        else:
                            hr_display = f"{heart_rate:.1f} BPM"
                
                else: # Frame đầu tiên, chưa có diff
                    hr_display = "Initializing..."
                    heart_rate = 0.0

                previous_processed_face = current_processed_face.copy()
                
                if bbox: # Vẽ bounding box của YOLO
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                  (0, 255, 0), 2)

            # --- Gửi Dữ liệu và Stream Frame ---
            cv2.putText(frame, f'Heart Rate: {hr_display}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            rgb_data = {
                'heart_rate': heart_rate if isinstance(heart_rate, float) else 0.0,
                'timestamp': time.time()
            }
            broadcast_rgb_data(rgb_data) 
            
            ret, buffer_img = cv2.imencode('.jpg', frame)
            frame_bytes = buffer_img.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        logger.info("Dọn dẹp tài nguyên...")
        cap.release()
        reset_processing_state()