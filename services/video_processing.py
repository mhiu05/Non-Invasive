import cv2
import numpy as np
import threading
import time
import queue
import logging
import torch

from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import spdiags
from collections import deque, OrderedDict, deque

from rPPG_Toolbox.neural_methods.model.DeepPhys import DeepPhys
from rPPG_Toolbox.dataset.data_loader.face_detector.YOLO5Face import YOLO5Face

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Create a logger instance

# --- Configuration Constants ---
MODEL_PATH = r"rPPG_Toolbox/final_model_release/PURE_DeepPhys.pth"  # Path to the pre-trained model
MODEL_INPUT_SIZE = (72, 72)  # The input resolution required by the DeepPhys model

FFT_BUFFER_SIZE = 300  # Size of the sliding window for FFT (in frames). e.g., 300 frames @ 30 FPS = 10s
FPS = 30  # Assumed Frames Per Second of the video stream
LOW_PASS = 0.75  # Low frequency cutoff (in Hz). 0.75 Hz = 45 BPM
HIGH_PASS = 2.5  # High frequency cutoff (in Hz). 2.5 Hz = 150 BPM

# --- Global Variables ---
# Buffer to store the raw pulse signal values from the model
pulse_signal_buffer = deque(maxlen=FFT_BUFFER_SIZE)
# Stores the previously processed face frame to calculate frame-to-frame differences
previous_processed_face = None
# List of client queues for broadcasting data (e.g., in a web server)
rgb_clients = []
# Dictionary holding the current data to be broadcast
current_rgb_data = {'heart_rate': 0.0}
# A lock to ensure thread-safe access to 'current_rgb_data' and 'rgb_clients'
rgb_data_lock = threading.Lock()

# --- Device and Model Initialization ---
# Select the computation device: CUDA (GPU) if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEEPPHYS_MODEL = None  # Global placeholder for the DeepPhys model
YOLO_FACE_DETECTOR = None  # Global placeholder for the YOLO face detector
logger.info(f"Using device: {DEVICE}")


def load_deepphys_model():
    """Loads the DeepPhys rPPG model into the global DEEPPHYS_MODEL variable."""
    global DEEPPHYS_MODEL
    try:
        logger.info(f"Loading model DeepPhys from: {MODEL_PATH}")
        model = DeepPhys(img_size=MODEL_INPUT_SIZE[0])

        # Load the saved model weights
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        # --- Handle 'module.' prefix ---
        # This is necessary if the model was trained using nn.DataParallel (multi-GPU)
        # and is now being loaded on a single device.
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if k.startswith('module.'):
                name = k[7:]  # Remove the 'module.' prefix
            new_state_dict[name] = v
        # -----------------------------

        model.load_state_dict(new_state_dict)  # Load the corrected weights
        model.to(DEVICE)  # Move the model to the selected device (GPU/CPU)
        model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
        DEEPPHYS_MODEL = model
        logger.info("DeepPhys model successfully loaded!")

    except Exception as e:
        logger.error(f"Error loading DeepPhys model: {e}")


def load_yolo_detector():
    """Loads the YOLO5Face face detector into the global YOLO_FACE_DETECTOR variable."""
    global YOLO_FACE_DETECTOR
    try:
        logger.info("Loading model YOLO5Face...")
        device_str = str(DEVICE)  # Get the device name as a string
        YOLO_FACE_DETECTOR = YOLO5Face(device=device_str)
        logger.info(f"YOLO5 model successfully loaded! (Device: {device_str})")
    except Exception as e:
        logger.error(f"Error loading YOLO5Face model: {e}")


def standardized_data(data):
    """
    Performs Z-score normalization on each color channel of the image.
    (data - mean) / std
    """
    if data.shape[2] == 0:  # Check if image has channels
        return data
    # Iterate over each channel (R, G, B)
    for i in range(data.shape[2]):
        channel = data[..., i]
        mean = np.mean(channel)
        std = np.std(channel) + 1e-7  # Add epsilon to prevent division by zero
        data[..., i] = (channel - mean) / std
    data[np.isnan(data)] = 0  # Replace any NaN values with 0
    return data


def _next_power_of_2(x):
    """Finds the next power of 2 greater than or equal to x. Used for FFT optimization."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value=100):
    """
    Removes the trend from a signal using a Hodrick-Prescott-like filter.
    This helps remove slow-moving variations, like changes in lighting.
    """
    signal_length = input_signal.shape[0]
    # Create an identity matrix (H)
    H = np.identity(signal_length)
    # Create a second-order difference matrix (D)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    # Apply the detrending formula
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def post_process_signal(signal):
    """
    Applies the signal processing pipeline to the raw pulse signal from the model.
    """
    # 1. Cumulative Sum: The DeepPhys model predicts the *difference* in the signal
    #    (a derivative). We integrate it (cumulative sum) to get the actual BVP signal shape.
    processed_signal = np.cumsum(signal)

    # 2. Detrend: Remove low-frequency baseline drift.
    processed_signal = _detrend(processed_signal, 100)

    # 3. Bandpass Filter:
    # Create a Butterworth bandpass filter (order 1)
    [b, a] = butter(1, [LOW_PASS / FPS * 2, HIGH_PASS / FPS * 2], btype='bandpass')
    # Apply the filter (filtfilt ensures zero phase shift)
    processed_signal = filtfilt(b, a, np.double(processed_signal))
    return processed_signal


def _calculate_fft_hr(ppg_signal, fs=FPS, low_pass=LOW_PASS, high_pass=HIGH_PASS):
    """
    Calculates the heart rate (HR) from the cleaned PPG signal using FFT.
    """
    if len(ppg_signal) < 60:  # Need a minimum amount of data
        return 0.0

    ppg_signal = np.expand_dims(ppg_signal, 0)
    # Get the optimal FFT size (next power of 2)
    N = _next_power_of_2(ppg_signal.shape[1])
    # Calculate the Power Spectral Density (PSD) using Welch's periodogram
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)

    # Create a frequency mask to isolate the valid HR range
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    if fmask_ppg.size == 0:
        logger.debug("No frequency found in valid range")
        return 0.0

    # Get the frequencies and their corresponding power values within the mask
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)

    # Find the frequency with the highest power in the valid range
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0]

    # Convert the frequency (in Hz) to Beats Per Minute (BPM)
    return fft_hr * 60


def detect_and_crop_face(image):
    """Detects, crops, and resizes the face from a given frame."""
    if YOLO_FACE_DETECTOR is None:
        logger.warning("YOLO_FACE_DETECTOR is not loaded.")
        return None, None

    # 1. Run face detection
    face_coords = YOLO_FACE_DETECTOR.detect_face(image)

    if face_coords is None:
        logger.debug("Face not detected.")
        return None, None

    # 2. Get bounding box coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in face_coords]

    # 3. Ensure coordinates are within frame boundaries (clipping)
    h, w, _ = image.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # 4. Check for invalid box dimensions
    if x_min >= x_max or y_min >= y_max:
        logger.debug("Invalid bounding box.")
        return None, None

    # 5. Crop the face
    cropped_face = image[y_min:y_max, x_min:x_max]

    # 6. Resize to the model's required input size
    resized_face = cv2.resize(cropped_face, MODEL_INPUT_SIZE,
                              interpolation=cv2.INTER_AREA)

    bbox = (x_min, y_min, x_max, y_max)
    return resized_face, bbox


def reset_processing_state():
    """Resets the signal buffer. Called when a face is lost or at the start."""
    global previous_processed_face
    logger.debug("Resetting processing state (face lost or stream started)")
    pulse_signal_buffer.clear()
    previous_processed_face = None


def set_kalman_defaults():
    """Placeholder function, here just used to reset state."""
    reset_processing_state()


def broadcast_rgb_data(data):
    """Broadcasts the latest HR data to all connected clients (thread-safe)."""
    with rgb_data_lock:  # Acquire lock to safely modify shared resources
        global current_rgb_data
        current_rgb_data = data
        disconnected_clients = []
        # Send data to each client in the list
        for client_queue in rgb_clients:
            try:
                client_queue.put(data, timeout=0.1)  # Put data in the client's queue
            except queue.Full:
                # If queue is full (client is slow/disconnected), mark for removal
                disconnected_clients.append(client_queue)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            rgb_clients.remove(client)


def generate_gray_frames(path):
    """
    A generator function that processes the video, calculates HR,
    and yields frames for streaming.
    """
    global previous_processed_face

    # --- Load models if they haven't been loaded yet ---
    if DEEPPHYS_MODEL is None:
        load_deepphys_model()
    if YOLO_FACE_DETECTOR is None:
        load_yolo_detector()

    # --- Check if models are ready ---
    if DEEPPHYS_MODEL is None:
        logger.error("DEEPPHYS_MODEL is not loaded. Exiting.")
        return
    if YOLO_FACE_DETECTOR is None:
        logger.error("YOLO model is not loaded. Exiting.")
        return

    # --- Open video source ---
    cap = cv2.VideoCapture(path)
    reset_processing_state()

    heart_rate = 0.0
    hr_display = "..."

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.info("End of video or frame read error.")
                break

            # 1. Detect and Crop Face
            resized_face, bbox = detect_and_crop_face(frame)

            if resized_face is None:
                # --- No Face Detected ---
                reset_processing_state()
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                hr_display = "..."
                heart_rate = 0.0

            else:
                # --- Face Detected: Start Processing ---
                # 2. Standardize (Normalize) Face
                current_processed_face = standardized_data(resized_face.astype(np.float32))

                if previous_processed_face is not None:
                    # 3. Calculate Inputs for DeepPhys
                    # The model takes two inputs:
                    # 1. Motion Frame (diff_frame): Difference between current and previous frame
                    # 2. Appearance Frame (raw_frame): The current normalized frame
                    diff_frame = current_processed_face - previous_processed_face
                    raw_frame = current_processed_face
                    
                    # 4. Convert to PyTorch Tensors
                    # Permute dims from (H, W, C) to (C, H, W) and add batch dim (B, C, H, W)
                    diff_tensor = torch.from_numpy(diff_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    raw_tensor = torch.from_numpy(raw_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    # Concatenate along the channel dimension
                    model_input = torch.cat((diff_tensor, raw_tensor), dim=1).float()

                    # 5. Get Pulse Value from Model
                    with torch.no_grad():  # Disable gradient calculation for inference
                        pulse_value = DEEPPHYS_MODEL(model_input).item()

                    # 6. Add to Signal Buffer
                    pulse_signal_buffer.append(pulse_value)

                    # 7. Calculate HR if buffer is full
                    if len(pulse_signal_buffer) < FFT_BUFFER_SIZE:
                        hr_display = "Detecting..."
                        heart_rate = 0.0
                    else:
                        # Full buffer: process the signal
                        signal_raw = np.array(pulse_signal_buffer)
                        signal_clean = post_process_signal(signal_raw)
                        heart_rate = _calculate_fft_hr(signal_clean)

                        if heart_rate == 0.0:
                            hr_display = "Detecting..."
                        else:
                            hr_display = f"{heart_rate:.1f} BPM"
                
                else:
                    # This is the first frame with a face
                    hr_display = "Initializing..."
                    heart_rate = 0.0
                
                # 8. Update previous face for the next loop
                previous_processed_face = current_processed_face.copy()

                # 9. Draw bounding box
                if bbox:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (0, 255, 0), 2)

            # --- Display HR and Broadcast Data ---
            cv2.putText(frame, f'Heart Rate: {hr_display}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            rgb_data = {
                'heart_rate': heart_rate if isinstance(heart_rate, float) else 0.0,
                'timestamp': time.time()
            }
            broadcast_rgb_data(rgb_data)  # Send data to any connected clients

            # --- Encode and Yield Frame for Streaming ---
            ret, buffer_img = cv2.imencode('.jpg', frame)
            frame_bytes = buffer_img.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        # --- Cleanup ---
        logger.info("Cleaning up resources...")
        cap.release()
        reset_processing_state()