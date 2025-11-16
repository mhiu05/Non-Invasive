// --- Biến toàn cục và Hằng số ---
// ĐÃ XÓA: let rgbChart;
let eventSource;
let isStreamActive = false;
// ĐÃ XÓA: let updateCount;
// ĐÃ XÓA: const maxDataPoints;

// --- Lấy các phần tử DOM ---
const uploadForm = document.getElementById("upload-form");
const uploadButton = document.getElementById("upload-button");
const videoInput = document.getElementById("video-input");
const fileLabel = document.getElementById("file-label");
const fileInputContainer = document.getElementById("file-input-container");
const errorContainer = document.getElementById("error-container");
const videoStream = document.getElementById("video-stream");
const heartRateEl = document.getElementById("heart-rate").querySelector("strong");

// --- ĐÃ XÓA: Toàn bộ hàm initChart() ---

// --- Hàm Xử lý Giao diện (UI) ---

function showError(message) {
    errorContainer.textContent = message;
    errorContainer.hidden = false;
}

function hideError() {
    errorContainer.textContent = "";
    errorContainer.hidden = true;
}

function setButtonLoading(isLoading) {
    if (isLoading) {
        uploadButton.disabled = true;
        uploadButton.textContent = 'Processing...';
    } else {
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload and Stream';
    }
}

function updateFileInputLabel() {
    if (videoInput.files && videoInput.files[0]) {
        fileLabel.textContent = videoInput.files[0].name;
        fileInputContainer.classList.add('has-file');
    } else {
        fileLabel.textContent = 'Choose Video File';
        fileInputContainer.classList.remove('has-file');
    }
}

// Reset giao diện về trạng thái ban đầu
function resetUI() {
    heartRateEl.textContent = 'Detecting...';
    heartRateEl.parentElement.classList.remove('active'); 
    videoStream.src = ""; 
    
    // --- ĐÃ XÓA: Logic reset biểu đồ ---
}

// --- Hàm Xử lý Dữ liệu và Kết nối ---

/**
 * Cập nhật giao diện với dữ liệu mới từ SSE
 */
function handleDataUpdate(data) {
    if (data.heartbeat) return; // Bỏ qua tin nhắn heartbeat
    
    // Cập nhật nhịp tim
    if (data.heart_rate <= 0) {
        heartRateEl.textContent = "Detecting...";
        heartRateEl.parentElement.classList.remove('active');
    } else {
        heartRateEl.textContent = `${data.heart_rate.toFixed(1)} BPM`;
        heartRateEl.parentElement.classList.add('active'); // Thêm class 'active'
    }

    // --- ĐÃ XÓA: Logic cập nhật biểu đồ ---
}

// Bắt đầu kết nối Server-Sent Events (SSE)
function startSSEConnection() {
    if (eventSource) {
        eventSource.close();
    }
    
    // SỬA ĐỔI: Tên route này phải khớp với video_routes.py
    // File video_routes.py của bạn đang dùng tên 'rgb_stream' (dù tên cũ)
    eventSource = new EventSource('/rgb_stream'); 
    
    eventSource.onopen = () => console.log('SSE connection opened');
    
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleDataUpdate(data); // Đổi tên hàm
        } catch (error) {
            console.error('Error parsing SSE data:', error);
        }
    };
    
    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        setTimeout(() => {
            if (isStreamActive) {
                console.log('Attempting to reconnect...');
                startSSEConnection();
            }
        }, 3000);
    };
    isStreamActive = true;
}

function stopSSEConnection() {
    isStreamActive = false;
    if (eventSource) {
        eventSource.close();
        eventSource = null;
        console.log('SSE connection closed');
    }
}

async function handleFormSubmit(e) {
    e.preventDefault(); 
    hideError(); 
    setButtonLoading(true);
    stopSSEConnection(); 
    resetUI();

    const formData = new FormData(uploadForm);

    try {
        const uploadResponse = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error(`Video upload failed: ${uploadResponse.statusText}`);
        }

        // -----------------------------------------------------------------
        // *** ĐÂY LÀ SỬA LỖI CHÍNH (FIX LỖI "NOT FOUND") ***
        // Sửa đường dẫn từ "/set_variable" thành "/reset_state"
        const resetResponse = await fetch("/reset_state", { 
        // -----------------------------------------------------------------
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ variable: "reset" })
        });

        if (!resetResponse.ok) {
            // Thông báo lỗi này sẽ hiển thị nếu bạn sửa sai tên, 
            // ví dụ: 'Failed to reset variables: Not Found'
            throw new Error(`Failed to reset variables: ${resetResponse.statusText}`);
        }

        setTimeout(() => {
            videoStream.src = "/video_feed?" + new Date().getTime(); 
            startSSEConnection();
        }, 500); 

    } catch (err) {
        console.error(err);
        // Đây chính là thông báo lỗi đỏ bạn thấy
        showError(`An error occurred: ${err.message}. Please try again.`);
    } finally {
        setButtonLoading(false); 
    }
}

// --- Gắn các Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // ĐÃ XÓA: initChart();
    uploadForm.addEventListener("submit", handleFormSubmit);
    videoInput.addEventListener('change', updateFileInputLabel);
});

window.addEventListener('beforeunload', stopSSEConnection);
