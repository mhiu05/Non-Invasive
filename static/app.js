/**
 * @file B-COM.js
 * Client-side script (browser) to handle video uploading,
 * display the video stream, and receive heart rate data
 * via Server-Sent Events (SSE).
 */

// --- Global Variables ---

/**
 * Holds the EventSource object for the Server-Sent Events (SSE) connection.
 * @type {EventSource | null}
 */
let eventSource;

/**
 * A boolean flag to track the status of the SSE stream.
 * true if the stream is active, false otherwise.
 * @type {boolean}
 */
let isStreamActive = false;

// --- DOM Element References ---
const uploadForm = document.getElementById("upload-form");
const uploadButton = document.getElementById("upload-button");
const videoInput = document.getElementById("video-input");
const fileLabel = document.getElementById("file-label");
const fileInputContainer = document.getElementById("file-input-container");
const errorContainer = document.getElementById("error-container");
const videoStream = document.getElementById("video-stream"); // <img> tag for displaying the video stream
const heartRateEl = document.getElementById("heart-rate").querySelector("strong"); // <strong> element for displaying HR

/**
 * Displays an error message in the UI.
 * @param {string} message - The error content to display.
 */
function showError(message) {
    errorContainer.textContent = message;
    errorContainer.hidden = false; // Make it visible
}

/**
 * Hides and clears the content of the error message container.
 */
function hideError() {
    errorContainer.textContent = "";
    errorContainer.hidden = true; // Hide it
}

/**
 * Updates the state of the upload button (e.g., loading).
 * @param {boolean} isLoading - true if loading, false if complete.
 */
function setButtonLoading(isLoading) {
    if (isLoading) {
        uploadButton.disabled = true; // Disable the button
        uploadButton.textContent = 'Processing...'; // Change the text
    } else {
        uploadButton.disabled = false; // Re-enable the button
        uploadButton.textContent = 'Upload and Stream'; // Restore original text
    }
}

/**
 * Updates the file input label to show the name of the selected file.
 */
function updateFileInputLabel() {
    if (videoInput.files && videoInput.files[0]) {
        // If a file is selected, show its name
        fileLabel.textContent = videoInput.files[0].name;
        fileInputContainer.classList.add('has-file'); // Add class for styling (CSS)
    } else {
        // Otherwise, show the default text
        fileLabel.textContent = 'Choose Video File';
        fileInputContainer.classList.remove('has-file');
    }
}

/**
 * Resets the user interface to its initial state.
 * (e.g., when uploading a new video).
 */
function resetUI() {
    heartRateEl.textContent = 'Detecting...'; // Reset heart rate text
    heartRateEl.parentElement.classList.remove('active'); // Remove 'active' class (used for styling)
    videoStream.src = ""; // Clear the old video source
}

/**
 * Handles the heart rate data received from the SSE stream.
 * @param {object} data - The data object parsed from JSON.
 */
function handleDataUpdate(data) {
    // Ignore if this is a "keep-alive" message
    if (data.heartbeat) return;

    if (data.heart_rate <= 0) {
        // If the heart rate is invalid (e.g., 0), display "Detecting..."
        heartRateEl.textContent = "Detecting...";
        heartRateEl.parentElement.classList.remove('active');
    } else {
        // If a valid heart rate is received, display it and add the 'active' class
        heartRateEl.textContent = `${data.heart_rate.toFixed(1)} BPM`;
        heartRateEl.parentElement.classList.add('active');
    }
}

/**
 * Initializes and starts the Server-Sent Events (SSE) connection.
 */
function startSSEConnection() {
    // Close any existing connection
    if (eventSource) {
        eventSource.close();
    }

    // Create a new connection to the '/rgb_stream' endpoint
    eventSource = new EventSource('/rgb_stream');

    eventSource.onopen = () => console.log('SSE connection opened');

    // Handle incoming messages
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data); // Parse the JSON data
            handleDataUpdate(data); // Update the UI
        } catch (error) {
            console.error('Error parsing SSE data:', error);
        }
    };

    // Handle connection errors
    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        // Automatically try to reconnect after 3 seconds if the stream is still "active"
        setTimeout(() => {
            if (isStreamActive) {
                console.log('Attempting to reconnect...');
                startSSEConnection();
            }
        }, 3000);
    };

    // Set the status flag to active
    isStreamActive = true;
}

/**
 * Stops and closes the SSE connection.
 */
function stopSSEConnection() {
    isStreamActive = false; // Set the flag to inactive
    if (eventSource) {
        eventSource.close(); // Close the connection
        eventSource = null;
        console.log('SSE connection closed');
    }
}

/**
 * Handles the form submission event (video upload).
 * This is the main function that controls the workflow.
 * @param {Event} e - The event object.
 */
async function handleFormSubmit(e) {
    e.preventDefault(); // Prevent the form from submitting traditionally
    hideError(); // Hide any old errors
    setButtonLoading(true); // Show the "Processing..." state
    stopSSEConnection(); // Stop any old SSE stream
    resetUI(); // Reset the UI

    const formData = new FormData(uploadForm); // Get the form data (the video file)

    try {
        // Step 1: Send the video to the server via the '/upload' endpoint
        const uploadResponse = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error(`Video upload failed: ${uploadResponse.statusText}`);
        }

        // Step 2: Send a POST request to '/reset_state'
        // This tells the backend to reset its processing state
        // (e.g., clear buffers, set 'previous_frame' = None)
        const resetResponse = await fetch("/reset_state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ variable: "reset" }) // Send a simple JSON body
        });

        if (!resetResponse.ok) {
            throw new Error(`Failed to reset variables: ${resetResponse.statusText}`);
        }

        // Step 3: Wait 0.5 seconds (500ms) to give the backend time
        // to reset and prepare the new video.
        setTimeout(() => {
            // Set the <img> tag's source to the '/video_feed' endpoint
            // Add a timestamp (new Date().getTime()) to "bust" the cache
            // This ensures the browser always requests a new stream.
            videoStream.src = "/video_feed?" + new Date().getTime();
            
            // Step 4: Start the SSE connection to receive heart rate data
            startSSEConnection();
        }, 500);

    } catch (err) {
        console.error(err);
        showError(`An error occurred: ${err.message}. Please try again.`);
    } finally {
        // Whether successful or not, always re-enable the button
        setButtonLoading(false);
    }
}

// --- Attach Event Listeners ---

// When the HTML document is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Attach handleFormSubmit to the form's 'submit' event
    uploadForm.addEventListener("submit", handleFormSubmit);
    // Attach updateFileInputLabel when the user selects a file
    videoInput.addEventListener('change', updateFileInputLabel);
});

// Stop the SSE connection when the user closes the tab or leaves the page
window.addEventListener('beforeunload', stopSSEConnection);