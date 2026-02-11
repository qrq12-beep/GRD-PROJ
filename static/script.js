const videoOutput = document.getElementById("videoOutput");
const alertCounter = document.getElementById("activeAlerts");
const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");
const fpsCounter = document.getElementById("fpsCounter");
const lastUpdate = document.getElementById("lastUpdate");
const peopleDetected = document.getElementById("peopleDetected");
const systemStatus = document.getElementById("systemStatus");
const loadingSpinner = document.getElementById("loadingSpinner");

let alertCount = 0;
let frameCount = 0;
let lastFpsTime = Date.now();

// Update timestamp
function updateTimestamp() {
    const now = new Date();
    lastUpdate.innerText = now.toLocaleTimeString();
}

// Update FPS
function updateFps() {
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        fpsCounter.innerText = `FPS: ${frameCount}`;
        frameCount = 0;
        lastFpsTime = now;
    }
}

// START WEBCAM
async function startCamera() {
    try {
        const video = document.createElement("video");
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        // Request camera with proper constraints
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        video.play();
        
        // Hide loading spinner after video starts
        video.onloadedmetadata = () => {
            loadingSpinner.classList.add("hidden");
        };

        // Process frames
        const processFrame = async () => {
            try {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                ctx.drawImage(video, 0, 0);

                canvas.toBlob(async (blob) => {
                    if (!blob) return;

                    const formData = new FormData();
                    formData.append("frame", blob);

                    try {
                        const res = await fetch("/process_frame", {
                            method: "POST",
                            body: formData
                        });

                        const data = await res.json();

                        if (data.success) {
                            videoOutput.src = "data:image/jpeg;base64," + data.frame;
                            updateTimestamp();
                            frameCount++;
                            updateFps();

                            // Update people count
                            if (data.detections && data.detections.length > 0) {
                                const personCount = data.detections.filter(d => d.class === 'person').length;
                                peopleDetected.innerText = personCount;
                            }

                            // Handle fight detection
                            if (data.fight) {
                                statusText.innerText = "🔴 FIGHT DETECTED";
                                statusText.classList.add("alert");
                                statusDot.classList.add("alert");

                                alertCount++;
                                alertCounter.innerText = alertCount;

                                // Auto-reset after 5 seconds
                                setTimeout(() => {
                                    statusText.innerText = "✅ Normal";
                                    statusText.classList.remove("alert");
                                    statusDot.classList.remove("alert");
                                }, 5000);
                            } else {
                                if (!statusText.classList.contains("alert")) {
                                    statusText.innerText = "✅ Normal";
                                    statusText.classList.remove("alert");
                                    statusDot.classList.remove("alert");
                                }
                            }

                            systemStatus.innerText = "Online";

                        } else if(!data.throttled) {
                            console.warn("Frame processing failed:", data.error);
                        }

                    } catch (err) {
                        console.error("Fetch error:", err);
                        systemStatus.innerText = "Error";
                    }

                }, "image/jpeg", 0.8);

            } catch (err) {
                console.error("Frame processing error:", err);
            }

            requestAnimationFrame(processFrame);
        };

        processFrame();

    } catch (err) {
        console.error("Camera error:", err);
        loadingSpinner.innerHTML = `
            <div class="spinner-error">
                <p style="color: var(--danger); font-weight: bold;">❌ Camera Access Denied</p>
                <small style="color: var(--text-secondary);">Please allow camera permissions and refresh</small>
            </div>
        `;
        systemStatus.innerText = "Camera Error";
    }
}

// Start camera when page loads
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startCamera);
} else {
    startCamera();
}
