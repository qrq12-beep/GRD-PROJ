// DOM Elements
const videoOutput = document.getElementById("videoOutput");
const alertCounter = document.getElementById("activeAlerts");
const statusText = document.getElementById("statusText");
const fpsCounter = document.getElementById("fpsCounter");
const lastUpdate = document.getElementById("lastUpdate");
const peopleDetected = document.getElementById("peopleDetected");
const systemStatus = document.getElementById("systemStatus");
const loadingSpinner = document.getElementById("loadingSpinner");
const statusBadge = document.getElementById("statusBadge");
const alertsList = document.getElementById("alertsList");
const alertsCenter = document.getElementById("alertsCenter");
const detectionLog = document.getElementById("detectionLog");
const fightCount = document.getElementById("fightCount");
const frameCount = document.getElementById("frameCount");
const avgConfidence = document.getElementById("avgConfidence");
const clearLogsBtn = document.getElementById("clearLogsBtn");
const alertBadge = document.getElementById("alertBadge");
const fpsLimit = document.getElementById("fpsLimit");
const sensitivity = document.getElementById("sensitivity");
const fpsLimitValue = document.getElementById("fpsLimitValue");
const sensitivityValue = document.getElementById("sensitivityValue");

// Statistics
let alertCount = 0;
let frameProcessedCount = 0;
let fightDetectionCount = 0;
let totalConfidence = 0;
let confidenceCount = 0;
let lastFpsTime = Date.now();
let frameCount_fps = 0;
let detectionHistory = [];
let maxConfidence = 0;
const maxHistoryItems = 50;

// Settings
let fps_limit = 20;
let detection_sensitivity = 0.4;
let soundEnabled = true;

// Initialize settings listeners
fpsLimit.addEventListener('input', (e) => {
    fps_limit = parseInt(e.target.value);
    fpsLimitValue.innerText = fps_limit;
});

sensitivity.addEventListener('input', (e) => {
    detection_sensitivity = parseFloat(e.target.value);
    sensitivityValue.innerText = detection_sensitivity;
});

document.getElementById('soundAlerts').addEventListener('change', (e) => {
    soundEnabled = e.target.checked;
});

clearLogsBtn.addEventListener('click', () => {
    detectionHistory = [];
    detectionLog.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-3">Logs cleared</td></tr>';
});

// Update timestamp
function updateTimestamp() {
    const now = new Date();
    lastUpdate.innerText = now.toLocaleTimeString();
}

// Update FPS
function updateFps() {
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        fpsCounter.innerText = frameCount_fps;
        frameCount_fps = 0;
        lastFpsTime = now;
    }
}

// Add alert
function addAlert(type, message) {
    const alertItem = document.createElement('div');
    alertItem.className = `alert-item ${type === 'fight' ? 'danger' : ''}`;
    alertItem.innerHTML = `
        <div class="alert-item-time">${new Date().toLocaleTimeString()}</div>
        <div class="alert-item-text">${message}</div>
    `;
    
    // Add to recent alerts
    if (alertsList.querySelector('p')) {
        alertsList.innerHTML = '';
    }
    alertsList.insertBefore(alertItem, alertsList.firstChild);
    if (alertsList.children.length > 5) {
        alertsList.removeChild(alertsList.lastChild);
    }

    // Add to alert center
    const centerAlert = alertItem.cloneNode(true);
    if (alertsCenter.querySelector('p')) {
        alertsCenter.innerHTML = '';
    }
    alertsCenter.insertBefore(centerAlert, alertsCenter.firstChild);
    if (alertsCenter.children.length > 20) {
        alertsCenter.removeChild(alertsCenter.lastChild);
    }
}

// Add to detection log
function addDetectionLog(type, confidence, peopleCount) {
    const now = new Date();
    const row = document.createElement('tr');
    
    const typeLabel = type === 'fight' ? '🔴 FIGHT' : '✓ Normal';
    const typeColor = type === 'fight' ? 'text-danger' : 'text-success';
    
    row.innerHTML = `
        <td>${now.toLocaleTimeString()}</td>
        <td><span class="${typeColor}">${typeLabel}</span></td>
        <td>${confidence}%</td>
        <td>${peopleCount}</td>
    `;
    
    if (detectionLog.querySelector('td[colspan]')) {
        detectionLog.innerHTML = '';
    }
    
    detectionLog.insertBefore(row, detectionLog.firstChild);
    
    // Keep max items
    while (detectionLog.children.length > 20) {
        detectionLog.removeChild(detectionLog.lastChild);
    }

    // Store in history
    detectionHistory.unshift({
        time: now,
        type: type,
        confidence: confidence,
        peopleCount: peopleCount
    });
    if (detectionHistory.length > maxHistoryItems) {
        detectionHistory.pop();
    }
}

// Play alert sound
function playAlertSound() {
    if (!soundEnabled) return;
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

// Update stats
function updateStats(fightDetected, peopleCount, confidence) {
    frameProcessedCount++;
    frameCount.innerText = frameProcessedCount;
    frameProgress.style.width = Math.min((frameProcessedCount / 1000) * 100, 100) + '%';

    if (confidence) {
        totalConfidence += confidence;
        confidenceCount++;
        maxConfidence = Math.max(maxConfidence, confidence);
        const avg = Math.round(totalConfidence / confidenceCount);
        avgConfidence.innerText = avg + '%';
        confidenceProgress.style.width = avg + '%';
    }

    if (fightDetected) {
        fightDetectionCount++;
        fightCount.innerText = fightDetectionCount;
        fightProgress.style.width = Math.min((fightDetectionCount / 20) * 100, 100) + '%';
    }

    peopleProgress.style.width = Math.min((peopleCount / 50) * 100, 100) + '%';
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

        let lastProcessTime = 0;
        const frameInterval = 1000 / fps_limit;

        // Process frames with animation frame
        const processFrame = async () => {
            const now = Date.now();
            
            if (now - lastProcessTime >= frameInterval) {
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
                                frameCount_fps++;
                                updateFps();

                                // Update people count
                                let personCount = 0;
                                let confidenceValue = 0;
                                
                                if (data.detections && data.detections.length > 0) {
                                    personCount = data.detections.filter(d => d.class === 'person').length;
                                    confidenceValue = Math.round(
                                        (data.detections.reduce((sum, d) => sum + (d.confidence || 0), 0) / data.detections.length) || 0
                                    );
                                }
                                
                                peopleDetected.innerText = personCount;
                                updateStats(data.fight, personCount, confidenceValue);

                                // Handle fight detection
                                if (data.fight) {
                                    addAlert('fight', '🔴 Fight or aggressive behavior detected!');
                                    addDetectionLog('fight', confidenceValue, personCount);
                                    playAlertSound();
                                    
                                    statusText.innerText = "🔴 FIGHT DETECTED";
                                    statusText.classList.add("alert");
                                    statusBadge.className = "badge bg-danger";
                                    statusBadge.innerHTML = '<span class="badge-pulse" style="background: #ef4444;"></span> ALERT';

                                    alertCount++;
                                    alertCounter.innerText = alertCount;
                                    alertBadge.style.display = 'inline-block';
                                    alertBadge.innerText = alertCount;
                                    alertProgress.style.width = Math.min((alertCount / 10) * 100, 100) + '%';

                                    // Auto-reset after 5 seconds
                                    setTimeout(() => {
                                        statusText.innerText = "✅ Normal";
                                        statusText.classList.remove("alert");
                                        statusBadge.className = "badge bg-success";
                                        statusBadge.innerHTML = '<span class="badge-pulse"></span> LIVE';
                                    }, 5000);
                                } else {
                                    if (!statusText.classList.contains("alert")) {
                                        statusText.innerText = "✅ Normal";
                                        statusText.classList.remove("alert");
                                        addDetectionLog('normal', confidenceValue, personCount);
                                    }
                                }

                                systemStatus.innerText = "Online";
                                systemStatus.style.color = '#22c55e';

                            } else if (!data.throttled) {
                                console.warn("Frame processing failed:", data.error);
                                systemStatus.innerText = "Error";
                                systemStatus.style.color = '#ef4444';
                            }

                        } catch (err) {
                            console.error("Fetch error:", err);
                            systemStatus.innerText = "Network Error";
                            systemStatus.style.color = '#ef4444';
                        }

                    }, "image/jpeg", 0.8);

                    lastProcessTime = now;

                } catch (err) {
                    console.error("Frame processing error:", err);
                }
            }

            requestAnimationFrame(processFrame);
        };

        processFrame();

    } catch (err) {
        console.error("Camera error:", err);
        loadingSpinner.innerHTML = `
            <div style="text-align: center;">
                <i class="bi bi-exclamation-circle" style="font-size: 3rem; color: var(--danger); display: block; margin-bottom: 1rem;"></i>
                <p style="color: var(--danger); font-weight: bold; margin-bottom: 0.5rem;">❌ Camera Access Denied</p>
                <small style="color: var(--text-secondary);">Please allow camera permissions in your browser and refresh the page.</small>
            </div>
        `;
        systemStatus.innerText = "Camera Error";
        systemStatus.style.color = '#ef4444';
    }
}

// Start camera when page loads
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startCamera);
} else {
    startCamera();
}
