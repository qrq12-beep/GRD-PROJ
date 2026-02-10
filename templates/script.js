// ===========================
// AI Monitoring System Scripts
// ===========================

// Sidebar Toggle (Mobile)
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarClose = document.getElementById('sidebarClose');
    const mainContent = document.querySelector('.main-content');

    // Toggle sidebar on mobile
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }

    // Close sidebar
    if (sidebarClose) {
        sidebarClose.addEventListener('click', function() {
            sidebar.classList.remove('active');
        });
    }

    // Close sidebar when clicking outside on mobile
    mainContent.addEventListener('click', function() {
        if (window.innerWidth < 992) {
            sidebar.classList.remove('active');
        }
    });

    // Navigation Menu
    const menuLinks = document.querySelectorAll('.sidebar-menu a');
    const contentSections = document.querySelectorAll('.content-section');

    menuLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all menu items
            menuLinks.forEach(l => l.parentElement.classList.remove('active'));
            
            // Add active class to clicked item
            this.parentElement.classList.add('active');
            
            // Hide all content sections
            contentSections.forEach(section => section.classList.remove('active'));
            
            // Show selected section
            const sectionId = this.dataset.section + '-section';
            const targetSection = document.getElementById(sectionId);
            if (targetSection) {
                targetSection.classList.add('active');
            }
            
            // Close sidebar on mobile
            if (window.innerWidth < 992) {
                sidebar.classList.remove('active');
            }
        });
    });

    // Camera Card Click - Open Modal
    const cameraCards = document.querySelectorAll('.camera-card');
    const cameraModal = new bootstrap.Modal(document.getElementById('cameraModal'));

    cameraCards.forEach(card => {
        card.addEventListener('click', function() {
            const cameraId = this.dataset.camera;
            openCameraModal(cameraId);
        });
    });

    function openCameraModal(cameraId) {
        // Sample data - in production, this would come from your backend
        const cameraData = {
            '1': {
                id: 'Camera 01',
                location: 'Main Entrance',
                status: 'Anomaly Detected',
                lastAnomaly: 'Fight Detected',
                confidence: '94%',
                events: [
                    { type: 'Fight Detected', time: '2 mins ago', confidence: '94%' },
                    { type: 'Normal Activity', time: '1 hour ago', confidence: '-' },
                    { type: 'Suspicious Activity', time: '3 hours ago', confidence: '76%' }
                ]
            },
            '2': {
                id: 'Camera 02',
                location: 'Lobby Area',
                status: 'Normal',
                lastAnomaly: 'None',
                confidence: '-',
                events: [
                    { type: 'Normal Activity', time: 'Live', confidence: '-' }
                ]
            },
            '3': {
                id: 'Camera 03',
                location: 'Parking Lot',
                status: 'Suspicious Activity',
                lastAnomaly: 'Suspicious Activity',
                confidence: '78%',
                events: [
                    { type: 'Suspicious Activity', time: '15 mins ago', confidence: '78%' },
                    { type: 'Normal Activity', time: '2 hours ago', confidence: '-' }
                ]
            },
            '6': {
                id: 'Camera 06',
                location: 'Back Alley',
                status: 'Anomaly Detected',
                lastAnomaly: 'Littering Detected',
                confidence: '88%',
                events: [
                    { type: 'Littering Detected', time: '45 mins ago', confidence: '88%' },
                    { type: 'Normal Activity', time: '2 hours ago', confidence: '-' }
                ]
            }
        };

        const data = cameraData[cameraId] || {
            id: `Camera ${cameraId.padStart(2, '0')}`,
            location: 'Unknown',
            status: 'Normal',
            lastAnomaly: 'None',
            confidence: '-',
            events: [{ type: 'Normal Activity', time: 'Live', confidence: '-' }]
        };

        // Update modal content
        document.getElementById('modalCameraId').textContent = data.id;
        document.getElementById('modalCameraLocation').textContent = data.location;
        document.getElementById('modalCameraStatus').textContent = data.status;
        document.getElementById('modalLastAnomaly').textContent = data.lastAnomaly;
        document.getElementById('modalConfidence').textContent = data.confidence;

        // Populate event history
        const eventHistory = document.getElementById('eventHistory');
        eventHistory.innerHTML = '';
        data.events.forEach(event => {
            const eventItem = document.createElement('div');
            eventItem.className = 'event-item';
            eventItem.innerHTML = `
                <div class="event-type">${event.type}</div>
                <div class="event-time">${event.time} ${event.confidence !== '-' ? '• Confidence: ' + event.confidence : ''}</div>
            `;
            eventHistory.appendChild(eventItem);
        });

        // Show modal
        cameraModal.show();
    }

    // Initialize Charts
    initializeCharts();

    // Simulate real-time updates
    simulateRealTimeUpdates();
});

// Chart Initialization
function initializeCharts() {
    // Chart.js default options
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';

    // Timeline Chart (Anomalies Over Time)
    const timelineCtx = document.getElementById('timelineChart');
    if (timelineCtx) {
        new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59'],
                datasets: [{
                    label: 'Fight',
                    data: [2, 1, 3, 5, 4, 6, 3],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Littering',
                    data: [1, 2, 1, 3, 2, 4, 2],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Suspicious Activity',
                    data: [3, 2, 4, 3, 5, 3, 4],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#334155'
                        }
                    },
                    x: {
                        grid: {
                            color: '#334155'
                        }
                    }
                }
            }
        });
    }

    // Pie Chart (Anomaly Types)
    const pieCtx = document.getElementById('pieChart');
    if (pieCtx) {
        new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: ['Fight', 'Littering', 'Suspicious Activity'],
                datasets: [{
                    data: [35, 25, 40],
                    backgroundColor: [
                        '#ef4444',
                        '#f59e0b',
                        '#3b82f6'
                    ],
                    borderColor: '#1e293b',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Bar Chart (Anomalies Per Camera)
    const barCtx = document.getElementById('barChart');
    if (barCtx) {
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Cam 01', 'Cam 02', 'Cam 03', 'Cam 04', 'Cam 05', 'Cam 06', 'Cam 07', 'Cam 08'],
                datasets: [{
                    label: 'Total Anomalies',
                    data: [12, 3, 8, 5, 2, 15, 4, 6],
                    backgroundColor: [
                        '#ef4444',
                        '#22c55e',
                        '#f59e0b',
                        '#22c55e',
                        '#22c55e',
                        '#ef4444',
                        '#22c55e',
                        '#22c55e'
                    ],
                    borderColor: '#1e293b',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#334155'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
}

// Simulate Real-Time Updates
function simulateRealTimeUpdates() {
    // Update timestamp every minute
    setInterval(() => {
        const timestamps = document.querySelectorAll('.timestamp');
        // In a real application, you would fetch actual timestamps from your backend
    }, 60000);

    // Simulate random alerts (for demo purposes)
    setInterval(() => {
        if (Math.random() > 0.95) { // 5% chance every 5 seconds
            addNewAlert();
        }
    }, 5000);
}

// Add New Alert (Demo Function)
function addNewAlert() {
    const alertTypes = [
        { type: 'Fight Detected', severity: 'danger', icon: 'bi-exclamation-triangle-fill' },
        { type: 'Littering Detected', severity: 'danger', icon: 'bi-trash-fill' },
        { type: 'Suspicious Activity', severity: 'warning', icon: 'bi-exclamation-circle-fill' }
    ];
    
    const cameras = ['Camera 01', 'Camera 02', 'Camera 03', 'Camera 04', 'Camera 05', 'Camera 06'];
    const locations = ['Main Entrance', 'Lobby Area', 'Parking Lot', 'Corridor A', 'Stairwell B', 'Back Alley'];
    
    const randomAlert = alertTypes[Math.floor(Math.random() * alertTypes.length)];
    const randomCamera = Math.floor(Math.random() * cameras.length);
    const confidence = Math.floor(Math.random() * 20) + 75; // 75-95%
    
    // Update alert count
    const alertCount = document.getElementById('alertCount');
    const currentCount = parseInt(alertCount.textContent);
    alertCount.textContent = currentCount + 1;
    
    // Add visual notification
    showToast(randomAlert.type, cameras[randomCamera]);
}

// Toast Notification (Optional)
function showToast(alertType, camera) {
    // You can implement a custom toast notification here
    console.log(`New Alert: ${alertType} at ${camera}`);
}

// Update Statistics (Called from backend in production)
function updateStatistics(data) {
    document.getElementById('activeCameras').textContent = data.activeCameras || '12';
    document.getElementById('activeAlerts').textContent = data.activeAlerts || '3';
    document.getElementById('todayEvents').textContent = data.todayEvents || '27';
    document.getElementById('systemUptime').textContent = data.systemUptime || '99.8%';
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        updateStatistics,
        addNewAlert
    };
}