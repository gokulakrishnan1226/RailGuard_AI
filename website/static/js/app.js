// RailGuard AI - Real-time Dashboard Logic

document.addEventListener("DOMContentLoaded", () => {
    // 1. Session check
    const token = localStorage.getItem("access_token");
    if (!token && window.location.pathname === "/dashboard") {
        window.location.href = "/login";
        return;
    }

    // Set Authorization header for api calls
    const headers = {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
    };

    // Keep track of alert IDs to avoid repeating alarm audio
    let registeredAlerts = new Set();
    
    // Web audio context buzzer synth for local alerts
    function playBuzzerSound() {
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            
            oscillator.type = "sawtooth";
            oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // High pitch A5
            
            gainNode.gain.setValueAtTime(0.12, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
            
            oscillator.start(audioCtx.currentTime);
            oscillator.stop(audioCtx.currentTime + 0.35);
        } catch (e) {
            console.log("Audio buzzer failed to initialize: " + e);
        }
    }

    // Fetch Alerts and log History
    function fetchTelemetryData() {
        // Fetch Status and Cleanliness
        fetch("/api/status", { headers })
            .then(res => {
                if (res.status === 401) logout();
                return res.json();
            })
            .then(data => {
                updateTelemetryUI(data);
            })
            .catch(err => console.error("Telemetry error:", err));

        // Fetch location
        fetch("/api/location", { headers })
            .then(res => res.json())
            .then(data => {
                document.getElementById("gps-coords").innerText = `${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}`;
                updateMockMap(data.latitude, data.longitude);
            })
            .catch(err => console.error("Location error:", err));

        // Fetch Alert Log
        fetch("/api/alerts?limit=15", { headers })
            .then(res => res.json())
            .then(data => {
                renderAlertList(data);
            })
            .catch(err => console.error("Alerts list error:", err));
    }

    // Update Telemetry Status Cards
    function updateTelemetryUI(data) {
        // Cleanliness Update
        const cleanEl = document.getElementById("cleanliness-status");
        cleanEl.innerText = data.cleanliness.replace("_", " ");
        if (data.cleanliness === "Clean") {
            cleanEl.className = "status-badge badge-green";
        } else if (data.cleanliness === "Dirty") {
            cleanEl.className = "status-badge badge-yellow";
        } else {
            cleanEl.className = "status-badge badge-red";
        }

        // Damage Update
        const damageEl = document.getElementById("damage-status");
        damageEl.innerText = data.damage.replace("_", " ");
        if (data.damage === "Normal_Track") {
            damageEl.className = "status-badge badge-green";
        } else {
            damageEl.className = "status-badge badge-red";
        }

        // Vibration Update
        const vibeEl = document.getElementById("vibration-level");
        vibeEl.innerText = `${data.vibration_level.toFixed(2)} g`;
        if (data.vibration_level > 5.0) {
            vibeEl.classList.add("text-danger");
        } else {
            vibeEl.classList.remove("text-danger");
        }
    }

    // Render Alerts logs
    function renderAlertList(alerts) {
        const listContainer = document.getElementById("alert-log-list");
        if (alerts.length === 0) {
            listContainer.innerHTML = `<div class="text-muted text-center py-4">No security threats detected. System safe.</div>`;
            return;
        }

        let html = "";
        let newAlertTriggered = false;

        alerts.forEach(alert => {
            const dateStr = new Date(alert.timestamp).toLocaleTimeString();
            const badgeType = alert.officer_verified ? "badge-green" : "badge-red";
            const verifyLabel = alert.officer_verified ? "Authorized Officer" : "Intrusion Alert";
            
            // Check if alert is newly discovered
            if (!registeredAlerts.has(alert.id)) {
                registeredAlerts.add(alert.id);
                if (!alert.officer_verified && registeredAlerts.size > 1) {
                    newAlertTriggered = true;
                }
            }

            html += `
                <div class="alert-row d-flex justify-content-between align-items-center">
                    <div>
                        <strong class="d-block text-white">${alert.detection_type}</strong>
                        <small class="text-muted">Time: ${dateStr} | GPS: ${alert.gps_latitude.toFixed(4)}, ${alert.gps_longitude.toFixed(4)}</small>
                    </div>
                    <div class="text-end">
                        <span class="status-badge ${badgeType} mb-1">${verifyLabel}</span>
                        <div class="text-white-50" style="font-size: 0.8rem;">Conf: ${(alert.confidence * 100).toFixed(0)}%</div>
                    </div>
                </div>
            `;
        });

        listContainer.innerHTML = html;

        if (newAlertTriggered) {
            playBuzzerSound();
            triggerVisualAlertEffect();
        }
    }

    // Flashes red borders around page briefly on dangerous alert
    function triggerVisualAlertEffect() {
        document.body.style.boxShadow = "inset 0 0 40px rgba(255, 61, 0, 0.4)";
        setTimeout(() => {
            document.body.style.boxShadow = "none";
        }, 1000);
    }

    // Mock map drawing using canvas element (simulates a moving GPS tracker)
    const mapCanvas = document.getElementById("map-canvas");
    let mapCtx = mapCanvas ? mapCanvas.getContext("2d") : null;

    function updateMockMap(lat, lon) {
        if (!mapCtx) return;
        const w = mapCanvas.width;
        const h = mapCanvas.height;
        
        // Dark background
        mapCtx.fillStyle = "#111424";
        mapCtx.fillRect(0, 0, w, h);
        
        // Draw grid
        mapCtx.strokeStyle = "rgba(255, 255, 255, 0.03)";
        mapCtx.lineWidth = 1;
        for (let x = 0; x < w; x += 30) {
            mapCtx.beginPath();
            mapCtx.moveTo(x, 0);
            mapCtx.lineTo(x, h);
            mapCtx.stroke();
        }
        for (let y = 0; y < h; y += 30) {
            mapCtx.beginPath();
            mapCtx.moveTo(0, y);
            mapCtx.lineTo(w, y);
            mapCtx.stroke();
        }
        
        // Draw simulated rail line
        mapCtx.strokeStyle = "#00bcd4";
        mapCtx.lineWidth = 4;
        mapCtx.beginPath();
        mapCtx.moveTo(20, h/2);
        mapCtx.lineTo(w - 20, h/2);
        mapCtx.stroke();
        
        // Draw patrolling vehicle location marker (blinking red circle)
        const markerX = (w / 2) + Math.sin(Date.now() / 1500) * 40;
        const markerY = h / 2;
        
        mapCtx.fillStyle = "rgba(255, 61, 0, 0.25)";
        mapCtx.beginPath();
        mapCtx.arc(markerX, markerY, 20 + Math.sin(Date.now() / 200) * 5, 0, 2*Math.PI);
        mapCtx.fill();
        
        mapCtx.fillStyle = "#ff3d00";
        mapCtx.beginPath();
        mapCtx.arc(markerX, markerY, 6, 0, 2*Math.PI);
        mapCtx.fill();
        
        mapCtx.fillStyle = "#ffffff";
        mapCtx.font = "10px Outfit";
        mapCtx.fillText("Inspection Vehicle", markerX - 45, markerY - 14);
    }

    // Trigger polling loops
    if (window.location.pathname === "/dashboard") {
        fetchTelemetryData();
        setInterval(fetchTelemetryData, 1500); // 1.5 seconds frequency
    }

    // Logout
    const logoutBtn = document.getElementById("btn-logout");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", logout);
    }

    function logout() {
        localStorage.removeItem("access_token");
        window.location.href = "/login";
    }
});
