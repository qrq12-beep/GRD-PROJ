/**
 * SENTINEL — Campus Monitor
 * Connects to:
 *   Flask (server.py):  GET /api/state, GET /video_feed
 *   FastAPI (api.py):   GET /alerts, WebSocket /ws, GET /stream
 *
 * Poll interval: 1 500 ms  (adjust POLL_MS as needed)
 */

"use strict";

// ══════════════════════════════════════════════════
//  CONFIG
// ══════════════════════════════════════════════════
const FLASK_BASE  = "http://localhost:5000";   // server.py
const API_BASE    = "http://localhost:8000";   // api.py
const WS_URL      = "ws://localhost:8000/ws";  // api.py WebSocket
const POLL_MS     = 1500;                      // state polling interval

const CAMERAS = [
  { id: "canteen",    label: "CANTEEN",     index: 0 },
  { id: "hallway",    label: "HALLWAY",     index: 0 },
  { id: "classroom1", label: "CLASSROOM 1", index: 0 },
  { id: "classroom2", label: "CLASSROOM 2", index: 0 },
  { id: "entrance",   label: "ENTRANCE",    index: 0 },
];

// ══════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════
let state = {
  running:     false,
  annotate:    false,
  activeCam:   "canteen",
  alertFilter: "all",
  alerts:      [],      // { id, type, cam, detail, time, severity }
  ws:          null,
  pollTimer:   null,
  sessionStart: null,
  frameCount:  0,
  alertCount:  0,
  uptimeTimer: null,
};

// ══════════════════════════════════════════════════
//  DOM REFS
// ══════════════════════════════════════════════════
const $ = id => document.getElementById(id);

const DOM = {
  connStatus:    $("connStatus"),
  clock:         $("clock"),
  // feed
  feedImg:       $("feedImg"),
  feedCamName:   $("feedCamName"),
  feedAlert:     $("feedAlert"),
  feedOffline:   $("feedOffline"),
  fpsBadge:      $("fpsBadge"),
  personsBadge:  $("personsBadge"),
  // conf bars
  violenceBar:   $("violenceBar"),
  violencePct:   $("violencePct"),
  litterBar:     $("litterBar"),
  litterPct:     $("litterPct"),
  fireBar:       $("fireBar"),
  firePct:       $("firePct"),
  // controls
  btnStart:      $("btnStart"),
  btnStop:       $("btnStop"),
  annotateToggle:$("annotateToggle"),
  modelStatus:   $("modelStatus"),
  // stat cards
  cardViolence:  $("cardViolence"),
  cardLitter:    $("cardLitter"),
  cardFire:      $("cardFire"),
  statViolence:  $("statViolence"),
  statLitter:    $("statLitter"),
  statFire:      $("statFire"),
  statPersons:   $("statPersons"),
  pulseViolence: $("pulseViolence"),
  pulseLitter:   $("pulseLitter"),
  pulseFire:     $("pulseFire"),
  // session
  sessFrames:    $("sessFrames"),
  sessAlerts:    $("sessAlerts"),
  sessFPS:       $("sessFPS"),
  sessUptime:    $("sessUptime"),
  // alerts
  alertList:     $("alertList"),
  // cam grid
  camGrid:       $("camGrid"),
  // log
  logTableBody:  $("logTableBody"),
  logFilterCam:  $("logFilterCam"),
  logFilterType: $("logFilterType"),
};

// ══════════════════════════════════════════════════
//  CLOCK
// ══════════════════════════════════════════════════
function tickClock() {
  const now = new Date();
  DOM.clock.textContent = now.toTimeString().slice(0, 8);
}
setInterval(tickClock, 1000);
tickClock();

// ══════════════════════════════════════════════════
//  TAB NAVIGATION
// ══════════════════════════════════════════════════
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ══════════════════════════════════════════════════
//  CAMERA TABS (dashboard)
// ══════════════════════════════════════════════════
document.querySelectorAll(".cam-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".cam-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    state.activeCam = tab.dataset.cam;
    DOM.feedCamName.textContent = tab.dataset.label;
    // All cameras point to the same /video_feed in this single-source demo.
    // In a real multi-cam setup, append ?source=<index> when server supports it.
    DOM.feedImg.src = `${FLASK_BASE}/video_feed?_ts=${Date.now()}`;
  });
});

// ══════════════════════════════════════════════════
//  START / STOP CONTROLS
// ══════════════════════════════════════════════════
DOM.btnStart.addEventListener("click", () => {
  state.running      = true;
  state.sessionStart = Date.now();
  state.frameCount   = 0;
  state.alertCount   = 0;
  DOM.btnStart.disabled = true;
  DOM.btnStop.disabled  = false;

  // Point video feed at Flask MJPEG
  DOM.feedImg.src = `${FLASK_BASE}/video_feed`;
  DOM.feedOffline.style.display = "none";

  startPolling();
  startUptime();
  openWebSocket();
  setConnStatus("connected", "Live");
});

DOM.btnStop.addEventListener("click", () => {
  state.running = false;
  DOM.btnStart.disabled = false;
  DOM.btnStop.disabled  = true;

  stopPolling();
  stopUptime();
  closeWebSocket();
  DOM.feedImg.src = "";
  DOM.feedOffline.style.display = "flex";
  setConnStatus("idle", "Stopped");
});

DOM.annotateToggle.addEventListener("change", () => {
  state.annotate = DOM.annotateToggle.checked;
});

// ══════════════════════════════════════════════════
//  CONNECTION STATUS HELPER
// ══════════════════════════════════════════════════
function setConnStatus(mode, label) {
  DOM.connStatus.className = "conn-status";
  if (mode === "connected") DOM.connStatus.classList.add("connected");
  if (mode === "error")     DOM.connStatus.classList.add("error");
  DOM.connStatus.querySelector(".conn-label").textContent = label;
}

// ══════════════════════════════════════════════════
//  POLL /api/state  (Flask — server.py)
// ══════════════════════════════════════════════════
function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  pollState();
  state.pollTimer = setInterval(pollState, POLL_MS);
}
function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function pollState() {
  try {
    const res  = await fetch(`${FLASK_BASE}/api/state`, { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    applyState(data);
    setConnStatus("connected", "Live");
  } catch (err) {
    // Flask not reachable — try FastAPI /status as fallback
    try {
      const res2  = await fetch(`${API_BASE}/status`, { cache: "no-store" });
      const data2 = await res2.json();
      applyStatusFallback(data2);
      setConnStatus("connected", "API");
    } catch {
      setConnStatus("error", "No Signal");
    }
  }
}

// ── Apply Flask /api/state payload ────────────────
function applyState(d) {
  state.frameCount++;
  DOM.sessFrames.textContent = state.frameCount;

  // Models ready
  if (d.models_ready) {
    DOM.modelStatus.className = "model-status ready";
    DOM.modelStatus.querySelector(".ms-label").textContent = "Models Ready";
  }

  // FPS
  const fps = d.fps != null ? Number(d.fps).toFixed(1) : "--";
  DOM.fpsBadge.textContent = `${fps} FPS`;
  DOM.sessFPS.textContent  = fps;

  // Person count
  const persons = d.person_count ?? 0;
  DOM.personsBadge.textContent = `${persons} person${persons !== 1 ? "s" : ""}`;
  DOM.statPersons.textContent  = persons;

  // Violence
  const vConf = d.v_conf != null ? Math.round(d.v_conf * 100) : 0;
  DOM.violenceBar.style.width = `${vConf}%`;
  DOM.violencePct.textContent = `${vConf}%`;
  setCard("violence", d.is_violent, `${vConf}%`, "CLEAR");

  if (d.is_violent) triggerAlert("violence", state.activeCam, `Confidence ${vConf}%`);

  // Littering
  const litterCnt = d.litter_dets ? d.litter_dets.length : 0;
  const litterPct  = Math.min(litterCnt * 25, 100);
  DOM.litterBar.style.width = `${litterPct}%`;
  DOM.litterPct.textContent = `${litterCnt} det`;
  setCard("litter", litterCnt > 0, litterCnt, 0);

  if (litterCnt > 0) triggerAlert("littering", state.activeCam, `${litterCnt} detection${litterCnt !== 1 ? "s" : ""}`);

  // Fire / Smoke
  const fireActive = d.is_fire || d.is_smoke;
  const fireLabel  = d.is_fire && d.is_smoke ? "FIRE+SMOKE" : d.is_fire ? "FIRE" : d.is_smoke ? "SMOKE" : "CLEAR";
  DOM.fireBar.style.width = fireActive ? "100%" : "0%";
  DOM.firePct.textContent = fireLabel;
  setCard("fire", fireActive, fireLabel, "CLEAR");

  if (fireActive) triggerAlert("fire", state.activeCam, fireLabel);

  // Feed overlay
  const anyAlert = d.is_violent || litterCnt > 0 || fireActive;
  if (anyAlert) {
    const labels = [];
    if (d.is_violent) labels.push(`⚠ VIOLENCE ${vConf}%`);
    if (litterCnt)    labels.push(`◈ LITTER ×${litterCnt}`);
    if (fireActive)   labels.push(`▲ ${fireLabel}`);
    DOM.feedAlert.textContent = labels.join("   ");
    DOM.feedAlert.style.display = "block";
  } else {
    DOM.feedAlert.style.display = "none";
  }
}

// ── Fallback: FastAPI /status ──────────────────────
function applyStatusFallback(d) {
  if (d.models_ready) {
    DOM.modelStatus.className = "model-status ready";
    DOM.modelStatus.querySelector(".ms-label").textContent = "Models Ready";
  }
}

// ══════════════════════════════════════════════════
//  CARD STATE HELPER
// ══════════════════════════════════════════════════
function setCard(type, triggered, activeVal, clearVal) {
  const cardMap = { violence: DOM.cardViolence, litter: DOM.cardLitter, fire: DOM.cardFire };
  const valMap  = { violence: DOM.statViolence, litter: DOM.statLitter, fire: DOM.statFire };
  const card = cardMap[type];
  const val  = valMap[type];
  if (!card) return;
  if (triggered) {
    card.classList.add("triggered");
    val.textContent = activeVal;
  } else {
    card.classList.remove("triggered");
    val.textContent = clearVal;
  }
}

// ══════════════════════════════════════════════════
//  ALERT MANAGEMENT
// ══════════════════════════════════════════════════
const recentAlertKeys = new Set();  // debounce duplicate alerts within 5s

function triggerAlert(type, cam, detail) {
  const key = `${type}-${cam}-${Math.floor(Date.now() / 5000)}`;
  if (recentAlertKeys.has(key)) return;
  recentAlertKeys.add(key);
  setTimeout(() => recentAlertKeys.delete(key), 6000);

  const alert = {
    id:       Date.now(),
    type,
    cam,
    detail,
    time:     new Date(),
    severity: type === "violence" ? "HIGH" : type === "fire" ? "HIGH" : "MEDIUM",
  };
  state.alerts.unshift(alert);
  if (state.alerts.length > 200) state.alerts.pop();
  state.alertCount++;
  DOM.sessAlerts.textContent = state.alertCount;

  renderAlertList();
  renderLogTable();
  updateCamGridAlerts(cam, type);
}

// ── Render sidebar alert list ──────────────────────
function renderAlertList() {
  const filter = state.alertFilter;
  const filtered = filter === "all"
    ? state.alerts
    : state.alerts.filter(a => a.type === filter);

  if (!filtered.length) {
    DOM.alertList.innerHTML = `<div class="alert-empty">No alerts to show.</div>`;
    return;
  }

  DOM.alertList.innerHTML = filtered.slice(0, 40).map(a => `
    <div class="alert-item" data-type="${a.type}">
      <div class="alert-sev sev-${a.type}"></div>
      <div class="alert-meta">
        <div class="alert-type alert-type-${a.type}">${a.type.toUpperCase()}</div>
        <div class="alert-info">${fmtCam(a.cam)} · ${a.detail} · ${fmtTime(a.time)}</div>
      </div>
    </div>
  `).join("");
}

// Filter pills
document.querySelectorAll(".pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    state.alertFilter = pill.dataset.filter;
    renderAlertList();
  });
});

$("btnClearAlerts").addEventListener("click", () => {
  state.alerts = [];
  state.alertCount = 0;
  DOM.sessAlerts.textContent = 0;
  renderAlertList();
  renderLogTable();
});

$("btnExportAlerts").addEventListener("click", exportCSV);

// ══════════════════════════════════════════════════
//  LOG TABLE (Alert Log tab)
// ══════════════════════════════════════════════════
function renderLogTable() {
  const camFilter  = DOM.logFilterCam.value;
  const typeFilter = DOM.logFilterType.value;

  const filtered = state.alerts.filter(a => {
    const camOk  = camFilter  === "all" || a.cam  === camFilter;
    const typeOk = typeFilter === "all" || a.type === typeFilter;
    return camOk && typeOk;
  });

  if (!filtered.length) {
    DOM.logTableBody.innerHTML = `<tr><td colspan="5" class="log-empty">No matching alerts.</td></tr>`;
    return;
  }

  DOM.logTableBody.innerHTML = filtered.map(a => `
    <tr>
      <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-low)">${fmtTime(a.time)}</td>
      <td style="font-family:var(--font-display);font-size:12px;letter-spacing:1px">${fmtCam(a.cam)}</td>
      <td><span class="alert-type alert-type-${a.type}" style="font-family:var(--font-display);font-size:12px;letter-spacing:1px">${a.type.toUpperCase()}</span></td>
      <td style="color:var(--text-mid);font-size:11px">${a.detail}</td>
      <td><span class="sev-chip ${a.type}">${a.severity}</span></td>
    </tr>
  `).join("");
}

DOM.logFilterCam.addEventListener("change",  renderLogTable);
DOM.logFilterType.addEventListener("change", renderLogTable);
$("btnClearLog").addEventListener("click", () => {
  state.alerts = [];
  renderAlertList();
  renderLogTable();
});

// ══════════════════════════════════════════════════
//  ALL CAMERAS GRID
// ══════════════════════════════════════════════════
function buildCamGrid() {
  DOM.camGrid.innerHTML = CAMERAS.map(cam => `
    <div class="cam-card" id="camcard-${cam.id}">
      <div class="cam-card-header">
        <span class="cam-card-name">${cam.label}</span>
        <span class="cam-status-dot"></span>
      </div>
      <div class="cam-card-feed">
        <img src="${FLASK_BASE}/video_feed" alt="${cam.label}" loading="lazy" />
      </div>
      <div class="cam-card-footer">
        <span class="cam-badge active-badge">ACTIVE</span>
        <span class="cam-badge cam-alert-badge" id="camcard-alert-${cam.id}">ALERT</span>
      </div>
    </div>
  `).join("");
}
buildCamGrid();

function updateCamGridAlerts(camId, type) {
  const card = document.getElementById(`camcard-${camId}`);
  if (!card) return;
  card.classList.add("has-alert");
  const badge = document.getElementById(`camcard-alert-${camId}`);
  if (badge) badge.textContent = type.toUpperCase();
  setTimeout(() => card.classList.remove("has-alert"), 8000);
}

// ══════════════════════════════════════════════════
//  WEBSOCKET  (api.py — for monitor.html style use)
//  Sends frames captured from browser webcam
//  Receives detection results as JSON
// ══════════════════════════════════════════════════
function openWebSocket() {
  if (state.ws && state.ws.readyState < 2) return;
  try {
    state.ws = new WebSocket(WS_URL);
    state.ws.onopen    = () => console.log("[WS] connected to", WS_URL);
    state.ws.onmessage = evt => handleWSMessage(JSON.parse(evt.data));
    state.ws.onerror   = () => console.warn("[WS] error — continuing on poll");
    state.ws.onclose   = () => console.log("[WS] closed");
  } catch (e) {
    console.warn("[WS] not available:", e.message);
  }
}

function closeWebSocket() {
  if (state.ws) { state.ws.close(); state.ws = null; }
}

function handleWSMessage(data) {
  // api.py returns same fields when frame is sent; merge into display
  if (data.error) return;
  const d = data;
  const vConf    = d.violence_confidence != null ? Math.round(d.violence_confidence * 100) : 0;
  const isViol   = d.is_violent || false;
  const litterCnt = Array.isArray(d.litter_detections) ? d.litter_detections.length : 0;
  const fireActive = d.is_fire || d.is_smoke || false;

  DOM.violenceBar.style.width = `${vConf}%`;
  DOM.violencePct.textContent = `${vConf}%`;
  setCard("violence", isViol, `${vConf}%`, "CLEAR");

  const litterPct = Math.min(litterCnt * 25, 100);
  DOM.litterBar.style.width = `${litterPct}%`;
  DOM.litterPct.textContent = `${litterCnt} det`;
  setCard("litter", litterCnt > 0, litterCnt, 0);

  const fireLabel = fireActive ? (d.is_fire && d.is_smoke ? "FIRE+SMOKE" : d.is_fire ? "FIRE" : "SMOKE") : "CLEAR";
  DOM.fireBar.style.width = fireActive ? "100%" : "0%";
  DOM.firePct.textContent = fireLabel;
  setCard("fire", fireActive, fireLabel, "CLEAR");
}

// ══════════════════════════════════════════════════
//  SESSION UPTIME
// ══════════════════════════════════════════════════
function startUptime() {
  if (state.uptimeTimer) clearInterval(state.uptimeTimer);
  state.uptimeTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
    const m = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const s = (elapsed % 60).toString().padStart(2, "0");
    DOM.sessUptime.textContent = `${m}:${s}`;
  }, 1000);
}
function stopUptime() {
  if (state.uptimeTimer) { clearInterval(state.uptimeTimer); state.uptimeTimer = null; }
  DOM.sessUptime.textContent = "00:00";
}

// ══════════════════════════════════════════════════
//  EXPORT CSV
// ══════════════════════════════════════════════════
function exportCSV() {
  if (!state.alerts.length) return;
  const header = "Time,Camera,Type,Detail,Severity\n";
  const rows   = state.alerts.map(a =>
    [fmtTime(a.time), fmtCam(a.cam), a.type, `"${a.detail}"`, a.severity].join(",")
  ).join("\n");
  const blob = new Blob([header + rows], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `sentinel-alerts-${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════════════════════════════════════
//  UTILS
// ══════════════════════════════════════════════════
function fmtTime(d) {
  if (!(d instanceof Date)) d = new Date(d);
  return d.toTimeString().slice(0, 8);
}
function fmtCam(id) {
  const cam = CAMERAS.find(c => c.id === id);
  return cam ? cam.label : id.toUpperCase();
}

// ══════════════════════════════════════════════════
//  INITIAL UI STATE
// ══════════════════════════════════════════════════
DOM.feedOffline.style.display = "flex";
DOM.feedImg.src = "";
setConnStatus("idle", "Not started");

// Try pinging Flask once on load to detect if backend is already running
(async () => {
  try {
    const res = await fetch(`${FLASK_BASE}/api/state`, { cache: "no-store" });
    if (res.ok) {
      const d = await res.json();
      if (d.models_ready) {
        DOM.modelStatus.className = "model-status ready";
        DOM.modelStatus.querySelector(".ms-label").textContent = "Models Ready";
      }
    }
  } catch {}
})();
