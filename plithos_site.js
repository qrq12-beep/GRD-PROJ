(function () {
  const cfg = window.PLITHOS || {};
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const API_STATE = `${window.location.origin}/api/state`;
  const POLL_MS = Number(cfg.pollMs) || 450;
  const SNAPSHOT_MS = Number(cfg.previewRefreshMs) || 1400;

  let audioCtx = null;
  let alarmTimer = null;
  let alarmVoice = null;
  let alarmMuted = false;
  let currentCameraId = cfg.selectedCameraId || null;
  let alertFilter = "all";
  let latestAlerts = [];

  function ensureAudio() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    if (!audioCtx) audioCtx = new Ctx();
    if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
    return audioCtx;
  }

  function pulseAlarmStep() {
    if (!alarmVoice) return;
    const { ctx, carrierA, carrierB, gain } = alarmVoice;
    const now = ctx.currentTime;
    const phase = alarmVoice.phase % 4;
    const highTone = phase < 2;
    const primary = highTone ? 1180 : 920;
    const secondary = highTone ? 960 : 760;
    carrierA.frequency.cancelScheduledValues(now);
    carrierB.frequency.cancelScheduledValues(now);
    carrierA.frequency.setValueAtTime(primary, now);
    carrierB.frequency.setValueAtTime(secondary, now);
    carrierA.frequency.linearRampToValueAtTime(primary + 40, now + 0.06);
    carrierB.frequency.linearRampToValueAtTime(secondary + 30, now + 0.06);
    gain.gain.cancelScheduledValues(now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
    alarmVoice.phase += 1;
  }

  function startAlarmSound(enabled) {
    if (!enabled || alarmMuted || alarmVoice) return;
    const ctx = ensureAudio();
    if (!ctx) return;
    const carrierA = ctx.createOscillator();
    const carrierB = ctx.createOscillator();
    const gain = ctx.createGain();
    carrierA.type = "square";
    carrierB.type = "square";
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    carrierA.connect(gain);
    carrierB.connect(gain);
    gain.connect(ctx.destination);
    carrierA.start();
    carrierB.start();
    alarmVoice = { ctx, carrierA, carrierB, gain, phase: 0 };
    pulseAlarmStep();
    alarmTimer = window.setInterval(pulseAlarmStep, 200);
  }

  function stopAlarmSound() {
    if (alarmTimer) {
      clearInterval(alarmTimer);
      alarmTimer = null;
    }
    if (!alarmVoice) return;
    const { ctx, carrierA, carrierB, gain } = alarmVoice;
    const now = ctx.currentTime;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setValueAtTime(gain.gain.value || 0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.04);
    try {
      carrierA.stop(now + 0.05);
      carrierB.stop(now + 0.05);
    } catch (error) {
      console.warn("Plithos alarm stop warning:", error.message);
    }
    alarmVoice = null;
  }

  function addCacheBuster(url) {
    if (!url) return "";
    const joiner = url.includes("?") ? "&" : "?";
    return `${url}${joiner}t=${Date.now()}`;
  }

  function fmtNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString("en-US") : "--";
  }

  function fmtFps(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number.toFixed(number >= 10 ? 1 : 2) : "--";
  }

  function fmtUptime(totalSeconds) {
    const seconds = Math.max(0, Number(totalSeconds) || 0);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function fmtTime(value) {
    if (!value) return "--";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString("en-GB", { hour12: false });
  }

  function fmtDateTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    const hh = String(date.getHours()).padStart(2, "0");
    const mi = String(date.getMinutes()).padStart(2, "0");
    const ss = String(date.getSeconds()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
  }

  function csvCell(value) {
    const text = value == null ? "" : String(value);
    return `"${text.replace(/"/g, '""')}"`;
  }

  function titleCase(value) {
    return String(value || "")
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function hydrateFooter() {
    const footerCopy = $(".footer span");
    if (!footerCopy) return;
    footerCopy.innerHTML = `&copy; ${new Date().getFullYear()} Plithos. All rights reserved.`;
  }

  async function fetchState() {
    const response = await fetch(API_STATE, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  function highestAlertCamera(cameras) {
    const priorities = { violence: 3, fire: 2, safety: 1, "": 0 };
    return [...cameras].sort((a, b) => priorities[b.alert_type || ""] - priorities[a.alert_type || ""])[0];
  }

  function setCameraFeed(cameraId) {
    currentCameraId = cameraId;
    const feed = $("#dashboardFeed");
    if (feed) feed.src = addCacheBuster(`${window.location.origin}/video_feed/${cameraId}`);
    $$(".camera-tab").forEach((button) => {
      button.classList.toggle("is-active", Number(button.dataset.cameraId) === cameraId);
    });
    const next = new URL(window.location.href);
    next.searchParams.set("camera", cameraId);
    window.history.replaceState({}, "", next.toString());
  }

  function refreshSnapshotImages(root = document) {
    $$("[data-snapshot-url]", root).forEach((image) => {
      const nextSrc = addCacheBuster(image.dataset.snapshotUrl);
      if (nextSrc) image.src = nextSrc;
    });
  }

  function startSnapshotLoop(root = document) {
    const images = $$("[data-snapshot-url]", root);
    if (!images.length) return;
    refreshSnapshotImages(root);
    window.setInterval(() => refreshSnapshotImages(root), SNAPSHOT_MS);
  }

  function renderAlertList(alerts) {
    const list = $("#alertList");
    if (!list) return;
    const filtered = alertFilter === "all" ? alerts : alerts.filter((alert) => alert.type === alertFilter);
    if (!filtered.length) {
      list.innerHTML = `<div class="empty-state">${alertFilter === "all" ? "No alerts yet." : "No alerts match this filter."}</div>`;
      return;
    }
    list.innerHTML = filtered.map((alert) => `
      <article class="alert-item" data-type="${alert.type}">
        <div class="alert-bar"></div>
        <div class="alert-meta">
          <strong class="alert-title">${titleCase(alert.type)}</strong>
          <span class="alert-copy">${alert.camera_name} | ${alert.detail}</span>
        </div>
        <span class="alert-time">${fmtTime(alert.time)}</span>
      </article>
    `).join("");
  }

  function exportAlertsCsv(alerts) {
    if (!alerts.length) return;
    const header = [
      "Started",
      "Ended",
      "Duration",
      "Camera",
      "Type",
      "Persons",
      "Footage"
    ].join(",") + "\n";
    const rows = alerts.map((alert) => [
      csvCell(fmtDateTime(alert.started_at || alert.time)),
      csvCell(alert.ended_at ? fmtDateTime(alert.ended_at) : ""),
      csvCell(alert.duration_label || ""),
      csvCell(alert.camera_name),
      csvCell(alert.type),
      csvCell(alert.persons || ""),
      csvCell(alert.evidence_url || "")
    ].join(",")).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `plithos-alerts-${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function renderDashboard(data) {
    latestAlerts = data.recent_alerts || [];
    const cameras = data.cameras || [];
    const modelSettings = data.settings?.models || {};
    const violenceEnabled = modelSettings.violence !== false;
    const safetyEnabled = modelSettings.safety !== false;
    const fireEnabled = modelSettings.fire !== false;
    if (!currentCameraId && cameras[0]) currentCameraId = cameras[0].camera_id;
    if (data.settings?.auto_switch_alerts && data.active_alert_camera_id && currentCameraId !== data.active_alert_camera_id) {
      setCameraFeed(data.active_alert_camera_id);
    }
    const activeCamera = cameras.find((camera) => camera.camera_id === currentCameraId) || cameras[0];
    if (!activeCamera) return;
    if (!$("#dashboardFeed").src) setCameraFeed(activeCamera.camera_id);

    const violence = Boolean(activeCamera.is_violent);
    const safetyPeople = activeCamera.ppe_people || [];
    const safetyActive = Boolean(activeCamera.is_safety_missing && safetyPeople.length);
    const fireActive = Boolean(activeCamera.is_fire || activeCamera.is_smoke);
    const fireLabel = activeCamera.is_fire && activeCamera.is_smoke ? "Fire + smoke" : activeCamera.is_fire ? "Fire" : activeCamera.is_smoke ? "Smoke" : "Clear";
    const violenceText = violence ? `${Math.round((activeCamera.v_conf || 0) * 100)}%` : "Clear";

    $("#feedLabel").textContent = activeCamera.camera_online ? "Live" : "Offline";
    $("#feedTime").textContent = activeCamera.video_timestamp || "--";

    $("#violenceValue").textContent = violenceEnabled ? (violence ? "Detected" : "Clear") : "Off";
    $("#violenceCopy").textContent = violenceEnabled ? (violence ? `${violenceText} confidence` : "No violence detected") : "Violence monitoring is off";
    $("#safetyValue").textContent = safetyEnabled ? (safetyActive ? "Missing" : "Clear") : "Off";
    $("#safetyCopy").textContent = safetyEnabled ? (safetyActive ? (activeCamera.safety_summary || "Safety gear missing") : "No missing safety equipment") : "Safety monitoring is off";
    $("#fireValue").textContent = fireEnabled ? fireLabel : "Off";
    $("#fireCopy").textContent = fireEnabled ? (fireActive ? "Attention needed" : "No fire or smoke detected") : "Fire monitoring is off";
    $("#personsValue").textContent = String(activeCamera.person_count || 0);
    $("#personsCopy").textContent = activeCamera.person_count ? "Live people count" : "No people counted";

    $("#framesValue").textContent = fmtNumber(activeCamera.frame);
    $("#alertsValue").textContent = fmtNumber(data.alerts_total);
    $("#fpsValue").textContent = fmtFps(activeCamera.fps);
    $("#uptimeValue").textContent = fmtUptime(activeCamera.uptime_sec);

    [
      ["#violenceCard", violenceEnabled && violence ? "alert-violence" : ""],
      ["#safetyCard", safetyEnabled && safetyActive ? "alert-safety" : ""],
      ["#fireCard", fireEnabled && fireActive ? "alert-fire" : ""]
    ].forEach(([selector, className]) => {
      const card = $(selector);
      if (!card) return;
      card.classList.remove("alert-violence", "alert-safety", "alert-fire");
      if (className) card.classList.add(className);
    });

    const focus = highestAlertCamera(cameras);
    const hasAlert = Boolean(focus && focus.active_alert);
    const banner = $("#alarmBanner");
    if (hasAlert) {
      banner.classList.add("alert");
      document.body.classList.add("alarm-visible");
      $("#alarmKick").textContent = "Active alert";
      $("#alarmTitle").textContent = `${focus.name} needs attention`;
      $("#alarmCopy").textContent = focus.alert_type === "violence"
        ? "Violence detected."
        : focus.alert_type === "fire"
        ? "Fire or smoke detected."
        : focus.alert_type === "safety"
        ? (focus.safety_summary || "Safety gear missing.")
        : "Attention needed.";
      startAlarmSound(Boolean(data.settings?.sound_enabled));
    } else {
      banner.classList.remove("alert");
      document.body.classList.remove("alarm-visible");
      $("#alarmKick").textContent = "Live monitoring";
      $("#alarmTitle").textContent = "No active alerts";
      $("#alarmCopy").textContent = "The dashboard is watching your selected camera.";
      alarmMuted = false;
      $("#muteAlarmBtn").textContent = "Mute alarm";
      stopAlarmSound();
    }

    renderAlertList(latestAlerts);
    const exportBtn = $("#exportAlertsBtn");
    if (exportBtn) {
      exportBtn.onclick = () => exportAlertsCsv(
        alertFilter === "all" ? latestAlerts : latestAlerts.filter((alert) => alert.type === alertFilter)
      );
    }
  }

  function renderCameras(data) {
    (data.cameras || []).forEach((camera) => {
      const status = $(`[data-camera-status="${camera.camera_id}"]`);
      const card = $(`[data-camera-card="${camera.camera_id}"]`);
      if (!status || !card) return;
      status.textContent = camera.active_alert ? titleCase(camera.alert_type) : camera.camera_online ? "Live" : "Offline";
      card.classList.toggle("alert-card", Boolean(camera.active_alert));
    });
  }

  function startPolling(handler) {
    let busy = false;
    const poll = async () => {
      if (busy) return;
      busy = true;
      try {
        const data = await fetchState();
        handler(data);
      } catch (error) {
        console.warn("Plithos polling error:", error.message);
      } finally {
        busy = false;
        window.setTimeout(poll, POLL_MS);
      }
    };
    poll();
  }

  function initDashboard() {
    const muteBtn = $("#muteAlarmBtn");
    muteBtn?.addEventListener("click", () => {
      alarmMuted = true;
      stopAlarmSound();
      muteBtn.textContent = "Alarm muted";
    });
    $("#alertFilters")?.addEventListener("click", (event) => {
      const button = event.target.closest("[data-filter]");
      if (!button) return;
      alertFilter = button.dataset.filter;
      $$("#alertFilters [data-filter]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderAlertList(latestAlerts);
    });
    $("#cameraTabs")?.addEventListener("click", (event) => {
      const button = event.target.closest("[data-camera-id]");
      if (!button) return;
      setCameraFeed(Number(button.dataset.cameraId));
    });
    document.addEventListener("pointerdown", ensureAudio, { once: true });
    startPolling(renderDashboard);
  }

  function initCameras() {
    startPolling(renderCameras);
  }

  function initSetupLikePages() {
    startSnapshotLoop();
  }

  function initVerify() {
    $('input[name="code"]')?.focus();
  }

  hydrateFooter();
  if (cfg.page === "dashboard") initDashboard();
  if (cfg.page === "cameras") initCameras();
  if (cfg.page === "setup" || cfg.page === "settings") initSetupLikePages();
  if (cfg.page === "verify") initVerify();
})();

