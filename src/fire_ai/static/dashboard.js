const form = document.getElementById("prediction-form");
const resetButton = document.getElementById("reset-engine");
const datasetStartButton = document.getElementById("dataset-start");
const datasetPauseButton = document.getElementById("dataset-pause");
const datasetResumeButton = document.getElementById("dataset-resume");
const datasetStopButton = document.getElementById("dataset-stop");
const hardwareStartButton = document.getElementById("hardware-start");
const hardwarePauseButton = document.getElementById("hardware-pause");
const hardwareResumeButton = document.getElementById("hardware-resume");
const hardwareStopButton = document.getElementById("hardware-stop");
const presetButtons = Array.from(document.querySelectorAll(".preset-btn"));

const temperatureInput = document.getElementById("temperature");
const humidityInput = document.getElementById("humidity");
const flameInput = document.getElementById("flame");
const flameDisplay = document.getElementById("flame-display");

const streamModeTitle = document.getElementById("stream-mode-title");
const streamModeCopy = document.getElementById("stream-mode-copy");
const streamStatePill = document.getElementById("stream-state-pill");
const streamStepPill = document.getElementById("stream-step-pill");
const sourceName = document.getElementById("source-name");
const truthLabel = document.getElementById("truth-label");
const scenarioLabel = document.getElementById("scenario-label");
const hardwareLabel = document.getElementById("hardware-label");

const verdictCard = document.getElementById("verdict-card");
const verdictText = document.getElementById("verdict-text");
const verdictSummary = document.getElementById("verdict-summary");
const confidencePill = document.getElementById("confidence-pill");
const anomalyPill = document.getElementById("anomaly-pill");
const buzzerPill = document.getElementById("buzzer-pill");
const probabilityBreakdown = document.getElementById("probability-breakdown");
const ruleBreakdown = document.getElementById("rule-breakdown");
const reasonsList = document.getElementById("reasons-list");
const eventLog = document.getElementById("event-log");
const featureGrid = document.getElementById("feature-grid");
const datasetHistoryBody = document.getElementById("dataset-history-body");
const hardwareHistoryBody = document.getElementById("hardware-history-body");
const datasetHistoryCount = document.getElementById("dataset-history-count");
const hardwareHistoryCount = document.getElementById("hardware-history-count");
const telemetryChart = document.getElementById("telemetry-chart");

const state = {
  mode: "idle",
  datasetTimerId: null,
  hardwareTimerId: null,
  browserAudio: null,
  chartSeries: [],
  datasetHistory: [],
  hardwareHistory: [],
  datasetEvents: [],
  hardwareEvents: [],
  hardware: {
    connected: false,
    running: false,
    port: null,
    lastError: null,
    message: "Hardware monitor idle. Start Live Feed to connect to the Arduino.",
  },
  lastHardwareIndex: 0,
  lastAlertKey: null,
};

function isDatasetMode(mode = state.mode) {
  return mode === "dataset" || mode === "dataset-paused";
}

function isHardwareMode(mode = state.mode) {
  return mode === "hardware" || mode === "hardware-paused";
}

function verdictClass(label) {
  if (label === "FIRE") {
    return "verdict-fire";
  }
  if (label === "NORMAL") {
    return "verdict-normal";
  }
  return "verdict-warning";
}

function clearVerdictClasses() {
  verdictCard.classList.remove("verdict-fire", "verdict-warning", "verdict-normal");
}

function formatNumber(value, digits = 2) {
  return Number.parseFloat(value).toFixed(digits);
}

function setLiveInputs(reading) {
  temperatureInput.value = formatNumber(reading.temperature, 1);
  humidityInput.value = formatNumber(reading.humidity, 1);
  flameInput.value = String(reading.flame);
  flameDisplay.value = reading.flame ? "1 - Flame detected" : "0 - No flame";
}

function makeMeterItem(label, value, variantClass = "") {
  const item = document.createElement("div");
  item.className = "meter-item";
  item.innerHTML = `
    <div class="meter-title-row">
      <span>${label}</span>
      <span>${formatNumber(value * 100, 2)}%</span>
    </div>
    <div class="meter-track">
      <div class="meter-fill ${variantClass}" style="width: ${Math.max(0, Math.min(value * 100, 100))}%"></div>
    </div>
  `;
  return item;
}

function makeListItems(target, items, fallbackText) {
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = fallbackText;
    target.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function renderFeatures(result) {
  const featureEntries = [
    ["Temperature", `${formatNumber(result.analysis.current_temperature, 2)} C`],
    ["Humidity", `${formatNumber(result.analysis.current_humidity, 2)} %`],
    ["Rolling Temp", `${formatNumber(result.analysis.rolling_temperature, 2)} C`],
    ["Rolling Humidity", `${formatNumber(result.analysis.rolling_humidity, 2)} %`],
    ["Temp Z-Score", formatNumber(result.features.temp_zscore, 3)],
    ["Humidity Z-Score", formatNumber(result.features.humidity_zscore, 3)],
    ["Temp Rate", formatNumber(result.features.temp_rate, 3)],
    ["Humidity Rate", formatNumber(result.features.humidity_rate, 3)],
    ["Dryness Index", formatNumber(result.analysis.dryness_index, 3)],
    ["Flame Persistence", formatNumber(result.analysis.flame_persistence, 3)],
    ["Rule Score", formatNumber(result.rule_breakdown.score, 3)],
    ["Anomaly Score", formatNumber(result.anomaly_score, 3)],
  ];

  featureGrid.innerHTML = "";
  featureEntries.forEach(([name, value]) => {
    const tile = document.createElement("div");
    tile.className = "feature-tile";
    tile.innerHTML = `
      <span class="feature-name">${name}</span>
      <strong class="feature-value">${value}</strong>
    `;
    featureGrid.appendChild(tile);
  });
}

function renderPrediction(result, context = {}) {
  clearVerdictClasses();
  verdictCard.classList.add(verdictClass(result.prediction));
  verdictText.textContent = result.prediction;

  const sourceText = context.source === "hardware"
    ? `Live hardware reading${context.port ? ` on ${context.port}` : ""}`
    : "Dataset reading";
  const classifierText = `${result.classifier.prediction} with ${result.classifier.confidence.toFixed(2)} confidence`;
  verdictSummary.textContent = `${sourceText}. The model ranked this reading as ${result.prediction}. Classifier chose ${classifierText}.`;

  confidencePill.textContent = `Confidence ${formatNumber(result.confidence, 2)}`;
  anomalyPill.textContent = `Anomaly ${result.anomaly_flag ? "Yes" : "No"} (${formatNumber(result.anomaly_score, 3)})`;
  buzzerPill.textContent = `Buzzer ${result.buzzer ? "ON" : "OFF"}`;

  probabilityBreakdown.innerHTML = "";
  Object.entries(result.classifier.probabilities).forEach(([label, value]) => {
    probabilityBreakdown.appendChild(makeMeterItem(label, value));
  });

  ruleBreakdown.innerHTML = "";
  result.rule_breakdown.components.forEach((component) => {
    const item = makeMeterItem(component.name, component.contribution, "rule-meter");
    const description = document.createElement("div");
    description.className = "meter-title-row meter-description-row";
    description.innerHTML = `<span>${component.summary}</span><span>w=${component.weight}</span>`;
    item.appendChild(description);
    ruleBreakdown.appendChild(item);
  });

  makeListItems(reasonsList, result.reasons, "No critical indicators were triggered for this reading.");
  renderFeatures(result);
}

function updateChartSeries(reading) {
  state.chartSeries.push({
    temperature: Number(reading.temperature),
    humidity: Number(reading.humidity),
    flame: Number(reading.flame),
  });
  state.chartSeries = state.chartSeries.slice(-18);
  renderChart();
}

function renderChart() {
  const rows = state.chartSeries;
  if (!rows.length) {
    telemetryChart.innerHTML = "";
    return;
  }

  const width = 760;
  const height = 240;
  const padding = 24;
  const temps = rows.map((row) => row.temperature);
  const hums = rows.map((row) => row.humidity);
  const minValue = Math.min(...temps, ...hums);
  const maxValue = Math.max(...temps, ...hums);
  const valueSpan = Math.max(maxValue - minValue, 1);
  const xStep = rows.length > 1 ? (width - padding * 2) / (rows.length - 1) : 0;

  const toX = (index) => padding + index * xStep;
  const toY = (value) => height - padding - ((value - minValue) / valueSpan) * (height - padding * 2);

  const tempPoints = rows.map((row, index) => `${toX(index)},${toY(row.temperature)}`).join(" ");
  const humPoints = rows.map((row, index) => `${toX(index)},${toY(row.humidity)}`).join(" ");
  const flameMarkers = rows
    .map((row, index) => row.flame ? `<circle cx="${toX(index)}" cy="${height - 18}" r="5" class="chart-flame-dot"></circle>` : "")
    .join("");

  const yGrid = [0, 0.5, 1]
    .map((ratio) => {
      const y = padding + ratio * (height - padding * 2);
      return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="chart-grid-line"></line>`;
    })
    .join("");

  telemetryChart.innerHTML = `
    ${yGrid}
    <polyline points="${tempPoints}" class="chart-line chart-line-temp"></polyline>
    <polyline points="${humPoints}" class="chart-line chart-line-humidity"></polyline>
    ${flameMarkers}
  `;
}

function renderDatasetHistory() {
  datasetHistoryBody.innerHTML = "";
  if (!state.datasetHistory.length) {
    datasetHistoryBody.innerHTML = `<tr><td colspan="7">No dataset readings yet.</td></tr>`;
    datasetHistoryCount.textContent = "0 saved";
    return;
  }

  state.datasetHistory.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${entry.index}</td>
      <td>${entry.generatedState}</td>
      <td>${formatNumber(entry.temperature, 1)} C</td>
      <td>${formatNumber(entry.humidity, 1)} %</td>
      <td>${entry.flame}</td>
      <td>${entry.prediction}</td>
      <td>${entry.scenario}</td>
    `;
    datasetHistoryBody.appendChild(row);
  });
  datasetHistoryCount.textContent = `${state.datasetHistory.length} saved`;
}

function renderHardwareHistory() {
  hardwareHistoryBody.innerHTML = "";
  if (!state.hardwareHistory.length) {
    hardwareHistoryBody.innerHTML = `<tr><td colspan="7">No hardware readings yet.</td></tr>`;
    hardwareHistoryCount.textContent = "0 saved";
    return;
  }

  state.hardwareHistory.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${entry.index}</td>
      <td>${entry.generatedState}</td>
      <td>${formatNumber(entry.temperature, 1)} C</td>
      <td>${formatNumber(entry.humidity, 1)} %</td>
      <td>${entry.flame}</td>
      <td>${entry.prediction}</td>
      <td>${entry.buzzer ? "ON" : "OFF"}</td>
    `;
    hardwareHistoryBody.appendChild(row);
  });
  hardwareHistoryCount.textContent = `${state.hardwareHistory.length} saved`;
}

function renderEventLog() {
  const activeEvents = isHardwareMode() ? state.hardwareEvents : state.datasetEvents;
  eventLog.innerHTML = "";
  if (!activeEvents.length) {
    const li = document.createElement("li");
    li.textContent = "No fire or buzzer events logged for the current mode yet.";
    eventLog.appendChild(li);
    return;
  }

  activeEvents.slice(0, 10).forEach((entry) => {
    const li = document.createElement("li");
    if (entry.source === "hardware") {
      li.textContent = `Hardware #${entry.index}: flame=${entry.flame}, prediction=${entry.prediction}, buzzer=${entry.buzzer ? "ON" : "OFF"}.`;
    } else {
      li.textContent = `Dataset #${entry.index}: truth=${entry.truth}, prediction=${entry.prediction}, flame=${entry.flame}, buzzer=${entry.buzzer ? "ON" : "OFF"}.`;
    }
    eventLog.appendChild(li);
  });
}

function currentReadingCount() {
  if (isDatasetMode()) {
    return state.datasetHistory.length ? state.datasetHistory[0].index : 0;
  }
  if (isHardwareMode()) {
    return state.hardwareHistory.length ? state.hardwareHistory[0].index : 0;
  }
  return 0;
}

function updateStatusCard() {
  if (state.mode === "dataset") {
    streamModeTitle.textContent = "Dataset Playback Running";
    streamModeCopy.textContent = "Dataset playback is running. Real rows are being fed into the same feature builder and ML decision pipeline.";
    streamStatePill.textContent = "Mode: DATASET";
  } else if (state.mode === "dataset-paused") {
    streamModeTitle.textContent = "Dataset Playback Paused";
    streamModeCopy.textContent = "Dataset playback is paused. Use Resume Dataset to continue from the next row.";
    streamStatePill.textContent = "Mode: DATASET PAUSED";
  } else if (state.mode === "hardware") {
    streamModeTitle.textContent = "Live Hardware Running";
    streamModeCopy.textContent = state.hardware.connected
      ? `Live hardware capture is running${state.hardware.port ? ` on ${state.hardware.port}` : ""}. External flame and environmental changes are being logged in real time.`
      : state.hardware.lastError || state.hardware.message || "Waiting for Arduino Uno serial feed.";
    streamStatePill.textContent = "Mode: LIVE";
  } else if (state.mode === "hardware-paused") {
    streamModeTitle.textContent = "Live Hardware Paused";
    streamModeCopy.textContent = "Live hardware updates are paused. Use Resume Live Feed to continue reading the Arduino values.";
    streamStatePill.textContent = "Mode: LIVE PAUSED";
  } else {
    streamModeTitle.textContent = "Idle";
    streamModeCopy.textContent = "Choose either dataset playback or live hardware capture. Only one mode runs at a time.";
    streamStatePill.textContent = "Mode: IDLE";
  }

  streamStepPill.textContent = `Reading #${currentReadingCount()}`;
  sourceName.textContent = isHardwareMode() ? "Live Hardware" : isDatasetMode() ? "Dataset Playback" : "Idle";
  hardwareLabel.textContent = state.hardware.connected
    ? `Connected ${state.hardware.port || ""}`.trim()
    : state.hardware.running
      ? "Connecting"
      : "Idle";
}

function primeAudio() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    return;
  }
  if (!state.browserAudio) {
    state.browserAudio = new AudioContextClass();
  }
  if (state.browserAudio.state === "suspended") {
    void state.browserAudio.resume();
  }
}

function playDatasetAlert(entryKey) {
  if (state.lastAlertKey === entryKey) {
    return;
  }
  state.lastAlertKey = entryKey;

  primeAudio();
  if (!state.browserAudio) {
    return;
  }

  const startTime = state.browserAudio.currentTime;
  const durations = [0.00, 0.18];
  durations.forEach((offset) => {
    const oscillator = state.browserAudio.createOscillator();
    const gain = state.browserAudio.createGain();
    oscillator.type = "square";
    oscillator.frequency.value = 880;
    gain.gain.value = 0.0001;
    oscillator.connect(gain);
    gain.connect(state.browserAudio.destination);
    gain.gain.exponentialRampToValueAtTime(0.08, startTime + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + offset + 0.14);
    oscillator.start(startTime + offset);
    oscillator.stop(startTime + offset + 0.15);
  });
}

function stopDatasetTimer() {
  if (state.datasetTimerId) {
    window.clearInterval(state.datasetTimerId);
    state.datasetTimerId = null;
  }
}

function stopHardwareTimer() {
  if (state.hardwareTimerId) {
    window.clearInterval(state.hardwareTimerId);
    state.hardwareTimerId = null;
  }
}

function stopAllTimers() {
  stopDatasetTimer();
  stopHardwareTimer();
}

function clearChart() {
  state.chartSeries = [];
  renderChart();
}

async function fetchDatasetNext() {
  try {
    const response = await fetch("/dataset/next", { method: "POST" });
    if (!response.ok) {
      return;
    }

    const packet = await response.json();
    const reading = packet.reading;
    setLiveInputs(reading);
    truthLabel.textContent = packet.dataset.label;
    scenarioLabel.textContent = packet.dataset.scenario;
    renderPrediction(packet.result, { source: "dataset" });

    state.datasetHistory = [
      {
        index: packet.index,
        generatedState: packet.dataset.label,
        temperature: packet.reading.temperature,
        humidity: packet.reading.humidity,
        flame: packet.reading.flame,
        prediction: packet.result.prediction,
        confidence: packet.result.confidence,
        scenario: packet.dataset.scenario,
      },
      ...state.datasetHistory,
    ].slice(0, 18);
    renderDatasetHistory();

    if (packet.reading.flame === 1 || packet.result.buzzer) {
      state.datasetEvents = [
        {
          source: "dataset",
          index: packet.index,
          truth: packet.dataset.label,
          prediction: packet.result.prediction,
          flame: packet.reading.flame,
          buzzer: packet.result.buzzer,
        },
        ...state.datasetEvents,
      ].slice(0, 18);
      playDatasetAlert(`dataset-${packet.index}`);
    }

    renderEventLog();
    updateChartSeries(reading);
    updateStatusCard();
  } catch (error) {
    streamModeCopy.textContent = "Dataset playback hit a loading error. Reset and try again.";
  }
}

async function refreshHardwareStatus(connectIfNeeded = false) {
  try {
    const response = await fetch(connectIfNeeded ? "/hardware/connect" : "/hardware/status", {
      method: connectIfNeeded ? "POST" : "GET",
      headers: connectIfNeeded ? { "Content-Type": "application/json" } : undefined,
      body: connectIfNeeded ? JSON.stringify({}) : undefined,
    });
    if (!response.ok) {
      return;
    }

    const status = await response.json();
    state.hardware = {
      connected: Boolean(status.connected),
      running: Boolean(status.running),
      port: status.port || null,
      lastError: status.last_error || null,
      message: status.message || "Hardware monitor idle. Start Live Feed to connect to the Arduino.",
    };

    state.hardwareHistory = status.history || [];
    state.hardwareEvents = state.hardwareHistory
      .filter((entry) => entry.flame === 1 || entry.buzzer)
      .map((entry) => ({
        source: "hardware",
        index: entry.index,
        prediction: entry.prediction,
        flame: entry.flame,
        buzzer: Boolean(entry.buzzer),
      }))
      .slice(0, 18);
    renderHardwareHistory();

    if (state.mode === "hardware" || state.mode === "hardware-paused") {
      if (status.last_reading) {
        setLiveInputs({
          temperature: status.last_reading.temperature,
          humidity: status.last_reading.humidity,
          flame: status.last_reading.flame,
        });
        truthLabel.textContent = "LIVE";
        scenarioLabel.textContent = status.port || "Arduino Uno";
        const newestIndex = state.hardwareHistory.length ? state.hardwareHistory[0].index : 0;
        if (state.mode === "hardware" && newestIndex && newestIndex !== state.lastHardwareIndex) {
          state.lastHardwareIndex = newestIndex;
          updateChartSeries({
            temperature: status.last_reading.temperature,
            humidity: status.last_reading.humidity,
            flame: status.last_reading.flame,
          });
        }
      } else {
        truthLabel.textContent = "--";
        scenarioLabel.textContent = status.port || "--";
      }

      if (status.last_result) {
        renderPrediction(status.last_result, {
          source: "hardware",
          port: status.port,
        });
      }
    }

    renderEventLog();
    updateStatusCard();
  } catch (error) {
    state.hardware = {
      connected: false,
      running: false,
      port: null,
      lastError: "Unable to reach the hardware status endpoint.",
      message: "Live feed could not load hardware status from the server.",
    };
    updateStatusCard();
  }
}

async function startDatasetMode() {
  stopAllTimers();
  await fetch("/dataset/reset", { method: "POST" });
  state.mode = "dataset";
  state.lastAlertKey = null;
  clearChart();
  primeAudio();
  truthLabel.textContent = "--";
  scenarioLabel.textContent = "--";
  await refreshHardwareStatus(false);
  updateStatusCard();
  void fetchDatasetNext();
  state.datasetTimerId = window.setInterval(() => {
    void fetchDatasetNext();
  }, 2400);
}

function pauseDatasetMode() {
  if (state.mode !== "dataset") {
    return;
  }
  stopDatasetTimer();
  state.mode = "dataset-paused";
  updateStatusCard();
  renderEventLog();
}

async function resumeDatasetMode() {
  if (!isDatasetMode()) {
    return;
  }
  stopAllTimers();
  state.mode = "dataset";
  primeAudio();
  await refreshHardwareStatus(false);
  updateStatusCard();
  state.datasetTimerId = window.setInterval(() => {
    void fetchDatasetNext();
  }, 2400);
}

function stopDatasetMode() {
  if (!isDatasetMode()) {
    return;
  }
  stopDatasetTimer();
  state.mode = "idle";
  updateStatusCard();
  renderEventLog();
}

async function startHardwareMode() {
  stopAllTimers();
  state.mode = "hardware";
  state.lastHardwareIndex = state.hardwareHistory.length ? state.hardwareHistory[0].index : 0;
  clearChart();
  updateStatusCard();
  await refreshHardwareStatus(true);
  state.hardwareTimerId = window.setInterval(() => {
    void refreshHardwareStatus(false);
  }, 1000);
}

async function pauseHardwareMode() {
  if (state.mode !== "hardware") {
    return;
  }
  stopHardwareTimer();
  state.mode = "hardware-paused";
  await refreshHardwareStatus(false);
  updateStatusCard();
  renderEventLog();
}

async function resumeHardwareMode() {
  if (!isHardwareMode()) {
    return;
  }
  stopAllTimers();
  state.mode = "hardware";
  state.lastHardwareIndex = state.hardwareHistory.length ? state.hardwareHistory[0].index : 0;
  updateStatusCard();
  await refreshHardwareStatus(!state.hardware.running);
  state.hardwareTimerId = window.setInterval(() => {
    void refreshHardwareStatus(false);
  }, 1000);
}

async function stopHardwareMode() {
  if (!isHardwareMode()) {
    return;
  }
  stopHardwareTimer();
  await fetch("/hardware/disconnect", { method: "POST" });
  state.mode = "idle";
  await refreshHardwareStatus(false);
  updateStatusCard();
  renderEventLog();
}

async function runManualPrediction(button) {
  const payload = {
    temperature: Number.parseFloat(button.dataset.temp),
    humidity: Number.parseFloat(button.dataset.humidity),
    flame: Number.parseInt(button.dataset.flame, 10),
  };
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    return;
  }

  const result = await response.json();
  setLiveInputs(payload);
  truthLabel.textContent = "MANUAL";
  scenarioLabel.textContent = "Demo sample";
  renderPrediction(result, { source: "dataset" });
  updateChartSeries(payload);
  if (payload.flame === 1 || result.buzzer) {
    playDatasetAlert(`manual-${Date.now()}`);
  }
  updateStatusCard();
}

async function resetAll() {
  stopAllTimers();
  await fetch("/reset", { method: "POST" });
  state.mode = "idle";
  state.chartSeries = [];
  state.datasetHistory = [];
  state.hardwareHistory = [];
  state.datasetEvents = [];
  state.hardwareEvents = [];
  state.lastHardwareIndex = 0;
  state.lastAlertKey = null;
  truthLabel.textContent = "--";
  scenarioLabel.textContent = "--";
  setLiveInputs({ temperature: 30.0, humidity: 58.0, flame: 0 });
  renderDatasetHistory();
  renderHardwareHistory();
  renderEventLog();
  renderChart();
  updateStatusCard();
  await refreshHardwareStatus(false);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
});

datasetStartButton.addEventListener("click", () => {
  void startDatasetMode();
});

datasetPauseButton.addEventListener("click", () => {
  pauseDatasetMode();
});

datasetResumeButton.addEventListener("click", () => {
  void resumeDatasetMode();
});

datasetStopButton.addEventListener("click", () => {
  stopDatasetMode();
});

hardwareStartButton.addEventListener("click", () => {
  void startHardwareMode();
});

hardwarePauseButton.addEventListener("click", () => {
  void pauseHardwareMode();
});

hardwareResumeButton.addEventListener("click", () => {
  void resumeHardwareMode();
});

hardwareStopButton.addEventListener("click", () => {
  void stopHardwareMode();
});

resetButton.addEventListener("click", () => {
  void resetAll();
});

presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    stopAllTimers();
    void runManualPrediction(button);
  });
});

window.addEventListener("load", async () => {
  renderDatasetHistory();
  renderHardwareHistory();
  renderEventLog();
  renderChart();
  updateStatusCard();
  await fetch("/dataset/reset", { method: "POST" });
  await refreshHardwareStatus(false);
});
