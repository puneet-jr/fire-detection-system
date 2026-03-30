const form = document.getElementById("prediction-form");
const resetButton = document.getElementById("reset-engine");
const toggleStreamButton = document.getElementById("toggle-stream");
const presetButtons = Array.from(document.querySelectorAll(".preset-btn"));

const temperatureInput = document.getElementById("temperature");
const humidityInput = document.getElementById("humidity");
const flameInput = document.getElementById("flame");
const flameDisplay = document.getElementById("flame-display");

const streamModeTitle = document.getElementById("stream-mode-title");
const streamModeCopy = document.getElementById("stream-mode-copy");
const streamStatePill = document.getElementById("stream-state-pill");
const streamStepPill = document.getElementById("stream-step-pill");

const verdictCard = document.getElementById("verdict-card");
const verdictText = document.getElementById("verdict-text");
const verdictSummary = document.getElementById("verdict-summary");
const confidencePill = document.getElementById("confidence-pill");
const anomalyPill = document.getElementById("anomaly-pill");
const buzzerPill = document.getElementById("buzzer-pill");
const probabilityBreakdown = document.getElementById("probability-breakdown");
const ruleBreakdown = document.getElementById("rule-breakdown");
const reasonsList = document.getElementById("reasons-list");
const overrideList = document.getElementById("override-list");
const featureGrid = document.getElementById("feature-grid");
const historyBody = document.getElementById("history-body");
const historyCount = document.getElementById("history-count");

const analysisConfig = window.__INITIAL_ANALYSIS__ || {};
const liveRanges = Object.fromEntries(
  (analysisConfig.live_ranges || []).map((item) => [item.label, item])
);

const streamState = {
  running: true,
  timerId: null,
  readingCount: 0,
  phase: "NORMAL",
  history: [],
};

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

function formatNumber(value, digits = 4) {
  return Number.parseFloat(value).toFixed(digits);
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

function renderPrediction(result) {
  clearVerdictClasses();
  verdictCard.classList.add(verdictClass(result.prediction));
  verdictText.textContent = result.prediction;

  const classifierText = `${result.classifier.prediction} with ${result.classifier.confidence.toFixed(2)} confidence`;
  const anomalyText = result.anomaly_flag
    ? "Isolation Forest flagged this reading as abnormal."
    : "Isolation Forest sees this reading as close to learned normal behavior.";
  verdictSummary.textContent = `Generated state: ${streamState.phase}. The model ranked this reading as ${result.prediction}. Classifier chose ${classifierText}. ${anomalyText}`;

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
  makeListItems(
    overrideList,
    result.fusion.safety_overrides,
    "No override was needed, so the final result stayed the same as the classifier output."
  );
  renderFeatures(result);
}

function setLiveInputs(reading) {
  temperatureInput.value = formatNumber(reading.temperature, 1);
  humidityInput.value = formatNumber(reading.humidity, 1);
  flameInput.value = String(reading.flame);
  flameDisplay.value = reading.flame ? "1 - Flame detected" : "0 - No flame";
}

function renderHistory() {
  historyBody.innerHTML = "";

  if (streamState.history.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="7">No readings captured yet.</td>`;
    historyBody.appendChild(row);
    historyCount.textContent = "0 saved";
    return;
  }

  streamState.history.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${entry.index}</td>
      <td>${entry.generatedState}</td>
      <td>${formatNumber(entry.temperature, 1)} C</td>
      <td>${formatNumber(entry.humidity, 1)} %</td>
      <td>${entry.flame}</td>
      <td>${entry.prediction}</td>
      <td>${formatNumber(entry.confidence, 2)}</td>
    `;
    historyBody.appendChild(row);
  });

  historyCount.textContent = `${streamState.history.length} saved`;
}

function pushHistory(reading, result) {
  streamState.history.unshift({
    index: streamState.readingCount,
    generatedState: reading.phase,
    temperature: reading.temperature,
    humidity: reading.humidity,
    flame: reading.flame,
    prediction: result.prediction,
    confidence: result.confidence,
  });
  streamState.history = streamState.history.slice(0, 18);
  renderHistory();
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function chooseNextPhase() {
  const roll = Math.random();

  if (streamState.phase === "NORMAL") {
    if (roll < 0.14) {
      return "WARNING";
    }
    if (roll < 0.16) {
      return "FIRE";
    }
    return "NORMAL";
  }

  if (streamState.phase === "WARNING") {
    if (roll < 0.30) {
      return "FIRE";
    }
    if (roll < 0.42) {
      return "NORMAL";
    }
    return "WARNING";
  }

  if (roll < 0.25) {
    return "WARNING";
  }
  if (roll < 0.35) {
    return "NORMAL";
  }
  return "FIRE";
}

function generateLiveReading() {
  streamState.phase = chooseNextPhase();

  let temperature;
  let humidity;
  let flame;

  if (streamState.phase === "NORMAL") {
    temperature = randomBetween(27.0, 32.5);
    humidity = randomBetween(52.0, 66.0);
    flame = Math.random() < 0.02 ? 1 : 0;
  } else if (streamState.phase === "WARNING") {
    temperature = randomBetween(35.0, 43.0);
    humidity = randomBetween(28.0, 42.0);
    flame = Math.random() < 0.18 ? 1 : 0;
  } else {
    temperature = randomBetween(48.0, 66.0);
    humidity = randomBetween(10.0, 24.0);
    flame = Math.random() < 0.88 ? 1 : 0;
  }

  temperature = clamp(temperature + randomBetween(-0.7, 0.7), 20.0, 80.0);
  humidity = clamp(humidity + randomBetween(-1.5, 1.5), 5.0, 90.0);
  streamState.readingCount += 1;

  return {
    phase: streamState.phase,
    temperature,
    humidity,
    flame,
  };
}

async function analyzeReading(payload) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    verdictSummary.textContent = "Prediction failed. Please check the Flask server logs.";
    return null;
  }

  const result = await response.json();
  renderPrediction(result);
  return result;
}

function updateStreamStatus() {
  const phaseRange = liveRanges[streamState.phase];

  if (streamState.running) {
    streamModeTitle.textContent = "Sensor Feed Running";
    streamModeCopy.textContent = phaseRange
      ? `Current generated state: ${streamState.phase}. Temperature range ${phaseRange.temperature}, humidity range ${phaseRange.humidity}, flame behavior: ${phaseRange.flame}.`
      : "The system is automatically generating live sensor readings and checking for NORMAL, WARNING, or FIRE conditions.";
    toggleStreamButton.textContent = "Pause Auto Feed";
  } else {
    streamModeTitle.textContent = "Sensor Feed Paused";
    streamModeCopy.textContent = "Live generation is paused. You can resume the automatic feed or use an optional manual demo scenario.";
    toggleStreamButton.textContent = "Resume Auto Feed";
  }

  streamStatePill.textContent = `Condition: ${streamState.phase}`;
  streamStepPill.textContent = `Reading #${streamState.readingCount}`;
}

async function processLiveReading() {
  const reading = generateLiveReading();
  setLiveInputs(reading);
  updateStreamStatus();
  const result = await analyzeReading({
    temperature: Number.parseFloat(temperatureInput.value),
    humidity: Number.parseFloat(humidityInput.value),
    flame: Number.parseInt(flameInput.value, 10),
  });
  if (result) {
    pushHistory(reading, result);
  }
}

function startStream() {
  if (streamState.timerId) {
    window.clearInterval(streamState.timerId);
  }
  streamState.running = true;
  updateStreamStatus();
  streamState.timerId = window.setInterval(() => {
    void processLiveReading();
  }, 2200);
}

function stopStream() {
  streamState.running = false;
  if (streamState.timerId) {
    window.clearInterval(streamState.timerId);
    streamState.timerId = null;
  }
  updateStreamStatus();
}

async function resetEngine() {
  await fetch("/reset", { method: "POST" });
  streamState.readingCount = 0;
  streamState.phase = "NORMAL";
  streamState.history = [];
  renderHistory();
  updateStreamStatus();
  if (streamState.running) {
    await processLiveReading();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
});

toggleStreamButton.addEventListener("click", async () => {
  if (streamState.running) {
    stopStream();
  } else {
    startStream();
    await processLiveReading();
  }
});

resetButton.addEventListener("click", async () => {
  await resetEngine();
});

presetButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    stopStream();
    streamState.phase = button.textContent.toUpperCase().includes("FIRE")
      ? "FIRE"
      : button.textContent.toUpperCase().includes("WARNING")
        ? "WARNING"
        : "NORMAL";
    streamState.readingCount += 1;
    const reading = {
      phase: streamState.phase,
      temperature: Number.parseFloat(button.dataset.temp),
      humidity: Number.parseFloat(button.dataset.humidity),
      flame: Number.parseInt(button.dataset.flame, 10),
    };
    setLiveInputs(reading);
    updateStreamStatus();
    const result = await analyzeReading({
      temperature: Number.parseFloat(button.dataset.temp),
      humidity: Number.parseFloat(button.dataset.humidity),
      flame: Number.parseInt(button.dataset.flame, 10),
    });
    if (result) {
      pushHistory(reading, result);
    }
  });
});

window.addEventListener("load", async () => {
  renderHistory();
  updateStreamStatus();
  startStream();
  await resetEngine();
});
