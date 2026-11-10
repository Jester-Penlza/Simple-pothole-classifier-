const IMAGE_SIZE = 16;
const HISTOGRAM_BINS = 12;

const elements = {
  modelStatus: document.querySelector("#model-status"),
  fileInput: document.querySelector("#file-input"),
  dropZone: document.querySelector("#drop-zone"),
  dropContent: document.querySelector("#drop-content"),
  previewImage: document.querySelector("#preview-image"),
  resetButton: document.querySelector("#reset-button"),
  emptyResult: document.querySelector("#empty-result"),
  prediction: document.querySelector("#prediction"),
  verdict: document.querySelector("#verdict"),
  fileName: document.querySelector("#file-name"),
  confidenceRing: document.querySelector("#confidence-ring"),
  confidenceValue: document.querySelector("#confidence-value"),
  potholeValue: document.querySelector("#pothole-value"),
  clearValue: document.querySelector("#clear-value"),
  potholeBar: document.querySelector("#pothole-bar"),
  clearBar: document.querySelector("#clear-bar"),
};

let runtimeModel = null;
let activeObjectUrl = null;

function setModelStatus(message, state = "") {
  elements.modelStatus.className = `model-status ${state}`.trim();
  elements.modelStatus.querySelector("span:last-child").textContent = message;
}

async function loadModel() {
  try {
    const response = await fetch("model_runtime.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
    runtimeModel = await response.json();
    const lengths = [
      runtimeModel.scaler_mean?.length,
      runtimeModel.scaler_scale?.length,
      runtimeModel.coefficients?.length,
    ];
    if (!lengths.every((length) => length === 810)) {
      throw new Error("The exported model has an unexpected feature count.");
    }
    setModelStatus("Model ready · runs locally", "ready");
  } catch (error) {
    console.error(error);
    setModelStatus("Model could not load", "error");
  }
}

function extractFeatures(image) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const data = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const channels = [[], [], []];
  const gray = [];

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index] / 255;
    const green = data[index + 1] / 255;
    const blue = data[index + 2] / 255;
    channels[0].push(red);
    channels[1].push(green);
    channels[2].push(blue);
    gray.push(0.299 * red + 0.587 * green + 0.114 * blue);
  }

  const horizontalEdges = [];
  const verticalEdges = [];
  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const index = y * IMAGE_SIZE + x;
      const right = x === IMAGE_SIZE - 1 ? index : index + 1;
      const below = y === IMAGE_SIZE - 1 ? index : index + IMAGE_SIZE;
      horizontalEdges.push(Math.abs(gray[right] - gray[index]));
      verticalEdges.push(Math.abs(gray[below] - gray[index]));
    }
  }

  const histograms = [];
  for (const values of channels) {
    const counts = Array(HISTOGRAM_BINS).fill(0);
    for (const value of values) {
      const bin = Math.min(HISTOGRAM_BINS - 1, Math.floor(value * HISTOGRAM_BINS));
      counts[bin] += 1;
    }
    histograms.push(...counts.map((count) => count / values.length));
  }

  const means = channels.map((values) => values.reduce((sum, value) => sum + value, 0) / values.length);
  const standardDeviations = channels.map((values, channel) => {
    const variance = values.reduce((sum, value) => sum + (value - means[channel]) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  });

  return [...gray, ...horizontalEdges, ...verticalEdges, ...histograms, ...means, ...standardDeviations];
}

function predict(features) {
  if (!runtimeModel) throw new Error("The model is not ready yet.");
  const standardized = features.map(
    (value, index) => (value - runtimeModel.scaler_mean[index]) / runtimeModel.scaler_scale[index],
  );
  const logit = runtimeModel.intercept + standardized.reduce(
    (sum, value, index) => sum + runtimeModel.coefficients[index] * value,
    0,
  );
  const potholeProbability = logit >= 0
    ? 1 / (1 + Math.exp(-logit))
    : Math.exp(logit) / (1 + Math.exp(logit));
  return {
    label: potholeProbability >= 0.5 ? "Pothole" : "No Pothole",
    potholeProbability,
  };
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function showPrediction(result, name) {
  const clearProbability = 1 - result.potholeProbability;
  const confidence = Math.max(result.potholeProbability, clearProbability);
  const isPothole = result.label === "Pothole";

  elements.emptyResult.hidden = true;
  elements.prediction.hidden = false;
  elements.verdict.textContent = result.label;
  elements.verdict.style.color = isPothole ? "var(--accent)" : "var(--green)";
  elements.fileName.textContent = name;
  elements.confidenceValue.textContent = percent(confidence);
  elements.confidenceRing.style.setProperty("--value", (confidence * 100).toFixed(1));
  elements.confidenceRing.style.background = `conic-gradient(${isPothole ? "var(--accent)" : "var(--green)"} calc(var(--value) * 1%), #303431 0)`;
  elements.potholeValue.textContent = percent(result.potholeProbability);
  elements.clearValue.textContent = percent(clearProbability);
  requestAnimationFrame(() => {
    elements.potholeBar.style.width = percent(result.potholeProbability);
    elements.clearBar.style.width = percent(clearProbability);
  });
}

function loadImage(source, name, shouldRevoke = false) {
  elements.dropZone.classList.add("processing");
  const image = new Image();
  image.decoding = "async";
  image.onload = () => {
    try {
      elements.previewImage.src = source;
      elements.previewImage.hidden = false;
      elements.dropContent.hidden = true;
      elements.dropZone.classList.add("has-image");
      elements.resetButton.hidden = false;
      showPrediction(predict(extractFeatures(image)), name);
    } catch (error) {
      console.error(error);
      window.alert(error.message);
    } finally {
      elements.dropZone.classList.remove("processing");
    }
  };
  image.onerror = () => {
    elements.dropZone.classList.remove("processing");
    window.alert("That image could not be read. Please choose another file.");
    if (shouldRevoke) URL.revokeObjectURL(source);
  };
  image.src = source;
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    window.alert("Please choose an image file.");
    return;
  }
  if (!runtimeModel) {
    window.alert("The model is still loading. Please try again in a moment.");
    return;
  }
  if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  activeObjectUrl = URL.createObjectURL(file);
  loadImage(activeObjectUrl, file.name, true);
}

function reset() {
  if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  activeObjectUrl = null;
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.previewImage.hidden = true;
  elements.dropContent.hidden = false;
  elements.dropZone.classList.remove("has-image", "processing");
  elements.resetButton.hidden = true;
  elements.emptyResult.hidden = false;
  elements.prediction.hidden = true;
  elements.potholeBar.style.width = "0";
  elements.clearBar.style.width = "0";
}

elements.fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
elements.resetButton.addEventListener("click", reset);

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!runtimeModel) return;
    reset();
    loadImage(button.dataset.sample, button.textContent.trim());
  });
});

loadModel().then(() => {
  const requestedSample = new URLSearchParams(window.location.search).get("sample");
  const samples = {
    pothole: "sample_inputs/sample_pothole.jpg",
    clear: "sample_inputs/sample_no_pothole.jpg",
  };
  if (runtimeModel && samples[requestedSample]) {
    loadImage(samples[requestedSample], `${requestedSample} sample`);
  }
});
