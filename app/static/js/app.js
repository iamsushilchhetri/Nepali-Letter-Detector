const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const drawTab = document.getElementById("draw-tab");
const uploadTab = document.getElementById("upload-tab");
const drawPanel = document.getElementById("draw-panel");
const uploadPanel = document.getElementById("upload-panel");
const clearButton = document.getElementById("clear-btn");
const cameraButton = document.getElementById("camera-btn");
const predictButton = document.getElementById("predict-btn");
const uploadInput = document.getElementById("upload-input");
const uploadPreview = document.getElementById("upload-preview");
const uploadDropzone = document.getElementById("upload-dropzone");
const cameraFallbackInput = document.getElementById("camera-fallback-input");
const cameraModal = document.getElementById("camera-modal");
const cameraVideo = document.getElementById("camera-video");
const cameraCaptureButton = document.getElementById("camera-capture-btn");
const cameraCancelButton = document.getElementById("camera-cancel-btn");
const topResult = document.getElementById("top-result");
const rankedList = document.getElementById("ranked-list");
const resultStatus = document.getElementById("result-status");

const STROKE_WIDTH = 18;
let drawing = false;
let lastPoint = null;
let activeMode = "draw";
let cameraStream = null;

function setMode(mode) {
  activeMode = mode;
  const drawActive = mode === "draw";
  drawTab.classList.toggle("active", drawActive);
  uploadTab.classList.toggle("active", !drawActive);
  drawPanel.classList.toggle("hidden", !drawActive);
  uploadPanel.classList.toggle("hidden", drawActive);
}

function resetCanvas() {
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#1f1b18";
  context.lineWidth = STROKE_WIDTH;
  context.lineCap = "round";
  context.lineJoin = "round";
}

function clearUploadedPreview() {
  uploadPreview.src = "";
  uploadPreview.classList.add("hidden");
}

function clearWorkspace() {
  resetCanvas();
  clearUploadedPreview();
  showPlaceholder("Draw or upload a handwritten character to begin.");
}

function showPlaceholder(message) {
  topResult.innerHTML = `<p class="text-sm text-parchment/60">${message}</p>`;
  rankedList.innerHTML = "";
  resultStatus.textContent = "Waiting";
}

function showError(message) {
  topResult.innerHTML = `
    <div class="space-y-3">
      <p class="text-xs uppercase tracking-[0.2em] text-red-200/80">Problem</p>
      <p class="text-base font-medium text-red-100">${message}</p>
    </div>
  `;
  rankedList.innerHTML = "";
  resultStatus.textContent = "Error";
}

function canvasIsBlank() {
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let index = 0; index < pixels.length; index += 4) {
    const red = pixels[index];
    const green = pixels[index + 1];
    const blue = pixels[index + 2];
    if (red < 250 || green < 250 || blue < 250) {
      return false;
    }
  }
  return true;
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  };
}

function startDrawing(event) {
  if (activeMode !== "draw") return;
  event.preventDefault();
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  lastPoint = pointFromEvent(event);
}

function draw(event) {
  if (!drawing || activeMode !== "draw") return;
  event.preventDefault();
  const point = pointFromEvent(event);
  context.beginPath();
  context.moveTo(lastPoint.x, lastPoint.y);
  context.lineTo(point.x, point.y);
  context.stroke();
  lastPoint = point;
}

function stopDrawing(event) {
  if (!drawing) return;
  event.preventDefault();
  drawing = false;
  lastPoint = null;
}

function drawSourceToCanvas(source, width, height) {
  resetCanvas();
  const scale = Math.min(canvas.width / width, canvas.height / height);
  const drawWidth = width * scale;
  const drawHeight = height * scale;
  const x = (canvas.width - drawWidth) / 2;
  const y = (canvas.height - drawHeight) / 2;
  context.drawImage(source, x, y, drawWidth, drawHeight);
}

function renderUploadPreview(url) {
  uploadPreview.src = url;
  uploadPreview.classList.remove("hidden");
}

function handleImageFile(file) {
  const previewUrl = URL.createObjectURL(file);
  renderUploadPreview(previewUrl);

  const image = new Image();
  image.onload = () => {
    drawSourceToCanvas(image, image.naturalWidth, image.naturalHeight);
    URL.revokeObjectURL(previewUrl);
  };
  image.src = previewUrl;
}

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraFallbackInput.click();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    cameraVideo.srcObject = cameraStream;
    cameraModal.classList.remove("hidden");
  } catch (error) {
    cameraFallbackInput.click();
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraModal.classList.add("hidden");
}

function captureCameraFrame() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    return;
  }
  drawSourceToCanvas(cameraVideo, cameraVideo.videoWidth, cameraVideo.videoHeight);
  closeCamera();
  setMode("draw");
}

function renderPredictions(predictions) {
  if (!predictions || predictions.length === 0) {
    showPlaceholder("No predictions were returned.");
    return;
  }

  const top = predictions[0];
  topResult.innerHTML = `
    <div class="space-y-3">
      <p class="text-xs uppercase tracking-[0.2em] text-parchment/50">Top prediction</p>
      <div class="font-display text-5xl leading-none text-parchment md:text-6xl">${top.label}</div>
      <p class="text-sm text-parchment/70">${(top.confidence * 100).toFixed(1)}% confidence</p>
    </div>
  `;

  rankedList.innerHTML = predictions
    .map((prediction, index) => {
      const percentage = (prediction.confidence * 100).toFixed(1);
      const glyph = prediction.label.split(" ")[0];
      return `
        <li class="rounded-2xl border border-white/8 bg-white/5 p-4">
          <div class="mb-3 flex items-center justify-between gap-4">
            <div class="flex items-center gap-3">
              <span class="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/8 font-display text-xl text-parchment">${glyph}</span>
              <div>
                <p class="text-sm font-medium text-parchment">${prediction.label}</p>
                <p class="text-xs uppercase tracking-[0.18em] text-parchment/45">Rank ${index + 1}</p>
              </div>
            </div>
            <span class="text-sm font-semibold text-parchment">${percentage}%</span>
          </div>
          <div class="result-bar-track h-2">
            <span class="result-bar-fill" style="width:${percentage}%"></span>
          </div>
        </li>
      `;
    })
    .join("");

  resultStatus.textContent = "Updated";
}

async function runPrediction() {
  if (canvasIsBlank()) {
    showError("Please draw or upload a character before predicting.");
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";
  resultStatus.textContent = "Running";

  try {
    const dataUrl = canvas.toDataURL("image/png");
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || "Prediction failed.");
    }
    renderPredictions(body.predictions);
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict Character";
  }
}

drawTab.addEventListener("click", () => setMode("draw"));
uploadTab.addEventListener("click", () => setMode("upload"));

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

clearButton.addEventListener("click", clearWorkspace);
cameraButton.addEventListener("click", openCamera);
predictButton.addEventListener("click", runPrediction);
cameraCaptureButton.addEventListener("click", captureCameraFrame);
cameraCancelButton.addEventListener("click", closeCamera);

uploadInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  setMode("upload");
  handleImageFile(file);
});

cameraFallbackInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  setMode("upload");
  handleImageFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadDropzone.classList.add("border-saffron/60", "bg-white");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadDropzone.classList.remove("border-saffron/60", "bg-white");
  });
});

uploadDropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  setMode("upload");
  handleImageFile(file);
});

resetCanvas();
showPlaceholder("Draw or upload a handwritten character to begin.");
