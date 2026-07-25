const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const uploadBtn = document.getElementById("upload-btn");
const uploadInput = document.getElementById("upload-input");
const cameraBtn = document.getElementById("camera-btn");
const cameraFallbackInput = document.getElementById("camera-fallback-input");
const cameraModal = document.getElementById("camera-modal");
const cameraVideo = document.getElementById("camera-video");
const cameraCaptureBtn = document.getElementById("camera-capture-btn");
const cameraCancelBtn = document.getElementById("camera-cancel-btn");
const topResult = document.getElementById("top-result");
const rankedList = document.getElementById("ranked-list");

const STROKE_WIDTH = 18;
let drawing = false;
let lastPoint = null;
let cameraStream = null;

function resetCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // faint sketchpad grid -- dim enough to stay below the server's foreground threshold
  ctx.strokeStyle = "rgba(76, 124, 240, 0.14)";
  ctx.lineWidth = 1;
  const step = canvas.width / 8;
  for (let i = 1; i < 8; i++) {
    ctx.beginPath();
    ctx.moveTo(i * step + 0.5, 0);
    ctx.lineTo(i * step + 0.5, canvas.height);
    ctx.moveTo(0, i * step + 0.5);
    ctx.lineTo(canvas.width, i * step + 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = "#fff";
  ctx.lineWidth = STROKE_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function showPlaceholder(message) {
  topResult.innerHTML = `<span class="placeholder">${message}</span>`;
  rankedList.innerHTML = "";
}

// --- freehand drawing ---
// Uses the Pointer Events API (unifies mouse/trackpad/touch/stylus) with
// pointer capture, so a fast drag that momentarily leaves the small canvas
// still keeps drawing instead of breaking into a native text-selection drag.

function pointFromEvent(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function startDraw(evt) {
  evt.preventDefault();
  drawing = true;
  canvas.setPointerCapture(evt.pointerId);
  lastPoint = pointFromEvent(evt);
}

function draw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const point = pointFromEvent(evt);
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
  lastPoint = point;
}

function endDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  drawing = false;
  lastPoint = null;
}

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", endDraw);
canvas.addEventListener("pointercancel", endDraw);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  showPlaceholder("Draw a letter and hit Predict");
});

// --- drawing an external image (upload or camera frame) onto the canvas ---

function drawSourceToCanvas(source, srcWidth, srcHeight) {
  const scale = Math.max(canvas.width / srcWidth, canvas.height / srcHeight);
  const w = srcWidth * scale;
  const h = srcHeight * scale;
  const x = (canvas.width - w) / 2;
  const y = (canvas.height - h) / 2;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, x, y, w, h);
}

function loadImageFile(file) {
  const img = new Image();
  img.onload = () => {
    drawSourceToCanvas(img, img.naturalWidth, img.naturalHeight);
    URL.revokeObjectURL(img.src);
    runPrediction();
  };
  img.src = URL.createObjectURL(file);
}

function handleFileInputChange(evt) {
  const file = evt.target.files[0];
  evt.target.value = ""; // allow re-selecting the same file later
  if (file) loadImageFile(file);
}

uploadBtn.addEventListener("click", () => uploadInput.click());
uploadInput.addEventListener("change", handleFileInputChange);
cameraFallbackInput.addEventListener("change", handleFileInputChange);

// --- camera capture ---

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraFallbackInput.click();
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraModal.classList.remove("hidden");
  } catch (err) {
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

cameraBtn.addEventListener("click", openCamera);
cameraCancelBtn.addEventListener("click", closeCamera);
cameraCaptureBtn.addEventListener("click", () => {
  drawSourceToCanvas(cameraVideo, cameraVideo.videoWidth, cameraVideo.videoHeight);
  closeCamera();
  runPrediction();
});

// --- prediction ---

predictBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";
  try {
    const dataUrl = canvas.toDataURL("image/png");
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    const body = await res.json();
    if (!res.ok) {
      throw new Error(body.error || "Prediction failed");
    }
    renderPredictions(body.predictions);
  } catch (err) {
    showPlaceholder(err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
}

function renderPredictions(predictions) {
  if (!predictions || predictions.length === 0) {
    showPlaceholder("No prediction returned");
    return;
  }

  const top = predictions[0];
  topResult.innerHTML = `
    <span class="glyph">${top.label}</span>
    <span class="conf">${(top.confidence * 100).toFixed(1)}% confidence</span>
  `;

  rankedList.innerHTML = predictions
    .map(
      (p) => `
      <li>
        <span class="rank-glyph">${p.label.split(" ")[0]}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(p.confidence * 100).toFixed(1)}%"></span></span>
        <span class="pct">${(p.confidence * 100).toFixed(1)}%</span>
      </li>`
    )
    .join("");
}

resetCanvas();
