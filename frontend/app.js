/**
 * TowerVision - Frontend Application Controller
 * Handles image upload, telemetry preview, execution sequence,
 * and rendering FastAPI backend inspection results.
 */

// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewWrap = document.getElementById("previewWrap");
const imagePreview = document.getElementById("imagePreview");
const fileInput = document.getElementById("fileInput");

const sampleSelect = document.getElementById("sampleSelect");
const btnLoadSample = document.getElementById("btnLoadSample");

const metadataBox = document.getElementById("metadataBox");
const metaFilename = document.getElementById("metaFilename");
const metaDimensions = document.getElementById("metaDimensions");
const metaSize = document.getElementById("metaSize");
const metaStatus = document.getElementById("metaStatus");

const confThreshold = document.getElementById("confThreshold");
const confVal = document.getElementById("confVal");
const blurThreshold = document.getElementById("blurThreshold");
const blurVal = document.getElementById("blurVal");

const btnExecute = document.getElementById("btnExecute");
const btnReset = document.getElementById("btnReset");

// Sequence Steps
const stepUpload = document.getElementById("stepUpload");
const stepQC = document.getElementById("stepQC");
const stepDetection = document.getElementById("stepDetection");
const stepResults = document.getElementById("stepResults");
const stepComplete = document.getElementById("stepComplete");

// Results Elements
const resultsEmptyState = document.getElementById("resultsEmptyState");
const diagnosticCard = document.getElementById("diagnosticCard");
const qcStatusBadge = document.getElementById("qcStatusBadge");
const qcSharpnessVal = document.getElementById("qcSharpnessVal");
const qcSharpnessThreshold = document.getElementById("qcSharpnessThreshold");
const qcBrightnessVal = document.getElementById("qcBrightnessVal");
const qcExposureVal = document.getElementById("qcExposureVal");
const rejectionMsgBox = document.getElementById("rejectionMsgBox");
const rejectionReasonText = document.getElementById("rejectionReasonText");

const detectionOutputWrap = document.getElementById("detectionOutputWrap");
const monoCountBadge = document.getElementById("monoCountBadge");
const monoConfidence = document.getElementById("monoConfidence");
const suppCountBadge = document.getElementById("suppCountBadge");
const suppConfidence = document.getElementById("suppConfidence");
const totalDetectionsVal = document.getElementById("totalDetectionsVal");
const overallConfidenceVal = document.getElementById("overallConfidenceVal");
const annotatedImage = document.getElementById("annotatedImage");
const btnDownload = document.getElementById("btnDownload");
const telemetryTableBody = document.getElementById("telemetryTableBody");

const systemStatusPill = document.getElementById("systemStatusPill");
const systemStatusText = document.getElementById("systemStatusText");

// Application State
let activeFile = null;
let annotatedDataUrl = null;

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  checkSystemStatus();
  fetchSampleList();
});

// Event Listeners
function initEventListeners() {
  // Dropzone click
  dropzone.addEventListener("click", () => fileInput.click());

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and Drop
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Threshold slider live labels
  confThreshold.addEventListener("input", (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  blurThreshold.addEventListener("input", (e) => {
    blurVal.textContent = parseFloat(e.target.value).toFixed(1);
    qcSharpnessThreshold.textContent = `Target: ≥ ${e.target.value}`;
  });

  // Sample selector
  sampleSelect.addEventListener("change", () => {
    btnLoadSample.disabled = !sampleSelect.value;
  });

  btnLoadSample.addEventListener("click", () => {
    const filename = sampleSelect.value;
    if (filename) {
      loadSampleImage(filename);
    }
  });

  // Action Buttons
  btnExecute.addEventListener("click", runInspection);
  btnReset.addEventListener("click", resetDashboard);

  // Download Button
  btnDownload.addEventListener("click", downloadAnnotatedImage);
}

// System Health Check
async function checkSystemStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/status`);
    if (res.ok) {
      const data = await res.json();
      systemStatusText.textContent = "AI ENGINE ONLINE";
      systemStatusPill.style.borderColor = "rgba(34, 197, 94, 0.4)";
    } else {
      setSystemDegraded("API NOT READY");
    }
  } catch (err) {
    setSystemDegraded("SERVER OFFLINE");
  }
}

function setSystemDegraded(label) {
  systemStatusText.textContent = label;
  systemStatusPill.style.borderColor = "rgba(239, 68, 68, 0.4)";
  systemStatusPill.style.color = "#F87171";
  const dot = systemStatusPill.querySelector(".status-dot");
  if (dot) dot.style.backgroundColor = "#EF4444";
}

// Fetch Sample Dataset Files
async function fetchSampleList() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/samples`);
    if (res.ok) {
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        sampleSelect.innerHTML = `<option value="">-- Choose from Test Dataset (${data.samples.length} items) --</option>`;
        data.samples.forEach((sample, i) => {
          const opt = document.createElement("option");
          opt.value = sample.filename;
          opt.textContent = `${i + 1}. ${sample.filename}`;
          // Default select 3rd sample (supporting tower ~96% conf)
          if (i === 2) opt.selected = true;
          sampleSelect.appendChild(opt);
        });
        btnLoadSample.disabled = false;
      }
    }
  } catch (err) {
    console.warn("Could not load samples list:", err);
  }
}

// Load Sample Image File
async function loadSampleImage(filename) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/samples/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error("Failed to fetch sample image.");
    const blob = await res.blob();
    const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
    handleFileSelected(file);
  } catch (err) {
    alert(`Error loading sample: ${err.message}`);
  }
}

// Ingest & Preview Selected File
function handleFileSelected(file) {
  const validExtensions = ["image/jpeg", "image/jpg", "image/png"];
  if (!validExtensions.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
    alert("Unsupported file format! Please upload a JPG or PNG tower image.");
    return;
  }

  activeFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzoneEmpty.classList.add("hidden");
    previewWrap.classList.remove("hidden");

    // Measure dimensions
    const imgObj = new Image();
    imgObj.onload = () => {
      metaDimensions.textContent = `${imgObj.naturalWidth} × ${imgObj.naturalHeight} px`;
    };
    imgObj.src = e.target.result;

    metaFilename.textContent = file.name;
    metaSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    metaStatus.textContent = "Loaded & Ready";

    metadataBox.classList.remove("hidden");
    btnExecute.disabled = false;

    // Reset results area
    resetResultsArea();
    updateSequence(1);
  };
  reader.readAsDataURL(file);
}

// Sequence Timeline Updater
function updateSequence(stepNum) {
  const steps = [stepUpload, stepQC, stepDetection, stepResults, stepComplete];
  steps.forEach((step, idx) => {
    const num = idx + 1;
    const icon = step.querySelector(".seq-icon");
    step.classList.remove("active", "completed");

    if (num < stepNum) {
      step.classList.add("completed");
      if (icon) icon.textContent = "✓";
    } else if (num === stepNum) {
      step.classList.add("active");
      if (icon) icon.textContent = "●";
    } else {
      if (icon) icon.textContent = "○";
    }
  });
}

// Run Inspection Pipeline via FastAPI
async function runInspection() {
  if (!activeFile) {
    alert("Please select or upload a tower image first.");
    return;
  }

  btnExecute.disabled = true;
  btnReset.disabled = true;
  metaStatus.textContent = "Analyzing image...";
  resultsEmptyState.classList.add("hidden");

  // Step 1: Uploading & preparing
  updateSequence(2);

  const formData = new FormData();
  formData.append("file", activeFile);

  const queryParams = new URLSearchParams({
    conf_threshold: confThreshold.value,
    blur_threshold: blurThreshold.value
  });

  try {
    // Step 2: Quality Check & YOLO Detection
    updateSequence(3);

    const res = await fetch(`${API_BASE_URL}/predict?${queryParams.toString()}`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server responded with HTTP ${res.status}`);
    }

    const data = await res.json();
    renderInspectionResults(data);

  } catch (err) {
    console.error("Inspection error:", err);
    metaStatus.textContent = "Error occurred";
    alert(`Inspection failed: ${err.message}\n\nPlease verify that the FastAPI backend is running.`);
    resetResultsArea();
  } finally {
    btnExecute.disabled = false;
    btnReset.disabled = false;
  }
}

// Render Results Received from Backend
function renderInspectionResults(data) {
  diagnosticCard.classList.remove("hidden");

  // Diagnostic Quality Check
  const qc = data.quality;
  qcSharpnessVal.textContent = qc.sharpness.toFixed(1);
  qcBrightnessVal.textContent = qc.brightness.toFixed(1);
  qcExposureVal.textContent = qc.exposure;

  if (data.status === "ACCEPTED") {
    // Quality Check: ACCEPTED
    qcStatusBadge.textContent = "● ACCEPTED";
    qcStatusBadge.className = "diag-badge status-accepted";
    rejectionMsgBox.classList.add("hidden");

    updateSequence(4);

    // Detection Telemetry
    const summary = data.summary;
    monoCountBadge.textContent = `${summary.monopole_tower_count} DETECTED`;
    monoConfidence.textContent = summary.monopole_tower_confidence;

    suppCountBadge.textContent = `${summary.supporting_tower_count} DETECTED`;
    suppConfidence.textContent = summary.supporting_tower_confidence;

    totalDetectionsVal.textContent = summary.total_detections;
    overallConfidenceVal.textContent = summary.overall_confidence;

    // Annotated Visualization
    if (data.annotated_image_base64) {
      annotatedDataUrl = data.annotated_image_base64;
      annotatedImage.src = annotatedDataUrl;
    }

    // Telemetry Table
    telemetryTableBody.innerHTML = "";
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((det, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><span style="color: #38BDF8; font-weight: 600;">${det.class}</span></td>
          <td>${det.confidence_percent}</td>
          <td>[x1=${det.bbox.x1}, y1=${det.bbox.y1}, w=${det.bbox.width}, h=${det.bbox.height}]</td>
        `;
        telemetryTableBody.appendChild(tr);
      });
    } else {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="4" style="text-align: center; color: #94A3B8;">No tower components detected at current confidence gate.</td>`;
      telemetryTableBody.appendChild(tr);
    }

    detectionOutputWrap.classList.remove("hidden");
    updateSequence(5);
    metaStatus.textContent = "Inspection Complete";

  } else {
    // Quality Check: REJECTED (YOLO aborted)
    qcStatusBadge.textContent = "● REJECTED";
    qcStatusBadge.className = "diag-badge status-rejected";

    rejectionReasonText.textContent = `Image rejected: ${qc.reason}`;
    rejectionMsgBox.classList.remove("hidden");

    detectionOutputWrap.classList.add("hidden");
    metaStatus.textContent = "Quality Check Failed";

    // Mark timeline as halted at quality check
    stepQC.classList.remove("active");
    stepQC.classList.add("completed");
    stepDetection.classList.remove("active", "completed");
    stepDetection.style.color = "#EF4444";
    stepDetection.querySelector(".seq-icon").textContent = "✕";
  }
}

// Download Annotated Result
function downloadAnnotatedImage() {
  if (!annotatedDataUrl) {
    alert("No annotated image available to download.");
    return;
  }
  const link = document.createElement("a");
  link.href = annotatedDataUrl;
  link.download = `annotated_${activeFile ? activeFile.name : "tower_inspection.jpg"}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Reset Dashboard State
function resetDashboard() {
  activeFile = null;
  annotatedDataUrl = null;
  fileInput.value = "";

  dropzoneEmpty.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  imagePreview.src = "";

  metadataBox.classList.add("hidden");
  btnExecute.disabled = true;

  resetResultsArea();
  updateSequence(1);
}

function resetResultsArea() {
  resultsEmptyState.classList.remove("hidden");
  diagnosticCard.classList.add("hidden");
  detectionOutputWrap.classList.add("hidden");
  rejectionMsgBox.classList.add("hidden");
  annotatedImage.src = "";
  telemetryTableBody.innerHTML = "";
  stepDetection.style.color = "";
}
