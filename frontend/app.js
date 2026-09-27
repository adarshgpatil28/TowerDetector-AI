/**
 * AI TowerVision - Professional Infrastructure Inspection Platform Controller
 * Communicates with FastAPI backend, manages image ingestion, runs real-time
 * telemetry pipeline, quality gate verification, and renders YOLOv8 detections.
 */

// Base API configuration
const API_BASE_URL = window.location.origin;

// DOM Element Registry
const elements = {
  // Toast notifications
  toastContainer: document.getElementById("toastContainer"),

  // System Status Pill
  systemStatusPill: document.getElementById("systemStatusPill"),
  systemStatusText: document.getElementById("systemStatusText"),

  // Sample Dataset Selector
  sampleSelect: document.getElementById("sampleSelect"),
  btnLoadSample: document.getElementById("btnLoadSample"),

  // Dropzone & Ingestion
  dropzone: document.getElementById("dropzone"),
  dropzoneEmpty: document.getElementById("dropzoneEmpty"),
  previewWrap: document.getElementById("previewWrap"),
  imagePreview: document.getElementById("imagePreview"),
  fileInput: document.getElementById("fileInput"),
  btnChangeImage: document.getElementById("btnChangeImage"),

  // Upload progress & metadata
  uploadProgressBox: document.getElementById("uploadProgressBox"),
  uploadProgressLabel: document.getElementById("uploadProgressLabel"),
  uploadProgressPercent: document.getElementById("uploadProgressPercent"),
  uploadProgressFill: document.getElementById("uploadProgressFill"),

  metadataBox: document.getElementById("metadataBox"),
  metaFilename: document.getElementById("metaFilename"),
  metaDimensions: document.getElementById("metaDimensions"),
  metaSize: document.getElementById("metaSize"),
  metaStatus: document.getElementById("metaStatus"),

  // Inspection Parameters
  confThreshold: document.getElementById("confThreshold"),
  confVal: document.getElementById("confVal"),
  blurThreshold: document.getElementById("blurThreshold"),
  blurVal: document.getElementById("blurVal"),

  // Execution CTA
  btnExecute: document.getElementById("btnExecute"),
  btnExecuteText: document.getElementById("btnExecuteText"),
  btnExecuteIcon: document.getElementById("btnExecuteIcon"),
  btnSpinner: document.getElementById("btnSpinner"),
  btnReset: document.getElementById("btnReset"),

  // Pipeline Stepper Nodes
  pNode1: document.getElementById("pNode1"),
  pNode2: document.getElementById("pNode2"),
  pNode3: document.getElementById("pNode3"),
  pNode4: document.getElementById("pNode4"),
  pNode5: document.getElementById("pNode5"),

  // Results State Panels
  resultsEmptyState: document.getElementById("resultsEmptyState"),

  // Quality Report
  diagnosticCard: document.getElementById("diagnosticCard"),
  qcStatusBadge: document.getElementById("qcStatusBadge"),
  chkFinalStatus: document.getElementById("chkFinalStatus"),
  qcSharpnessVal: document.getElementById("qcSharpnessVal"),
  qcSharpnessThreshold: document.getElementById("qcSharpnessThreshold"),
  qcBrightnessVal: document.getElementById("qcBrightnessVal"),
  qcExposureVal: document.getElementById("qcExposureVal"),
  rejectionMsgBox: document.getElementById("rejectionMsgBox"),
  rejectionReasonText: document.getElementById("rejectionReasonText"),

  // Detection Results Wrap
  detectionOutputWrap: document.getElementById("detectionOutputWrap"),
  annotatedImage: document.getElementById("annotatedImage"),
  btnDownload: document.getElementById("btnDownload"),

  // Detection Statistics Cards
  suppCountBadge: document.getElementById("suppCountBadge"),
  suppConfidence: document.getElementById("suppConfidence"),
  monoCountBadge: document.getElementById("monoCountBadge"),
  monoConfidence: document.getElementById("monoConfidence"),
  totalDetectionsVal: document.getElementById("totalDetectionsVal"),
  overallConfidenceVal: document.getElementById("overallConfidenceVal"),

  // Confidence Bars
  barSuppPercent: document.getElementById("barSuppPercent"),
  barSuppFill: document.getElementById("barSuppFill"),
  barMonoPercent: document.getElementById("barMonoPercent"),
  barMonoFill: document.getElementById("barMonoFill"),
  barOverallPercent: document.getElementById("barOverallPercent"),
  barOverallFill: document.getElementById("barOverallFill"),

  // Processing Status Checklist Tags
  psUpload: document.getElementById("psUpload"),
  psQuality: document.getElementById("psQuality"),
  psDetection: document.getElementById("psDetection"),
  psResults: document.getElementById("psResults"),

  // Inspection Summary Details
  sumStatus: document.getElementById("sumStatus"),
  sumFilename: document.getElementById("sumFilename"),
  sumQuality: document.getElementById("sumQuality"),
  sumTotal: document.getElementById("sumTotal"),
  sumSupp: document.getElementById("sumSupp"),
  sumMono: document.getElementById("sumMono"),
  sumAvgConf: document.getElementById("sumAvgConf"),
  sumLatency: document.getElementById("sumLatency"),

  // Telemetry Table
  telemetryTableBody: document.getElementById("telemetryTableBody"),
};

// Application State
let activeFile = null;
let currentAnnotatedDataUrl = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  checkSystemStatus();
  fetchSampleList();
});

/**
 * Register all event handlers
 */
function initEventListeners() {
  // Navigation smooth scrolling & active indicator
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", function () {
      document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
      this.classList.add("active");
    });
  });

  // Dropzone click triggers file picker
  elements.dropzoneEmpty.addEventListener("click", () => elements.fileInput.click());

  // Change image button in preview frame
  if (elements.btnChangeImage) {
    elements.btnChangeImage.addEventListener("click", (e) => {
      e.stopPropagation();
      elements.fileInput.click();
    });
  }

  // File input selection
  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop handlers
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.remove("dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      handleFileSelected(dt.files[0]);
    }
  });

  // Sample dropdown & load button
  elements.sampleSelect.addEventListener("change", () => {
    elements.btnLoadSample.disabled = !elements.sampleSelect.value;
  });

  elements.btnLoadSample.addEventListener("click", loadSelectedSample);

  // Confidence Gate Slider Sync
  elements.confThreshold.addEventListener("input", (e) => {
    elements.confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Sharpness Gate Sync
  elements.blurThreshold.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value) || 100.0;
    elements.blurVal.textContent = val.toFixed(1);
    elements.qcSharpnessThreshold.textContent = `Target: ≥ ${val.toFixed(1)}`;
  });

  // Execute Inspection CTA
  elements.btnExecute.addEventListener("click", executeInspection);

  // Reset Inspection CTA
  elements.btnReset.addEventListener("click", resetInspection);

  // Download Annotated Image
  elements.btnDownload.addEventListener("click", downloadResultImage);
}

/**
 * Verify backend & model availability against GET /api/status
 */
async function checkSystemStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status`);
    if (response.ok) {
      const data = await response.json();
      if ((data.status === "ONLINE" || data.status === "ready") && data.weights_available !== false) {
        elements.systemStatusPill.classList.remove("status-offline");
        elements.systemStatusText.textContent = "SYSTEM READY";
      } else {
        markSystemOffline("MODEL UNLOADED");
      }
    } else {
      markSystemOffline("BACKEND OFFLINE");
    }
  } catch (err) {
    markSystemOffline("BACKEND OFFLINE");
  }
}

function markSystemOffline(msg) {
  elements.systemStatusPill.classList.add("status-offline");
  elements.systemStatusText.textContent = msg;
}

/**
 * Fetch available sample dataset images from GET /api/samples
 */
async function fetchSampleList() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/samples`);
    if (!response.ok) return;
    const data = await response.json();
    const samples = data.samples || (Array.isArray(data) ? data : []);
    if (Array.isArray(samples) && samples.length > 0) {
      elements.sampleSelect.innerHTML = `<option value="">-- Choose from Test Dataset (${samples.length} items) --</option>`;
      samples.forEach((s, idx) => {
        const opt = document.createElement("option");
        opt.value = s.filename;
        opt.textContent = `${idx + 1}. ${s.filename}`;
        elements.sampleSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load sample list:", err);
  }
}

/**
 * Load sample image from GET /api/samples/{filename}
 */
async function loadSelectedSample() {
  const filename = elements.sampleSelect.value;
  if (!filename) return;

  try {
    elements.btnLoadSample.disabled = true;
    elements.btnLoadSample.textContent = "Loading...";

    const res = await fetch(`${API_BASE_URL}/api/samples/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error("Failed to load sample image");

    const blob = await res.blob();
    const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
    handleFileSelected(file);
    showToast("success", "Sample Loaded", `Loaded test sample: ${filename}`);
  } catch (err) {
    showToast("error", "Sample Error", err.message || "Failed to load image sample");
  } finally {
    elements.btnLoadSample.disabled = false;
    elements.btnLoadSample.textContent = "Load Sample";
  }
}

/**
 * Handle new file selection (User upload or sample loader)
 */
function handleFileSelected(file) {
  if (!file.type.match(/^image\/(jpeg|png)$/) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
    showToast("error", "Invalid File", "Please select a valid JPG, JPEG, or PNG image.");
    return;
  }

  activeFile = file;

  // Display simulated upload progress
  simulateUploadProgress(() => {
    // Read and display thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
      elements.imagePreview.src = e.target.result;
      elements.dropzoneEmpty.classList.add("hidden");
      elements.previewWrap.classList.remove("hidden");

      // Measure dimensions
      const img = new Image();
      img.onload = () => {
        elements.metaDimensions.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
      };
      img.src = e.target.result;

      // Populate file telemetry
      elements.metaFilename.textContent = file.name;
      elements.metaSize.textContent = formatBytes(file.size);
      elements.metaStatus.textContent = "Ready for Inspection";
      elements.metadataBox.classList.remove("hidden");

      // Enable execute button
      elements.btnExecute.disabled = false;
      elements.btnExecuteText.textContent = "EXECUTE INSPECTION";

      // Reset pipeline indicators
      resetPipelineIndicators();
      setPipelineStep(1, "completed"); // Step 1: Image Uploaded
      updateProcessingTag("psUpload", "COMPLETED", "tag-completed");
    };
    reader.readAsDataURL(file);
  });
}

/**
 * Simulate upload progress animation
 */
function simulateUploadProgress(callback) {
  elements.uploadProgressBox.classList.remove("hidden");
  elements.uploadProgressFill.style.width = "0%";
  elements.uploadProgressPercent.textContent = "0%";
  elements.uploadProgressLabel.textContent = "Uploading Image...";

  let p = 0;
  const interval = setInterval(() => {
    p += 25;
    if (p > 100) p = 100;
    elements.uploadProgressFill.style.width = `${p}%`;
    elements.uploadProgressPercent.textContent = `${p}%`;

    if (p >= 100) {
      clearInterval(interval);
      elements.uploadProgressLabel.textContent = "✓ Upload completed";
      setTimeout(() => {
        elements.uploadProgressBox.classList.add("hidden");
        if (callback) callback();
      }, 300);
    }
  }, 35);
}

/**
 * Format bytes to readable size
 */
function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

/**
 * Helper to parse confidence values (e.g. "96.2%", 0.962, "No detections") into a percentage number [0-100]
 */
function parseConfidencePercent(val) {
  if (!val || val === "No detections" || val === "N/A" || val === "NOT DETECTED") return 0;
  if (typeof val === "number") return val <= 1 ? +(val * 100).toFixed(1) : +val.toFixed(1);
  const parsed = parseFloat(String(val).replace("%", ""));
  return isNaN(parsed) ? 0 : parsed;
}

/**
 * Main Inspection Execution
 */
async function executeInspection() {
  if (!activeFile) {
    showToast("warning", "No Image", "Please upload or select a tower image first.");
    return;
  }

  // UI state during execution
  elements.btnExecute.disabled = true;
  elements.btnSpinner.classList.remove("hidden");
  elements.btnExecuteIcon.classList.add("hidden");
  elements.btnExecuteText.textContent = "ANALYZING...";

  // Hide initial idle message
  elements.resultsEmptyState.classList.add("hidden");

  // Step 2 Active: Image Quality Check
  setPipelineStep(1, "completed");
  setPipelineStep(2, "active");
  updateProcessingTag("psQuality", "PROCESSING", "tag-processing");

  const startTime = performance.now();

  try {
    const formData = new FormData();
    formData.append("file", activeFile);

    const conf = elements.confThreshold.value;
    const blur = elements.blurThreshold.value;

    const queryParams = new URLSearchParams({
      conf_threshold: conf,
      blur_threshold: blur,
    });

    const response = await fetch(`${API_BASE_URL}/predict?${queryParams.toString()}`, {
      method: "POST",
      body: formData,
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error (${response.status})`);
    }

    const data = await response.json();
    renderInspectionResults(data, elapsed);

  } catch (err) {
    console.error("Inspection error:", err);
    showToast("error", "Detection Failed", err.message || "Backend service unavailable.");
    setPipelineStep(2, "failed");
    updateProcessingTag("psQuality", "FAILED", "tag-failed");
  } finally {
    elements.btnExecute.disabled = false;
    elements.btnSpinner.classList.add("hidden");
    elements.btnExecuteIcon.classList.remove("hidden");
    elements.btnExecuteText.textContent = "EXECUTE INSPECTION";
  }
}

/**
 * Render Results and Telemetry from Backend Response
 */
function renderInspectionResults(data, elapsedSec) {
  const isAccepted = data.status === "ACCEPTED";
  const quality = data.quality || {};

  // Display Quality Report Card
  elements.diagnosticCard.classList.remove("hidden");

  // Populate Quality Metrics
  elements.qcSharpnessVal.textContent = quality.sharpness !== undefined ? quality.sharpness.toFixed(1) : "-";
  elements.qcBrightnessVal.textContent = quality.brightness !== undefined ? quality.brightness.toFixed(1) : "-";
  elements.qcExposureVal.textContent = quality.exposure ? quality.exposure.toUpperCase() : "NORMAL";

  if (isAccepted) {
    // Quality Pass
    elements.qcStatusBadge.className = "q-badge badge-pass";
    elements.qcStatusBadge.textContent = "✓ ACCEPTED";
    elements.chkFinalStatus.className = "check-item";
    elements.chkFinalStatus.innerHTML = '<span class="chk-icon">✓</span> Quality Approved';
    elements.rejectionMsgBox.classList.add("hidden");

    // Stepper updates
    setPipelineStep(2, "completed");
    setPipelineStep(3, "completed");
    setPipelineStep(4, "completed");
    setPipelineStep(5, "completed");

    // Processing Status Tags
    updateProcessingTag("psQuality", "COMPLETED", "tag-completed");
    updateProcessingTag("psDetection", "COMPLETED", "tag-completed");
    updateProcessingTag("psResults", "COMPLETED", "tag-completed");

    // Show detection layout
    elements.detectionOutputWrap.classList.remove("hidden");

    // Result Image
    if (data.annotated_image_base64) {
      elements.annotatedImage.src = data.annotated_image_base64;
      currentAnnotatedDataUrl = data.annotated_image_base64;
    }

    // Parse summary
    const summary = data.summary || {};
    const detections = data.detections || [];

    const suppCount = summary.supporting_tower_count || 0;
    const monoCount = summary.monopole_tower_count || 0;
    const totalCount = summary.total_detections || 0;

    const suppConfRaw = summary.supporting_tower_confidence;
    const monoConfRaw = summary.monopole_tower_confidence;
    const avgConfRaw = summary.overall_confidence;

    // Stat Cards
    elements.suppCountBadge.textContent = String(suppCount).padStart(2, "0");
    elements.suppConfidence.textContent = suppCount > 0 && suppConfRaw !== "No detections"
      ? (typeof suppConfRaw === "string" ? suppConfRaw : `${(suppConfRaw * 100).toFixed(1)}%`)
      : "Not detected";

    elements.monoCountBadge.textContent = String(monoCount).padStart(2, "0");
    elements.monoConfidence.textContent = monoCount > 0 && monoConfRaw !== "No detections"
      ? (typeof monoConfRaw === "string" ? monoConfRaw : `${(monoConfRaw * 100).toFixed(1)}%`)
      : "Not detected";

    elements.totalDetectionsVal.textContent = String(totalCount).padStart(2, "0");
    elements.overallConfidenceVal.textContent = totalCount > 0 && avgConfRaw !== "No detections"
      ? (typeof avgConfRaw === "string" ? avgConfRaw : `${(avgConfRaw * 100).toFixed(1)}%`)
      : "0.0%";

    // Confidence Progress Bars
    const suppPct = suppCount > 0 ? parseConfidencePercent(suppConfRaw) : 0;
    elements.barSuppPercent.textContent = `${suppPct}%`;
    elements.barSuppFill.style.width = `${suppPct}%`;

    const monoPct = monoCount > 0 ? parseConfidencePercent(monoConfRaw) : 0;
    elements.barMonoPercent.textContent = `${monoPct}%`;
    elements.barMonoFill.style.width = `${monoPct}%`;

    const avgPct = totalCount > 0 ? parseConfidencePercent(avgConfRaw) : 0;
    elements.barOverallPercent.textContent = `${avgPct}%`;
    elements.barOverallFill.style.width = `${avgPct}%`;

    // Summary Card Details
    elements.sumStatus.textContent = "Detection Completed";
    elements.sumStatus.className = "sv highlight-green";
    elements.sumFilename.textContent = activeFile ? activeFile.name : "-";
    elements.sumQuality.textContent = "Accepted";
    elements.sumTotal.textContent = totalCount;
    elements.sumSupp.textContent = suppCount;
    elements.sumMono.textContent = monoCount;
    elements.sumAvgConf.textContent = `${avgPct}%`;
    elements.sumLatency.textContent = `${elapsedSec}s`;

    // Populate Telemetry Table
    populateTelemetryTable(detections);

    // Toast Notification
    if (totalCount > 0) {
      showToast(
        "success",
        "Detection Completed Successfully",
        `Identified ${totalCount} tower components in ${elapsedSec}s.`
      );
    } else {
      showToast(
        "warning",
        "No Components Detected",
        "Image was accepted, but no tower structures met the confidence gate."
      );
    }

  } else {
    // Quality Rejected
    elements.qcStatusBadge.className = "q-badge badge-fail";
    elements.qcStatusBadge.textContent = "✕ REJECTED";
    elements.chkFinalStatus.className = "check-item item-failed";
    elements.chkFinalStatus.innerHTML = '<span class="chk-icon">✕</span> Quality Rejected';

    // Show Rejection Reason Panel
    elements.rejectionMsgBox.classList.remove("hidden");
    const reason = quality.reason || "Image quality check failed.";
    elements.rejectionReasonText.textContent = reason;

    // Stepper updates: Pipeline stops at Step 2
    setPipelineStep(2, "failed");
    setPipelineStep(3, "waiting");
    setPipelineStep(4, "waiting");
    setPipelineStep(5, "waiting");

    // Processing Status Tags
    updateProcessingTag("psQuality", "FAILED", "tag-failed");
    updateProcessingTag("psDetection", "WAITING", "tag-waiting");
    updateProcessingTag("psResults", "WAITING", "tag-waiting");

    // Hide detection results container
    elements.detectionOutputWrap.classList.add("hidden");

    // Toast Alert
    showToast("error", "Image Rejected", reason);
  }
}

/**
 * Populate Bounding Box Telemetry Table
 */
function populateTelemetryTable(detections) {
  elements.telemetryTableBody.innerHTML = "";

  if (!detections || detections.length === 0) {
    elements.telemetryTableBody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-dim); padding: 16px;">
          No bounding box telemetry records generated.
        </td>
      </tr>
    `;
    return;
  }

  detections.forEach((det, idx) => {
    const tr = document.createElement("tr");
    const isSupp = det.class === "supporting_tower";
    const pillColor = isSupp ? "color: #2DD4BF;" : "color: #F59E0B;";

    const bboxStr = det.bbox
      ? `[${det.bbox.x1}, ${det.bbox.y1}, ${det.bbox.width}, ${det.bbox.height}]`
      : "-";

    tr.innerHTML = `
      <td>${String(idx + 1).padStart(2, "0")}</td>
      <td style="font-weight: 600; ${pillColor}">${det.class}</td>
      <td>${det.confidence_percent || `${(det.confidence * 100).toFixed(1)}%`}</td>
      <td style="font-family: var(--font-mono); color: var(--text-muted);">${bboxStr}</td>
    `;
    elements.telemetryTableBody.appendChild(tr);
  });
}

/**
 * Update pipeline stepper node styling
 */
function setPipelineStep(stepNumber, state) {
  const node = elements[`pNode${stepNumber}`];
  if (!node) return;

  node.classList.remove("active", "completed", "failed");
  if (state === "active") node.classList.add("active");
  if (state === "completed") node.classList.add("completed");
  if (state === "failed") node.classList.add("failed");
}

function resetPipelineIndicators() {
  for (let i = 1; i <= 5; i++) {
    const node = elements[`pNode${i}`];
    if (node) node.classList.remove("active", "completed", "failed");
  }
}

/**
 * Update processing tag status pill
 */
function updateProcessingTag(elementId, text, className) {
  const el = elements[elementId];
  if (!el) return;
  el.className = `status-tag-sm ${className}`;
  el.textContent = text;
}

/**
 * Download Annotated Result Image
 */
function downloadResultImage() {
  if (!currentAnnotatedDataUrl) {
    showToast("warning", "No Result Image", "Please execute inspection to generate result image.");
    return;
  }

  const link = document.createElement("a");
  const filename = activeFile ? activeFile.name.replace(/\.[^/.]+$/, "") : "inspection";
  link.download = `towervision_result_${filename}.jpg`;
  link.href = currentAnnotatedDataUrl;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("success", "Download Started", "Result image saved to your device.");
}

/**
 * Reset inspection workstation
 */
function resetInspection() {
  activeFile = null;
  currentAnnotatedDataUrl = null;
  elements.fileInput.value = "";
  elements.sampleSelect.value = "";
  elements.btnLoadSample.disabled = true;

  // Workstation UI Reset
  elements.imagePreview.src = "";
  elements.dropzoneEmpty.classList.remove("hidden");
  elements.previewWrap.classList.add("hidden");
  elements.metadataBox.classList.add("hidden");
  elements.uploadProgressBox.classList.add("hidden");

  // CTA button reset
  elements.btnExecute.disabled = true;
  elements.btnExecuteText.textContent = "WAITING FOR IMAGE";

  // Results reset
  elements.resultsEmptyState.classList.remove("hidden");
  elements.diagnosticCard.classList.add("hidden");
  elements.detectionOutputWrap.classList.add("hidden");
  elements.rejectionMsgBox.classList.add("hidden");

  // Stepper reset
  resetPipelineIndicators();

  // Tags reset
  updateProcessingTag("psUpload", "WAITING", "tag-waiting");
  updateProcessingTag("psQuality", "WAITING", "tag-waiting");
  updateProcessingTag("psDetection", "WAITING", "tag-waiting");
  updateProcessingTag("psResults", "WAITING", "tag-waiting");

  showToast("warning", "Workspace Reset", "Image and inspection records cleared.");
}

/**
 * Display professional toast notification
 */
function showToast(type, title, message) {
  if (!elements.toastContainer) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const icon = type === "success" ? "✓" : type === "error" ? "✕" : "⚠";

  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
    <button class="toast-close" type="button">&times;</button>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  closeBtn.addEventListener("click", () => removeToast(toast));

  elements.toastContainer.appendChild(toast);

  // Auto remove after 4.5 seconds
  setTimeout(() => removeToast(toast), 4500);
}

function removeToast(toast) {
  if (!toast || toast.classList.contains("toast-hiding")) return;
  toast.classList.add("toast-hiding");
  setTimeout(() => {
    if (toast.parentElement) toast.parentElement.removeChild(toast);
  }, 300);
}
