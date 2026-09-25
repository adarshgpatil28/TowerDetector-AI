"""
TowerVision - FastAPI Backend Server
====================================
Integrates Phase 1 dataset model weights & Phase 2 image-quality / YOLO inference pipeline.

Endpoints:
  - GET  /                     -> Serves the Phase 3 web dashboard (HTML)
  - GET  /api/status           -> System health, active model, and class metadata
  - GET  /api/samples          -> List of test dataset samples available for quick testing
  - GET  /api/samples/{name}   -> Retrieve sample image file
  - POST /predict              -> Main inspection endpoint (Quality check -> YOLO detection)
  - POST /api/inspect          -> Alias for /predict
"""

import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Direct reuse of existing Phase 2 pipeline functions
from phase2_tower_detection import (
    load_yolo_model,
    check_image_quality,
    calculate_average_confidence,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_DARK_THRESHOLD,
    DEFAULT_BRIGHT_THRESHOLD,
    DEFAULT_CLASSES
)

# Initialize FastAPI App
app = FastAPI(
    title="TowerVision API",
    description="AI-Based Tower Component Detection and Quality Assessment API",
    version="3.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEST_IMAGES_DIR = BASE_DIR / "test" / "images"

# Load and cache model at startup
_model = None

def get_model():
    global _model
    if _model is None:
        if not DEFAULT_WEIGHTS_PATH.exists():
            raise RuntimeError(f"Model weights file not found at: {DEFAULT_WEIGHTS_PATH.resolve()}")
        _model = load_yolo_model(DEFAULT_WEIGHTS_PATH)
    return _model


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/api/status")
async def get_system_status():
    """Returns system telemetry, active model architecture, and class mapping."""
    weights_exist = DEFAULT_WEIGHTS_PATH.exists()
    return {
        "status": "ONLINE" if weights_exist else "DEGRADED",
        "engine": "AI Engine Online",
        "model": "YOLOv8",
        "weights_path": str(DEFAULT_WEIGHTS_PATH),
        "weights_available": weights_exist,
        "classes": DEFAULT_CLASSES,
        "class_count": len(DEFAULT_CLASSES),
        "default_thresholds": {
            "confidence": 0.60,
            "blur_threshold": DEFAULT_BLUR_THRESHOLD,
            "dark_threshold": DEFAULT_DARK_THRESHOLD,
            "bright_threshold": DEFAULT_BRIGHT_THRESHOLD
        }
    }


@app.get("/api/samples")
async def list_sample_images():
    """Lists available test dataset images for quick validation."""
    if not TEST_IMAGES_DIR.exists():
        return {"samples": []}
    
    samples = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        for path in sorted(TEST_IMAGES_DIR.glob(ext)):
            samples.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size
            })
    return {"samples": samples}


@app.get("/api/samples/{filename}")
async def get_sample_image(filename: str):
    """Retrieves a specific sample image from test dataset."""
    sample_path = TEST_IMAGES_DIR / filename
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(sample_path))


@app.post("/predict")
@app.post("/api/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.60, ge=0.05, le=1.00),
    blur_threshold: float = Query(DEFAULT_BLUR_THRESHOLD, ge=1.0),
    dark_threshold: float = Query(DEFAULT_DARK_THRESHOLD, ge=0.0),
    bright_threshold: float = Query(DEFAULT_BRIGHT_THRESHOLD, le=255.0)
):
    """
    Executes the end-to-end inspection pipeline:
      1. Ingest image
      2. Image Quality Gate (Laplacian blur & brightness exposure)
      3. If REJECTED -> Halt and return explicit failure reason (no YOLO)
      4. If ACCEPTED -> Run YOLO detection on runs/detect/train/weights/best.pt
      5. Calculate per-class and overall average confidence
      6. Return telemetry, bounding boxes, and base64 annotated visualization
    """
    # 1. Validate file extension
    valid_exts = {".jpg", ".jpeg", ".png"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: JPG, JPEG, PNG."
        )

    # 2. Decode image buffer
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image buffer is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {str(e)}")

    height, width, channels = img.shape

    # 3. Image Quality Check
    qc_res = check_image_quality(
        image=img,
        blur_threshold=blur_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold
    )

    # If rejected, DO NOT send to YOLO
    if not qc_res["passed"]:
        exposure_desc = "Normal"
        if qc_res["exposure_details"]["is_underexposed"]:
            exposure_desc = "Underexposed"
        elif qc_res["exposure_details"]["is_overexposed"]:
            exposure_desc = "Overexposed"

        return JSONResponse(content={
            "status": "REJECTED",
            "passed": False,
            "filename": file.filename,
            "metadata": {
                "width": width,
                "height": height,
                "channels": channels
            },
            "quality": {
                "status": "REJECTED",
                "reason": qc_res["reason"],
                "sharpness": round(qc_res["blur_score"], 2),
                "sharpness_threshold": blur_threshold,
                "brightness": round(qc_res["brightness"], 2),
                "exposure": exposure_desc,
                "dark_threshold": dark_threshold,
                "bright_threshold": bright_threshold
            },
            "detection_executed": False,
            "detections": [],
            "summary": {
                "total_detections": 0,
                "monopole_tower_count": 0,
                "supporting_tower_count": 0,
                "monopole_tower_confidence": "N/A",
                "supporting_tower_confidence": "N/A",
                "overall_confidence": "N/A"
            },
            "annotated_image_base64": None,
            "output_path": None
        })

    # 4. Object Detection (Only when Quality Check passes)
    try:
        model = get_model()
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        first_res = results[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error with trained model: {str(e)}"
        )

    # Extract detections
    detections = []
    for box in first_res.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [round(float(c), 2) for c in xyxy]
        box_w = round(x2 - x1, 2)
        box_h = round(y2 - y1, 2)

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "confidence_percent": f"{conf * 100:.1f}%",
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": box_w,
                "height": box_h
            }
        })

    # Generate annotated image
    annotated_bgr = first_res.plot()

    # Save to runs/phase2_output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = f"annotated_{Path(file.filename).name}"
    out_file_path = DEFAULT_OUTPUT_DIR / out_filename
    cv2.imwrite(str(out_file_path), annotated_bgr)

    # Calculate average confidence metrics
    conf_summary = calculate_average_confidence(
        [{"class_name": d["class"], "confidence": d["confidence"]} for d in detections],
        target_classes=DEFAULT_CLASSES
    )

    # Convert annotated image to Base64 data URL for frontend display
    _, buffer = cv2.imencode(".jpg", annotated_bgr)
    b64_image = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_image}"

    mono_avg = conf_summary["class_averages"].get("monopole_tower", "No detections")
    supp_avg = conf_summary["class_averages"].get("supporting_tower", "No detections")
    overall_avg = conf_summary["overall"]

    return JSONResponse(content={
        "status": "ACCEPTED",
        "passed": True,
        "filename": file.filename,
        "metadata": {
            "width": width,
            "height": height,
            "channels": channels
        },
        "quality": {
            "status": "ACCEPTED",
            "reason": None,
            "sharpness": round(qc_res["blur_score"], 2),
            "sharpness_threshold": blur_threshold,
            "brightness": round(qc_res["brightness"], 2),
            "exposure": "Normal",
            "dark_threshold": dark_threshold,
            "bright_threshold": bright_threshold
        },
        "detection_executed": True,
        "detections": detections,
        "summary": {
            "total_detections": len(detections),
            "monopole_tower_count": conf_summary["class_counts"].get("monopole_tower", 0),
            "supporting_tower_count": conf_summary["class_counts"].get("supporting_tower", 0),
            "monopole_tower_confidence": f"{mono_avg * 100:.1f}%" if isinstance(mono_avg, float) else mono_avg,
            "supporting_tower_confidence": f"{supp_avg * 100:.1f}%" if isinstance(supp_avg, float) else supp_avg,
            "overall_confidence": f"{overall_avg * 100:.1f}%" if isinstance(overall_avg, float) else overall_avg
        },
        "annotated_image_base64": data_url,
        "output_path": str(out_file_path.resolve())
    })


# ==========================================
# FRONTEND STATIC FILES & SPA SERVING
# ==========================================
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend index.html not found"}


if __name__ == "__main__":
    import uvicorn
    print("\n[+] Starting TowerVision FastAPI Server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
