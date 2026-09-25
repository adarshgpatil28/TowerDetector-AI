"""
AI-Based Tower Component Detection and Visualization - Phase 2
==============================================================
Pipeline:
  1. Input image ingestion
  2. Image Quality Check (Laplacian Blur & Brightness Exposure)
  3. YOLO Object Detection (using trained weights: runs/detect/train/weights/best.pt)
  4. Confidence & Bounding Box extraction for 'monopole_tower' & 'supporting_tower'
  5. Average confidence calculation per class and overall
  6. Visual annotation and saving to runs/phase2_output/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONSTANTS & DEFAULT PATHS
# ==========================================
DEFAULT_WEIGHTS_PATH = Path("runs/detect/train/weights/best.pt")
DEFAULT_OUTPUT_DIR = Path("runs/phase2_output")
DEFAULT_CLASSES = ["monopole_tower", "supporting_tower"]

# Default thresholds
DEFAULT_BLUR_THRESHOLD = 100.0       # Variance of Laplacian < 100.0 => Blurry
DEFAULT_DARK_THRESHOLD = 40.0        # Mean brightness < 40.0 => Underexposed
DEFAULT_BRIGHT_THRESHOLD = 225.0     # Mean brightness > 225.0 => Overexposed
DEFAULT_CONF_THRESHOLD = 0.25        # YOLO detection confidence threshold


# ==========================================
# 1. IMAGE QUALITY CHECK FUNCTIONS
# ==========================================
def check_blur(image: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> Dict[str, Any]:
    """
    Checks if an image is blurred using the Variance of Laplacian method.

    Args:
        image: Input image (BGR or Grayscale as np.ndarray).
        threshold: Variance threshold below which the image is considered blurry.

    Returns:
        dict: {
            'is_blurry': bool,
            'blur_score': float,
            'threshold': float,
            'passed': bool
        }
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute the Laplacian of the image and return the variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    is_blurry = blur_score < threshold

    return {
        "is_blurry": is_blurry,
        "blur_score": blur_score,
        "threshold": threshold,
        "passed": not is_blurry
    }


def check_exposure(
    image: np.ndarray,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD
) -> Dict[str, Any]:
    """
    Analyzes image brightness to detect underexposed or overexposed images.

    Args:
        image: Input image (BGR or Grayscale as np.ndarray).
        dark_threshold: Brightness threshold below which the image is underexposed.
        bright_threshold: Brightness threshold above which the image is overexposed.

    Returns:
        dict: {
            'brightness': float,
            'is_underexposed': bool,
            'is_overexposed': bool,
            'dark_threshold': float,
            'bright_threshold': float,
            'passed': bool,
            'reason': Optional[str]
        }
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    brightness = float(np.mean(gray))
    is_underexposed = brightness < dark_threshold
    is_overexposed = brightness > bright_threshold

    reason = None
    if is_underexposed:
        reason = f"Image is underexposed (too dark: {brightness:.2f} < {dark_threshold})"
    elif is_overexposed:
        reason = f"Image is overexposed (too bright: {brightness:.2f} > {bright_threshold})"

    passed = not (is_underexposed or is_overexposed)

    return {
        "brightness": brightness,
        "is_underexposed": is_underexposed,
        "is_overexposed": is_overexposed,
        "dark_threshold": dark_threshold,
        "bright_threshold": bright_threshold,
        "passed": passed,
        "reason": reason
    }


def check_image_quality(
    image: np.ndarray,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD
) -> Dict[str, Any]:
    """
    Combines blur and exposure checks to determine if the image is suitable for YOLO detection.

    Args:
        image: Input image (BGR or Grayscale np.ndarray).
        blur_threshold: Variance of Laplacian threshold.
        dark_threshold: Minimum acceptable mean brightness.
        bright_threshold: Maximum acceptable mean brightness.

    Returns:
        dict: Quality assessment summary with status 'ACCEPTED' or 'REJECTED'.
    """
    blur_result = check_blur(image, threshold=blur_threshold)
    exposure_result = check_exposure(image, dark_threshold=dark_threshold, bright_threshold=bright_threshold)

    reasons = []
    if blur_result["is_blurry"]:
        reasons.append(f"Image is too blurry (Blur score: {blur_result['blur_score']:.2f} < {blur_threshold})")

    if exposure_result["is_underexposed"]:
        reasons.append(f"Image is underexposed (Brightness: {exposure_result['brightness']:.2f} < {dark_threshold})")
    elif exposure_result["is_overexposed"]:
        reasons.append(f"Image is overexposed (Brightness: {exposure_result['brightness']:.2f} > {bright_threshold})")

    passed = (len(reasons) == 0)
    status = "ACCEPTED" if passed else "REJECTED"

    return {
        "status": status,
        "passed": passed,
        "reasons": reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "blur_score": blur_result["blur_score"],
        "blur_threshold": blur_threshold,
        "brightness": exposure_result["brightness"],
        "dark_threshold": dark_threshold,
        "bright_threshold": bright_threshold,
        "blur_details": blur_result,
        "exposure_details": exposure_result
    }


# ==========================================
# 2. OBJECT DETECTION & CONFIDENCE FUNCTIONS
# ==========================================
def load_yolo_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH) -> YOLO:
    """
    Loads the trained YOLO model from the given weights path.

    Args:
        weights_path: Path to best.pt.

    Returns:
        YOLO: Loaded Ultralytics model instance.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found at '{weights_path.resolve()}'. "
            "Please ensure trained weights are present."
        )
    return YOLO(str(weights_path))


def run_detection(
    model: Union[YOLO, str, Path],
    image: Union[np.ndarray, str, Path],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> Dict[str, Any]:
    """
    Runs YOLO object detection on an image.

    Args:
        model: Either a loaded YOLO instance or path to weights.
        image: Either an image np.ndarray or a file path.
        conf_threshold: Minimum detection confidence threshold.

    Returns:
        dict: {
            'detections': List of dicts per detected box,
            'annotated_image': np.ndarray with bounding boxes and labels drawn,
            'raw_results': Ultralytics Results object
        }
    """
    if not isinstance(model, YOLO):
        model = load_yolo_model(model)

    # Perform prediction
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    first_result = results[0]

    detections = []
    for box in first_result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0].item())
        xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": xyxy  # [x1, y1, x2, y2]
        })

    # Render annotations using Ultralytics plot()
    annotated_image = first_result.plot()

    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "raw_results": first_result
    }


def calculate_average_confidence(
    detections: List[Dict[str, Any]],
    target_classes: List[str] = DEFAULT_CLASSES
) -> Dict[str, Any]:
    """
    Calculates the average confidence score per class and overall.

    Args:
        detections: List of detection dictionaries containing 'class_name' and 'confidence'.
        target_classes: List of class names to evaluate.

    Returns:
        dict: {
            'class_averages': {class_name: float or 'No detections'},
            'class_counts': {class_name: int},
            'overall': float or 'No detections',
            'total_detections': int
        }
    """
    class_scores: Dict[str, List[float]] = {cls: [] for cls in target_classes}
    all_scores: List[float] = []

    for det in detections:
        cls_name = det["class_name"]
        conf = det["confidence"]
        all_scores.append(conf)
        if cls_name in class_scores:
            class_scores[cls_name].append(conf)
        else:
            class_scores[cls_name] = [conf]

    # Calculate per-class averages
    class_averages: Dict[str, Union[float, str]] = {}
    class_counts: Dict[str, int] = {}
    for cls in target_classes:
        scores = class_scores.get(cls, [])
        class_counts[cls] = len(scores)
        if scores:
            class_averages[cls] = round(float(np.mean(scores)), 4)
        else:
            class_averages[cls] = "No detections"

    # Calculate overall average
    if all_scores:
        overall_avg: Union[float, str] = round(float(np.mean(all_scores)), 4)
    else:
        overall_avg = "No detections"

    return {
        "class_averages": class_averages,
        "class_counts": class_counts,
        "overall": overall_avg,
        "total_detections": len(detections)
    }


# ==========================================
# 3. HIGH-LEVEL PIPELINE PROCESSOR
# ==========================================
def process_image(
    image_input: Union[str, Path, np.ndarray],
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    save_output: bool = True,
    model: Optional[YOLO] = None
) -> Dict[str, Any]:
    """
    End-to-end processing pipeline for a single image:
      1. Ingest image
      2. Quality check (blur & exposure)
      3. If ACCEPTED: run YOLO detection
      4. Compute class-wise and overall average confidence
      5. Save annotated image to output directory

    Args:
        image_input: File path or np.ndarray image.
        weights_path: Path to best.pt weights.
        output_dir: Directory to save annotated outputs.
        blur_threshold: Laplacian threshold.
        dark_threshold: Min brightness threshold.
        bright_threshold: Max brightness threshold.
        conf_threshold: YOLO detection threshold.
        save_output: Whether to write annotated image to disk.
        model: Pre-loaded YOLO model (optional, for reuse in dashboard).

    Returns:
        dict: Complete pipeline results dictionary.
    """
    # 1. Load image
    input_path = None
    if isinstance(image_input, (str, Path)):
        input_path = Path(image_input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path.resolve()}")
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to read image from path: {input_path.resolve()}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # 2. Quality Check
    quality_result = check_image_quality(
        image=image,
        blur_threshold=blur_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold
    )

    result_payload: Dict[str, Any] = {
        "input_path": str(input_path) if input_path else None,
        "quality_check": quality_result,
        "detection_executed": False,
        "detections": [],
        "confidence_summary": None,
        "annotated_image": None,
        "output_path": None
    }

    # If rejected, DO NOT run YOLO detection
    if not quality_result["passed"]:
        return result_payload

    # 3. Object Detection (Only executed if quality check passed)
    if model is None:
        model = load_yolo_model(weights_path)

    det_result = run_detection(model=model, image=image, conf_threshold=conf_threshold)
    detections = det_result["detections"]
    annotated_image = det_result["annotated_image"]

    # 4. Confidence Calculations
    conf_summary = calculate_average_confidence(detections, target_classes=DEFAULT_CLASSES)

    # 5. Save Output Image
    saved_file_path = None
    if save_output and annotated_image is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if input_path is not None:
            out_filename = f"annotated_{input_path.name}"
        else:
            out_filename = "annotated_detection.jpg"

        saved_file_path = out_dir / out_filename
        cv2.imwrite(str(saved_file_path), annotated_image)

    result_payload.update({
        "detection_executed": True,
        "detections": detections,
        "confidence_summary": conf_summary,
        "annotated_image": annotated_image,
        "output_path": str(saved_file_path) if saved_file_path else None
    })

    return result_payload


# ==========================================
# 4. FORMATTED CONSOLE REPORTER
# ==========================================
def print_pipeline_report(results: Dict[str, Any]) -> None:
    """
    Prints a clear, structured terminal status report according to Phase 2 specifications.
    """
    qc = results["quality_check"]

    print("\n" + "=" * 50)
    print("IMAGE QUALITY CHECK")
    print("-" * 19)
    print(f"Status: {qc['status']}")
    if not qc["passed"]:
        print(f"Reason: {qc['reason']}")
    print(f"Blur score: {qc['blur_score']:.2f}")
    print(f"Brightness: {qc['brightness']:.2f}")

    if not results["detection_executed"]:
        print("=" * 50)
        print("[!] Object detection skipped because the image did not pass the quality check.")
        print("=" * 50 + "\n")
        return

    print("\nOBJECT DETECTION")
    print("-" * 16)
    detections = results["detections"]
    conf_summary = results["confidence_summary"]

    # Display detections summary per class
    for cls in DEFAULT_CLASSES:
        cls_count = conf_summary["class_counts"].get(cls, 0)
        if cls_count == 0:
            print(f"{cls}: No detections")
        else:
            print(f"{cls}: {cls_count} detected")

    print("\nDetected Tower Details:")
    if detections:
        for idx, det in enumerate(detections, 1):
            print(f"  {idx}. Class: {det['class_name']} | Confidence: {det['confidence']:.4f} | Box: {det['bbox']}")
    else:
        print("  None")

    print("\nAVERAGE CONFIDENCE")
    print("-" * 18)
    for cls in DEFAULT_CLASSES:
        avg = conf_summary["class_averages"].get(cls, "No detections")
        if isinstance(avg, float):
            print(f"{cls}: {avg:.4f}")
        else:
            print(f"{cls}: {avg}")

    overall = conf_summary["overall"]
    if isinstance(overall, float):
        print(f"Overall: {overall:.4f}")
    else:
        print(f"Overall: {overall}")

    print("=" * 50)
    if results.get("output_path"):
        print(f"Annotated output saved to:\n  -> {Path(results['output_path']).resolve()}")
    print("=" * 50 + "\n")


# ==========================================
# 5. COMMAND-LINE INTERFACE (CLI)
# ==========================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: AI-Based Tower Component Detection and Visualization Pipeline"
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        default=None,
        help="Positional path to the input image file."
    )
    parser.add_argument(
        "--image", "-i",
        dest="image_flag",
        default=None,
        help="Flag-based path to the input image file."
    )
    parser.add_argument(
        "--weights", "-w",
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"Path to trained YOLO weights file (default: {DEFAULT_WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for annotated images (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=DEFAULT_BLUR_THRESHOLD,
        help=f"Variance of Laplacian threshold for blur detection (default: {DEFAULT_BLUR_THRESHOLD})"
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=DEFAULT_DARK_THRESHOLD,
        help=f"Minimum mean brightness threshold for underexposure (default: {DEFAULT_DARK_THRESHOLD})"
    )
    parser.add_argument(
        "--bright-threshold",
        type=float,
        default=DEFAULT_BRIGHT_THRESHOLD,
        help=f"Maximum mean brightness threshold for overexposure (default: {DEFAULT_BRIGHT_THRESHOLD})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONF_THRESHOLD})"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine image path: prefer explicit flag, then positional, else fallback to a default test image
    image_path = args.image_flag or args.image_pos

    if not image_path:
        # Fallback to an available test image if user didn't specify one
        test_dir = Path("test/images")
        sample_images = list(test_dir.glob("*.jpg"))
        if sample_images:
            image_path = str(sample_images[0])
            print(f"[INFO] No image provided. Using sample test image: {image_path}")
        else:
            print("[ERROR] Please provide an image path: python phase2_tower_detection.py <path_to_image>")
            sys.exit(1)

    try:
        results = process_image(
            image_input=image_path,
            weights_path=args.weights,
            output_dir=args.output_dir,
            blur_threshold=args.blur_threshold,
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold,
            conf_threshold=args.conf,
            save_output=True
        )
        print_pipeline_report(results)
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
