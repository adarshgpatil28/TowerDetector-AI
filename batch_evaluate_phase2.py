"""
Batch Evaluation Script for Phase 2: Tower Detection & Quality Check
====================================================================
Evaluates the entire test dataset (test/images):
  1. Performs image quality check (Blur & Exposure)
  2. Runs YOLO detection on accepted images
  3. Computes class-wise and overall average confidence
  4. Saves annotated images to runs/phase2_output/
  5. Exports a comprehensive CSV summary and Markdown report
"""

import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

from phase2_tower_detection import (
    load_yolo_model,
    process_image,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_DARK_THRESHOLD,
    DEFAULT_BRIGHT_THRESHOLD,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_CLASSES
)


def run_batch_evaluation(
    test_dir: Union[str, Path] = Path("test/images"),
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
) -> Dict[str, Any]:
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths: List[Path] = []
    for ext in image_extensions:
        image_paths.extend(test_dir.glob(ext))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in '{test_dir.resolve()}'")

    print(f"\n[INFO] Found {len(image_paths)} images in '{test_dir}'.")
    print(f"[INFO] Loading YOLO model weights from '{weights_path}'...")
    model = load_yolo_model(weights_path)
    print("[INFO] Model loaded successfully. Starting batch evaluation...\n")

    records: List[Dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0

    all_monopole_confs: List[float] = []
    all_supporting_confs: List[float] = []

    print("-" * 110)
    print(f"{'#':<3} | {'Image Filename':<32} | {'Status':<8} | {'Blur':<7} | {'Bright':<7} | {'Mono':<5} | {'Supp':<5} | {'Overall Conf':<12}")
    print("-" * 110)

    for idx, img_path in enumerate(image_paths, 1):
        res = process_image(
            image_input=img_path,
            weights_path=weights_path,
            output_dir=output_dir,
            blur_threshold=blur_threshold,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
            conf_threshold=conf_threshold,
            save_output=True,
            model=model
        )

        qc = res["quality_check"]
        status = qc["status"]
        blur_val = qc["blur_score"]
        bright_val = qc["brightness"]

        if status == "ACCEPTED":
            accepted_count += 1
            cs = res["confidence_summary"]
            mono_cnt = cs["class_counts"].get("monopole_tower", 0)
            supp_cnt = cs["class_counts"].get("supporting_tower", 0)
            mono_avg = cs["class_averages"].get("monopole_tower", "No detections")
            supp_avg = cs["class_averages"].get("supporting_tower", "No detections")
            overall_avg = cs["overall"]
            rejection_reason = "N/A"
            total_dets = cs["total_detections"]

            for d in res["detections"]:
                if d["class_name"] == "monopole_tower":
                    all_monopole_confs.append(d["confidence"])
                elif d["class_name"] == "supporting_tower":
                    all_supporting_confs.append(d["confidence"])

            disp_mono = f"{mono_cnt} ({mono_avg if isinstance(mono_avg, str) else f'{mono_avg:.2f}'})"
            disp_supp = f"{supp_cnt} ({supp_avg if isinstance(supp_avg, str) else f'{supp_avg:.2f}'})"
            disp_overall = f"{overall_avg:.4f}" if isinstance(overall_avg, float) else str(overall_avg)
        else:
            rejected_count += 1
            mono_cnt = 0
            supp_cnt = 0
            mono_avg = "N/A"
            supp_avg = "N/A"
            overall_avg = "N/A"
            rejection_reason = qc["reason"]
            total_dets = 0
            disp_mono = "N/A"
            disp_supp = "N/A"
            disp_overall = "SKIPPED"

        short_name = img_path.name[:30] + ".." if len(img_path.name) > 32 else img_path.name
        print(f"{idx:<3} | {short_name:<32} | {status:<8} | {blur_val:<7.1f} | {bright_val:<7.1f} | {mono_cnt:<5} | {supp_cnt:<5} | {disp_overall:<12}")

        records.append({
            "image_filename": img_path.name,
            "quality_status": status,
            "rejection_reason": rejection_reason,
            "blur_score": round(blur_val, 2),
            "brightness": round(bright_val, 2),
            "monopole_tower_count": mono_cnt,
            "supporting_tower_count": supp_cnt,
            "total_detections": total_dets,
            "monopole_avg_conf": mono_avg,
            "supporting_avg_conf": supp_avg,
            "overall_avg_conf": overall_avg,
            "annotated_output_path": res["output_path"] or "N/A"
        })

    print("-" * 110)

    # Dataset Level Statistics
    total_mono_dets = len(all_monopole_confs)
    total_supp_dets = len(all_supporting_confs)
    total_all_dets = total_mono_dets + total_supp_dets

    macro_mono_avg = round(sum(all_monopole_confs) / total_mono_dets, 4) if total_mono_dets > 0 else "No detections"
    macro_supp_avg = round(sum(all_supporting_confs) / total_supp_dets, 4) if total_supp_dets > 0 else "No detections"
    all_confs = all_monopole_confs + all_supporting_confs
    macro_overall_avg = round(sum(all_confs) / total_all_dets, 4) if total_all_dets > 0 else "No detections"

    print("\n" + "=" * 50)
    print("BATCH EVALUATION DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Test Images:            {len(image_paths)}")
    print(f"Quality Check Passed:         {accepted_count} ({(accepted_count / len(image_paths) * 100):.1f}%)")
    print(f"Quality Check Rejected:       {rejected_count}")
    print(f"Total Detections:             {total_all_dets}")
    print(f"  - monopole_tower count:     {total_mono_dets}")
    print(f"  - supporting_tower count:   {total_supp_dets}")
    print(f"Macro Average Confidence:")
    print(f"  - monopole_tower:           {macro_mono_avg}")
    print(f"  - supporting_tower:         {macro_supp_avg}")
    print(f"  - Overall Dataset Average:  {macro_overall_avg}")
    print("=" * 50)

    # 1. Export CSV summary
    csv_file_path = output_dir / "batch_evaluation_summary.csv"
    fieldnames = [
        "image_filename",
        "quality_status",
        "rejection_reason",
        "blur_score",
        "brightness",
        "monopole_tower_count",
        "supporting_tower_count",
        "total_detections",
        "monopole_avg_conf",
        "supporting_avg_conf",
        "overall_avg_conf",
        "annotated_output_path"
    ]

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[+] CSV Summary report saved to:\n    -> {csv_file_path.resolve()}")

    # 2. Export Markdown report
    md_file_path = output_dir / "batch_evaluation_report.md"
    with open(md_file_path, mode="w", encoding="utf-8") as f:
        f.write("# Phase 2: Batch Evaluation Summary Report\n\n")
        f.write(f"- **Evaluated Images**: {len(image_paths)}\n")
        f.write(f"- **Accepted (Passed QC)**: {accepted_count}\n")
        f.write(f"- **Rejected (Failed QC)**: {rejected_count}\n")
        f.write(f"- **Total Tower Detections**: {total_all_dets}\n")
        f.write(f"  - `monopole_tower`: {total_mono_dets} (Avg Conf: {macro_mono_avg})\n")
        f.write(f"  - `supporting_tower`: {total_supp_dets} (Avg Conf: {macro_supp_avg})\n")
        f.write(f"- **Overall Dataset Average Confidence**: {macro_overall_avg}\n\n")

        f.write("## Per-Image Evaluation Results\n\n")
        f.write("| # | Image | Status | Blur | Brightness | Monopole (Avg Conf) | Supporting (Avg Conf) | Overall Conf |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for idx, r in enumerate(records, 1):
            f.write(
                f"| {idx} | `{r['image_filename']}` | {r['quality_status']} | "
                f"{r['blur_score']} | {r['brightness']} | "
                f"{r['monopole_tower_count']} ({r['monopole_avg_conf']}) | "
                f"{r['supporting_tower_count']} ({r['supporting_avg_conf']}) | "
                f"{r['overall_avg_conf']} |\n"
            )

    print(f"[+] Markdown report saved to:\n    -> {md_file_path.resolve()}\n")

    return {
        "records": records,
        "total_images": len(image_paths),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_detections": total_all_dets,
        "macro_mono_avg": macro_mono_avg,
        "macro_supp_avg": macro_supp_avg,
        "macro_overall_avg": macro_overall_avg,
        "csv_path": str(csv_file_path),
        "md_path": str(md_file_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluate Phase 2 Tower Detection on Dataset")
    parser.add_argument("--test-dir", default="test/images", help="Path to test images directory")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="Path to YOLO weights file")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for reports & images")
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD, help="Blur threshold")
    parser.add_argument("--dark-threshold", type=float, default=DEFAULT_DARK_THRESHOLD, help="Underexposure threshold")
    parser.add_argument("--bright-threshold", type=float, default=DEFAULT_BRIGHT_THRESHOLD, help="Overexposure threshold")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="YOLO confidence threshold")

    args = parser.parse_args()

    run_batch_evaluation(
        test_dir=args.test_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()
