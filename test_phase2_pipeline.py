"""
Unit & Integration Tests for Phase 2: Tower Detection & Quality Check Pipeline
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from phase2_tower_detection import (
    check_blur,
    check_exposure,
    check_image_quality,
    calculate_average_confidence,
    process_image,
    DEFAULT_WEIGHTS_PATH
)


class TestPhase2Pipeline(unittest.TestCase):

    def setUp(self):
        self.test_img_path = Path("test/images/img_6e26e601e671417e_JPG.rf.cd88e849905ac0df0fb30eee3cc312d5.jpg")
        self.assertTrue(self.test_img_path.exists(), "Test sample image does not exist.")
        self.assertTrue(DEFAULT_WEIGHTS_PATH.exists(), "Weights file best.pt does not exist.")

    def test_check_blur(self):
        # A uniform flat image has 0 variance (extremely blurry)
        flat_image = np.zeros((100, 100, 3), dtype=np.uint8)
        res_flat = check_blur(flat_image, threshold=100.0)
        self.assertTrue(res_flat["is_blurry"])
        self.assertFalse(res_flat["passed"])
        self.assertEqual(res_flat["blur_score"], 0.0)

        # Real test image should be sharp (> 100)
        img = cv2.imread(str(self.test_img_path))
        res_sharp = check_blur(img, threshold=100.0)
        self.assertFalse(res_sharp["is_blurry"])
        self.assertTrue(res_sharp["passed"])
        self.assertGreater(res_sharp["blur_score"], 100.0)

    def test_check_exposure(self):
        # Pure black (underexposed)
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res_dark = check_exposure(black_img, dark_threshold=40.0, bright_threshold=225.0)
        self.assertTrue(res_dark["is_underexposed"])
        self.assertFalse(res_dark["passed"])

        # Pure white (overexposed)
        white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        res_bright = check_exposure(white_img, dark_threshold=40.0, bright_threshold=225.0)
        self.assertTrue(res_bright["is_overexposed"])
        self.assertFalse(res_bright["passed"])

        # Mid-tone gray (normal exposure)
        gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        res_norm = check_exposure(gray_img, dark_threshold=40.0, bright_threshold=225.0)
        self.assertFalse(res_norm["is_underexposed"])
        self.assertFalse(res_norm["is_overexposed"])
        self.assertTrue(res_norm["passed"])

    def test_check_image_quality_aggregate(self):
        # Black image should be rejected
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        qc = check_image_quality(black_img)
        self.assertEqual(qc["status"], "REJECTED")
        self.assertFalse(qc["passed"])

        # Real image should be accepted
        img = cv2.imread(str(self.test_img_path))
        qc_real = check_image_quality(img)
        self.assertEqual(qc_real["status"], "ACCEPTED")
        self.assertTrue(qc_real["passed"])

    def test_calculate_average_confidence(self):
        mock_dets = [
            {"class_name": "supporting_tower", "confidence": 0.90},
            {"class_name": "supporting_tower", "confidence": 0.94},
        ]
        summary = calculate_average_confidence(mock_dets, target_classes=["monopole_tower", "supporting_tower"])
        self.assertEqual(summary["class_averages"]["monopole_tower"], "No detections")
        self.assertAlmostEqual(summary["class_averages"]["supporting_tower"], 0.92, places=2)
        self.assertAlmostEqual(summary["overall"], 0.92, places=2)

    def test_end_to_end_acceptance(self):
        res = process_image(
            image_input=self.test_img_path,
            weights_path=DEFAULT_WEIGHTS_PATH,
            output_dir=Path("runs/phase2_output"),
            save_output=True
        )
        self.assertEqual(res["quality_check"]["status"], "ACCEPTED")
        self.assertTrue(res["detection_executed"])
        self.assertGreater(len(res["detections"]), 0)
        self.assertIsNotNone(res["output_path"])
        self.assertTrue(Path(res["output_path"]).exists())

    def test_end_to_end_rejection_skips_yolo(self):
        # Force rejection by setting strict blur threshold
        res = process_image(
            image_input=self.test_img_path,
            weights_path=DEFAULT_WEIGHTS_PATH,
            blur_threshold=999999.0,
            save_output=True
        )
        self.assertEqual(res["quality_check"]["status"], "REJECTED")
        self.assertFalse(res["detection_executed"])
        self.assertEqual(len(res["detections"]), 0)
        self.assertIsNone(res["output_path"])


if __name__ == "__main__":
    unittest.main()
