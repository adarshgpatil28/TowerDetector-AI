# AI-Based Tower Component Detection and Visualization

An end-to-end Computer Vision system that evaluates aerial/ground images of transmission and telecommunication towers for image quality, detects tower components using a fine-tuned Ultralytics YOLO model, calculates detailed confidence analytics, and provides an interactive web dashboard.

---

## 🌟 Project Highlights

- **Phase 1: Dataset & Model Training**
  - Dataset configured in YOLO format with two classes:
    - `0: monopole_tower`
    - `1: supporting_tower`
  - High-precision YOLO weights saved at `runs/detect/train/weights/best.pt`.
- **Phase 2: Quality Filtering & Component Detection Pipeline**
  - **Laplacian Blur Detection**: Evaluates variance of Laplacian against configurable thresholds.
  - **Exposure & Brightness Analysis**: Rejects underexposed (too dark) or overexposed (too bright) images.
  - **Quality Gate**: Unsuitable images are rejected **before** sending them to the neural network.
  - **Confidence Metrics**: Calculates class-specific average confidence and dataset-wide macro statistics.
- **Phase 3: Interactive Streamlit Web Dashboard**
  - Web UI for uploading images or choosing test dataset samples.
  - Live progress indication and execution trigger button.
  - Side-by-side visualization of input vs. annotated output.
  - Dynamic metrics, bounding box breakdowns, and direct image download.

---

## 📁 Repository Structure

```text
Tower-Detection/
├── app.py                      # Phase 3 Streamlit Web Dashboard
├── phase2_tower_detection.py   # Phase 2 Core Pipeline & CLI Processor
├── batch_evaluate_phase2.py    # Batch evaluation script for test dataset
├── test_phase2_pipeline.py     # Automated unit & integration tests
├── requirements.txt            # Python dependencies
├── data.yaml                   # YOLO dataset configuration
├── README.md                   # Project documentation
├── train/                      # Training images & labels
├── valid/                      # Validation images & labels
├── test/                       # Test images & labels
│   ├── images/
│   └── labels/
└── runs/
    ├── detect/train/weights/
    │   └── best.pt             # Trained YOLO weights
    └── phase2_output/          # Output directory for annotated images & reports
        ├── batch_evaluation_report.md
        ├── batch_evaluation_summary.csv
        └── annotated_*.jpg
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Install required dependencies:

```powershell
pip install -r requirements.txt
```

Key packages installed:
- `streamlit`
- `ultralytics`
- `opencv-python`
- `numpy`
- `pillow`

---

## 💻 Usage Instructions

### 1. Launching the Phase 3 Web Dashboard

To launch the interactive dashboard, run:

```powershell
python -m streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

#### Dashboard Workflow:
1. **Upload Image**: Choose your own file (JPG, JPEG, PNG) or select one of the test samples from the dropdown.
2. **Configure Settings (Sidebar)**: Adjust the detection confidence threshold (default: `0.60`) or quality parameters.
3. **Execute**: Click the **"🚀 Execute Pipeline"** button.
4. **Quality Check**: The system validates sharpness and lighting. If rejected, processing stops with an explicit alert.
5. **Component Detection**: YOLO detects `monopole_tower` and `supporting_tower`.
6. **Results & Visualization**: View instance metrics, bounding boxes, and download the annotated output.

---

### 2. Running Phase 2 CLI Detection (Single Image)

You can run detection directly from the command line:

```powershell
# Analyze supporting tower image
python phase2_tower_detection.py test/images/img_6e26e601e671417e_JPG.rf.cd88e849905ac0df0fb30eee3cc312d5.jpg

# Analyze monopole tower image
python phase2_tower_detection.py test/images/img_759864cee7794560_jpg.rf.7abad289eca8bdc5f3667d5b0ef80f00.jpg

# Custom blur threshold
python phase2_tower_detection.py test/images/sample.jpg --blur-threshold 150.0
```

---

### 3. Running Batch Evaluation

To evaluate all images in the `test/images/` directory and export summary reports:

```powershell
python batch_evaluate_phase2.py
```

Outputs generated:
- `runs/phase2_output/batch_evaluation_summary.csv`
- `runs/phase2_output/batch_evaluation_report.md`
- Annotated images saved to `runs/phase2_output/`

---

### 4. Running Automated Tests

Run the unit test suite to verify the pipeline logic and quality gate constraints:

```powershell
python test_phase2_pipeline.py
```

---

## ⚙️ Quality Check Thresholds

| Metric | Measurement Method | Default Threshold | Rejection Criteria |
| :--- | :--- | :--- | :--- |
| **Blur** | Variance of Laplacian | `100.0` | Variance $< 100.0$ |
| **Underexposure** | Mean Grayscale Brightness | `40.0` | Mean Brightness $< 40.0$ |
| **Overexposure** | Mean Grayscale Brightness | `225.0` | Mean Brightness $> 225.0$ |

---

## 🏷️ Class Definitions

- `monopole_tower`: Single, tubular or solid mast tower structures.
- `supporting_tower`: Lattice, truss, guyed, or multi-leg pylons and transmission structures.
