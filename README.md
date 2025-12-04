# AI CUP 2025 Fall – Aortic Valve CT Object Detection

Team: TEAM_9104  
Member: 王丞頤 (WCY91)  
Private Leaderboard: 0.973565 / Rank 11

This repository contains the training and inference code used for the **AI CUP 2025 Fall – Aortic Valve CT Object Detection** competition.  
All code was designed to be run on **Google Colab**, but can also be adapted to local environments.

---

## 1. Environment & Installation

### 1.1 Recommended environment (Google Colab)

- OS: Linux (Colab default)
- Python: 3.10+
- GPU: A100 / L4 (Colab Pro / Pro+)
- Main libraries:
  - `ultralytics` (YOLO v11/v12)
  - `torch`, `torchvision`
  - `ensemble-boxes`
  - `gdown`
  - `opencv-python`, `numpy`, `pandas`

In a Colab notebook, run:

```python
!pip install ultralytics
!pip install ensemble-boxes
!pip install gdown
```

If you want to fully match my environment, you can also:

```python
!pip install -r requirements.txt
```

---

## 2. Project Structure

```text
AICUP_contest2_9104/
├─ train.ipynb        # Training and validation pipeline
├─ predict.ipynb      # Inference + ensemble + txt generation
├─ requirements.txt   # Python dependencies
├─ 9c.pt              # Trained weights (partial)
├─ 10m_a.pt
├─ 11m.pt
├─ 11m_a.pt
├─ 11n_a.pt
├─ 12m.pt
├─ 12n_a.pt
└─ ...
```

- `train.ipynb`: trains multiple YOLO models on the official training set, with two different patient splits.
- `predict.ipynb`: downloads test images and trained weights, runs multi-model inference, applies Weighted Boxes Fusion (WBF), and generates the final submission txt.

---

## 3. Data & Inputs / Outputs

### 3.1 Training (`train.ipynb`)

**Input:**

- Official competition training data (CT images + labels).
- Data is organized in YOLO format via a `data.yaml` file (paths to train/val folders and class names).

**Main configurable hyper-parameters:**

- `epochs = 85`
- `batch = 32`
- `imgsz = 640`
- `amp = True`
- `close_mosaic = 3` (last 3 epochs without mosaic)
- `optimizer = AdamW`

**Output:**

- Trained YOLO weights stored under `runs/detect/exp*/weights/best.pt`.
- Selected weights are renamed / copied (e.g. `12m.pt`, `12n_a.pt`, `10m_a.pt`, `9c.pt`, etc.) and used in `predict.ipynb`.

There are **two data split strategies** (by patient index), so models with `_a` in the name are trained on a different split, which helps diversity for ensemble.

---

### 3.2 Inference & Ensemble (`predict.ipynb`)

`predict.ipynb` is intended to be run on Google Colab.

**Step 1 – Download data & weights**

- Creates folders:
  - `./datasets/`
  - `./datasets/test/`
- Downloads the official testing zip by `gdown` into `/content/datasets/testing.zip`.
- Downloads all trained weight files (e.g. `yolo12m95.pt`, `yolo11m95.pt`, `yolo10m_a.pt`, `yolo9c.pt`, `yolo12n.pt`, `yolo11n.pt`) from Google Drive.
- Unzips the testing data into `/content/datasets/test/tmp`.

**Step 2 – Split test images**

The script:

- Automatically finds the root directory that contains `patient*` subfolders.
- Collects all `.png` images from those patient folders.
- Sorts them by filename and splits them into two halves:
  - First half → `./datasets/test/images1/`
  - Second half → `./datasets/test/images2/`

This is just for convenience and resource control; the two halves are processed separately but finally merged.

**Step 3 – Multi-model YOLO inference + WBF**

For each half:

- Load models:

  ```python
  from ultralytics import YOLO
  from ensemble_boxes import weighted_boxes_fusion

  model_paths = [
      "/content/yolo12m95.pt",
      "/content/yolo11m95.pt",
      "/content/yolo9c.pt",
      "/content/yolo10m_a.pt",  # or slightly different order per half
      "/content/yolo12n.pt",
      "/content/yolo11n.pt",
  ]
  models = [YOLO(p) for p in model_paths]
  ```

- For each `.png` image:
  - Run each model with:
    - `imgsz = 640`
    - `conf = 0.08`
    - `iou = 0.45`
  - Collect bounding boxes (`xyxy`), scores, and labels from all models.
  - Normalize boxes to `[0, 1]` and feed them into `weighted_boxes_fusion`.

- **Ensemble parameters (example for images1):**
  - `weights = [1.4, 1.3, 1.2, 1.2, 1.2, 1.2]`
  - `iou_thr = 0.5`
  - `skip_box_thr = 0.01`
  - Final score threshold: `MIN_SCORE = 0.088`

- **Ensemble parameters (example for images2):**
  - `weights = [1.4, 1.3, 1.1, 1.3, 1.2, 1.2]`
  - `iou_thr = 0.6`
  - `skip_box_thr = 0.01`
  - Final score threshold: `MIN_SCORE = 0.1`

**Output format per detection line:**

```text
image_id class_id score x1 y1 x2 y2
```

- `image_id`: file name without `.png`
- `class_id`: integer class index
- `score`: confidence (e.g. 0.9123)
- `x1, y1, x2, y2`: pixel coordinates in the original 512×512 image

Each half is saved to:

- `./predict_txt/images1_wbf.txt`
- `./predict_txt/images2_wbf.txt`

Then both are concatenated into:

- `./predict_txt/merged_wbf.txt`  
  → This is the final file submitted to the competition.

---

## 4. How to Reproduce My Results (Quick Guide)

1. **Open this repo in Google Colab**
   - `File > Open Notebook > GitHub` and search for `WCY91/AICUP_contest2_9104`
   - Or upload this folder as a zip.

2. **Run training (optional)**
   - Open `train.ipynb`.
   - Install dependencies (`!pip install -r requirements.txt` or minimal set).
   - Configure the training/validation split in the cells (as in notebook).
   - Run all cells to train models and generate weights.

3. **Run inference**
   - Open `predict.ipynb` in Colab.
   - Run cells from top to bottom:
     1. Install `ultralytics`, `ensemble-boxes`, `gdown`.
     2. Download test data and all weight files and change the model path to your own path.
     3. Split images into `images1`/`images2`.
     4. Run the two ensemble sections to generate `images1_wbf.txt` and `images2_wbf.txt`.
     5. Merge txt files into `merged_wbf.txt` and download it.

With these steps, a third-party user should be able to debug, retrain, and reproduce the competition results.

---

## 5. Notes

- All training data is from the **official AI CUP 2025 dataset**.  
- Pretrained models are from the **Ultralytics YOLO** open-source project.  
- Ensemble is implemented using the **`ensemble-boxes`** Python package.

---

## 6. External Resources & References

1. Tian, Y., Ye, Q., & Doermann, D. (2025). *Yolov12: Attention-centric real-time object detectors*. arXiv preprint arXiv:2502.12524.
2. Ultralytics YOLO: https://github.com/ultralytics/ultralytics
3. ensemble-boxes: https://pypi.org/project/ensemble-boxes/
