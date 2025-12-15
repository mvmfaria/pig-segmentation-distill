# Data Distillation (or Weak label annotation?) Experiment

## 1. Baseline: Supervised Learning (Human Labels)
*Performance of YOLOv8 trained directly on the original PigLife human-annotated dataset.*
*(TBD)*

---

## 2. Teacher Baseline: SAM3 Zero-shot Performance
*Evaluated on the PigLife Test Set (Human Ground Truth).*

| Subset | mAP | mAP@50 | mAP@75 | AP (Medium) | AP (Large) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test** | **80.73%** | **93.61%** | **88.38%** | 3.74% | 81.13% |

### Observations on Ground Truth Quality
We observed a discrepancy between SAM3 predictions and Human Ground Truth:

**1. "False Positives" are actually Unlabeled Pigs:**
SAM3 successfully detects pigs that were missing from the human annotations (likely background pigs or piglets intentionally ignored by annotators). These correct detections are penalized as "False Positives" by the mAP metric.

<div align="center">
  <img src="teacher/outputs/1010s1120s2001-2s5300-1-31.png" width="45%">
  <img src="teacher/outputs/1010s1121s2001-2s5101-2-631.png" width="45%">
  <p><em>Figure 1 & 2: SAM3 detections (green) identifying pigs not present in Ground Truth.</em></p>
</div>

**2. Impact on Small/Medium Object Metrics:**
The low score on `AP_Medium` (3.74%) correlates with the observation above. The unlabeled background pigs are typically smaller/medium-sized due to distance, leading to a skewed performance metric for those size categories.

---

## 3. Student Performance: Distilled YOLOv8
*YOLOv8 models trained on **SAM3-generated pseudo-labels** (not human labels).*

### 3.1 Validation Performance (vs. SAM3 Labels)
*How well did the student learn to mimic the teacher?*

| Model | Params (M) | mAP@50 | mAP@50-95 | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 3.0 | 95.99% | 84.57% | 5.66 | 177 |
| **YOLOv8s** | 11.1 | 96.70% | 87.53% | 6.12 | 163 |
| **YOLOv8m** | 25.9 | **97.08%** | **88.77%** | 14.90 | 67 |

### 3.2 Test Performance (vs. Human Ground Truth)
*How well does the student perform on the real-world task?*

> **Note:** We've got that warning: "UserWarning: Encountered more than 100 detections in a single image. This means that certain detections with the lowest scores will be ignored, that may have an undesirable impact on performance. Please consider adjusting the `max_detection_threshold` to suit your use case."

| Model | mAP | mAP@50 | mAP@75 | AP (Medium) | AP (Large) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 76.79% | 93.27% | 86.31% | 4.84% | 77.18% |
| **YOLOv8s** | 78.77% | **93.61%** | 87.74% | 6.31% | 79.14% |
| **YOLOv8m** | **79.39%** | 93.55% | **88.22%** | **6.53%** | **79.80%** |