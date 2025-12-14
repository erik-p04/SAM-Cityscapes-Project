# SAM–Cityscapes Project

This repository contains a course project evaluating **Segment Anything (SAM)** on an urban semantic segmentation task using the **Cityscapes** dataset. The project compares:

1. **Phase 1:** Zero-shot, prompt-based segmentation using SAM
2. **Phase 2:** A Cityscapes-tuned semantic segmentation head trained on frozen SAM features

The goal is to study the trade-offs between **promptable zero-shot segmentation** and **task-specific supervised adaptation**.

---

## Project Overview

* **Model:** Segment Anything Model (SAM)
* **Dataset:** Cityscapes (urban street scenes)
* **Framework:** PyTorch
* **Hardware:** RTX 3080 GPU (or equivalent)

### Classes Evaluated

* Road
* Sidewalk
* Building
* Person
* Car

---

## Repository Structure

```
SAM-Cityscapes-Project/
├── checkpoints/                 # SAM pretrained weights
├── configs/                     # YAML configuration files
├── data/                        # Cityscapes dataset (not included)
├── experiments/                 # Runnable experiment scripts
│   ├── run_zero_shot.py         # Phase 1: zero-shot evaluation
│   ├── test_zero_shot.py        # Phase 1 quick sanity check
│   ├── train_sam_head.py        # Phase 2: train semantic head
│   ├── phase2/                  # Phase 2 checkpoints & outputs
│   └── run_*/                   # Timestamped Phase 1 runs
├── src/                         # Reusable source code
│   ├── sam_predict.py           # SAM zero-shot wrapper
│   ├── sam_backbone.py          # Frozen SAM image encoder
│   ├── seg_head.py              # Semantic segmentation head
│   ├── dataset.py               # Phase 1 dataset (prompt-based)
│   ├── cityscapes_semantic.py   # Phase 2 semantic dataset
│   ├── metrics.py               # IoU metrics
│   ├── visualize.py             # Visualization utilities
│   ├── logger.py                # CSV experiment logging
│   └── utils.py                 # Shared helpers (device, seeds)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SAM-Cityscapes-Project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 5. Download SAM checkpoint

Download the ViT-H checkpoint and place it in:

```
checkpoints/sam_vit_h_4b8939.pth
```

---

## Dataset Setup (Cityscapes)

1. Download Cityscapes from the official website
2. Place it under:

```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

The project uses the **labelIds** masks provided by Cityscapes.

---

## Phase 1: Zero-Shot SAM Evaluation

### Description

SAM is evaluated *without training* using **point prompts** automatically sampled from Cityscapes ground-truth masks. For each image and class:

* A single point prompt is generated
* SAM predicts a mask
* IoU is computed against ground truth

### Run a quick sanity check

```bash
python experiments/test_zero_shot.py
```

### Run full zero-shot evaluation

```bash
python experiments/run_zero_shot.py --split val
```

### Outputs

Each run creates a timestamped directory under `experiments/` containing:

* `detailed_results.csv` – per-image, per-class IoU
* `summary.csv` – per-class IoU and mean IoU
* `visualizations/` – qualitative mask comparisons

---

## Phase 2: Cityscapes-Tuned Semantic Head

### Description

In Phase 2, SAM is used as a **frozen feature extractor**. A lightweight semantic segmentation head is trained on Cityscapes.

#### Architecture

* Frozen SAM image encoder
* Two 3×3 convolution layers + BatchNorm + ReLU
* Final 1×1 convolution for class prediction

#### Training Setup

* Loss: Cross-Entropy (ignore unlabeled pixels)
* Optimizer: Adam
* Learning Rate: 1e-3
* Batch Size: 2
* Epochs: 5

### Train the semantic head

```bash
python experiments/train_sam_head.py
```

### Outputs

```
experiments/phase2/
├── seg_head_epoch_*.pth     # Model checkpoints
├── summary.csv              # Final validation mIoU
└── visualizations/          # Qualitative predictions
```

---

## Evaluation Metrics

* **Per-class IoU**
* **Mean Intersection-over-Union (mIoU)**

Phase 1 and Phase 2 results can be directly compared to analyze:

* Performance gains from supervised adaptation
* Trade-offs between promptability and training cost

---

## Key Findings (Expected)

* Zero-shot SAM performs reasonably well on large, well-defined classes
* Training a lightweight head significantly improves mIoU
* Phase 2 produces more coherent and stable semantic regions

---

## Reproducibility

* Random seeds are fixed
* SAM backbone weights are frozen
* All experiments are fully scripted

---

## Notes

* This project is designed for academic experimentation, not real-time deployment
* Batch size is intentionally small due to SAM feature memory usage

---

## Acknowledgements

* Segment Anything Model (Meta AI)
* Cityscapes Dataset
* PyTorch

---
