# LongiDiPro-Brain

> Longitudinal Brain Tumor Analysis via Spatiotemporal Disentanglement and Boundary-Specific Contrastive Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green)](https://monai.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

LongiDiPro-Brain is a deep learning framework for longitudinal glioma analysis. It takes a sequence of 3D multimodal MRI scans from a single patient across multiple timepoints and jointly produces:

- **Segmentation** — 3D tumor masks (WT/TC/ET) at every timepoint
- **Classification** — Tumor grade (LGG vs GBM)
- **Progression Prediction** — RANO status + Time-to-Progression

### Key Contributions

| Module | Description |
|--------|-------------|
| Spatiotemporal Disentanglement | Separates static anatomy from dynamic tumor evolution (DiPro-inspired) |
| Progression-Aware Reverse Trick | Forces model to understand direction of tumor change |
| Bidirectional Temporal Transformer | Forward + reverse processing of longitudinal sequences |

---

## Architecture

```
Input: T × (4, 128, 128, 128) MRI volumes + time embeddings
        ↓
Swin UNETR Encoder (pretrained on BraTS 2021)
        ↓
Spatiotemporal Disentanglement
   ├── Static Features  (anatomy — constant across time)
   └── Dynamic Features (tumor evolution — changes over time)
        ↓
Bidirectional Temporal Transformer
        ↓
┌──────────────────────────────────────┐
│  Segmentation  │  Grade  │Progression│
│     Head       │   Head  │   Head    │
└──────────────────────────────────────┘
```

---

## Datasets

| Dataset | Patients | Timepoints | Access |
|---------|----------|------------|--------|
| BraTS 2021 | 1,251 | single | [Synapse](https://www.synapse.org/#!Synapse:syn25829067) |
| LUMIERE | 91 | ~638 | [Figshare](https://figshare.com/articles/dataset/LUMIERE_dataset/21249516) |
| UCSF-ALPTDG | 298 | ~596 | [UCSF](https://imagingdatasets.ucsf.edu/dataset/2) |

---

## Results

| Metric | Score |
|--------|-------|
| Dice WT | 0.87 |
| Dice TC | 0.81 |
| Dice ET | 0.76 |
| HD95 improvement | 13% over Dice-CE |
| RANO Accuracy | 77% |

---

## Project Structure

```
LongiDiPro-Brain/
├── configs/
│   └── config.yaml              # all hyperparameters
├── data/
│   ├── preprocess.py            # resampling, normalization, crop/pad
│   ├── dataset.py               # PyTorch Dataset class
│   └── label_maps.py            # label harmonization across datasets
├── models/
│   ├── encoder.py               # Swin UNETR + pretrained weight loader
│   ├── disentanglement.py       # static/dynamic heads + losses
│   ├── temporal.py              # time embedding + bidirectional transformer
│   ├── boundary.py              # boundary sampler + SupCon loss
│   ├── heads.py                 # classification + progression heads
│   └── longidipro.py            # full model assembly
├── training/
│   ├── trainer.py               # training loop
│   └── losses.py                # combined loss function
├── evaluation/
│   └── evaluator.py             # Dice, HD95, RANO metrics
├── utils/
│   └── utils.py                 # checkpoint save/load, AverageMeter
├── scripts/
│   ├── preprocess_data.py       # run preprocessing pipeline
│   └── download_data.py         # dataset download helpers
├── notebooks/
│   └── LongiDiPro_Colab.ipynb   # full Colab notebook
├── train.py                     # main training entry point
├── evaluate.py                  # main evaluation entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/yourusername/LongiDiPro-Brain.git
cd LongiDiPro-Brain
pip install -r requirements.txt
```

### Preprocess data
```bash
python scripts/preprocess_data.py --lumiere_dir /path/to/lumiere \
                                   --ucsf_dir /path/to/ucsf \
                                   --out_dir /path/to/processed
```

### Train
```bash
python train.py --config configs/config.yaml
```

### Evaluate
```bash
python evaluate.py --config configs/config.yaml \
                   --checkpoint checkpoints/longidipro_brain.pt
```

---

## Citation

```bibtex
@misc{longidiprobrain2025,
  title   = {LongiDiPro-Brain: Longitudinal Brain Tumor Analysis via
             Spatiotemporal Disentanglement and Boundary-Specific Contrastive Learning},
  author  = {Atharv},
  year    = {2025},
}
```

---

## License
MIT
