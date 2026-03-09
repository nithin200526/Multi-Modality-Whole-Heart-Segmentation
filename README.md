<div align="center">

# 🫀 Robust Multi-Modality Whole Heart Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.12.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MONAI](https://img.shields.io/badge/MONAI-%3E%3D1.0.0-76B900?style=for-the-badge)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![IEEE 2024](https://img.shields.io/badge/IEEE-Conference%202024-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieee.org/)

**Official PyTorch Implementation**

*"An Advanced Deep Learning Framework for Robust Multi-Modality Whole Heart Segmentation"*



</div>

---

## 📌 Table of Contents

- [Abstract](#-abstract)
- [Key Contributions](#-key-contributions--methodology)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Quantitative Results](#-quantitative-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation--requirements)
- [Usage](#-usage)
- [Authors & Acknowledgements](#-authors--acknowledgements)
- [Citation](#-citation)

---

## 📖 Abstract

Precise whole-heart segmentation is critical for diagnosing cardiovascular diseases and planning surgical interventions. However, **manual segmentation is time-consuming and prone to observer variability**.

- **3D deep learning models** are accurate but require massive computational resources.
- **Standard 2D models** are efficient but lack crucial spatial depth awareness.

This repository provides a **highly optimized, hybrid deep learning framework** to automate whole heart segmentation across multi-modality scans (CT and MRI), striking the optimal balance between **3D spatial awareness** and **2D computational efficiency**.

---

## 💡 Key Contributions & Methodology

To overcome the memory bottlenecks of 3D networks and the poor spatial awareness of 2D networks, we introduce a **Unified 2.5D Framework** alongside specialized networks for complex pathologies:

---

### 1. 🧠 Unified 2.5D ResNet34 Architecture

| Component | Description |
|---|---|
| **2.5D Slice Stacking** | Extracts spatial depth by stacking adjacent slices `(n-1, n, n+1)` into a 3-channel input, retaining volumetric awareness without the GPU RAM overhead of full 3D convolutions |
| **Transfer Learning** | Leverages pre-trained **ImageNet weights** on the ResNet34 backbone for rapid and robust feature extraction |

---

### 2. ⚡ HyperSeg Defibrillator for Congenital Defects

| Component | Description |
|---|---|
| **Nested UNet++ Design** | A dense, highly interconnected architecture specifically engineered to segment **structurally deformed hearts** (e.g., Congenital Heart Disease), preventing the model from losing thin or broken heart walls |

---

### 3. 🔧 Advanced Pipeline Enhancements

| Enhancement | Description |
|---|---|
| **Domain Adaptation (CycleGAN)** | Translates clean CT scans into synthetic MRI representations, training the model to be robust against modality-specific noise |
| **Test Time Augmentation (TTA)** | Eliminates Z-axis jaggedness during inference by spatially averaging horizontally flipped predictions |
| **Optimized Loss Function** | Combines **Tversky Loss** and **Cross-Entropy (CE)** to handle the extreme class imbalance between small heart structures and the large background |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   TIER 1: INPUT & PREPARATION                                               │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│   │  Scans (CT/MRI) │───▶│  Preprocessing  │───▶│ CycleGAN (Optional) │   │
│   │ MM-WHS & HVSMR  │    │ Isotropic+CLAHE │    │  Synthetic MRI Aug  │   │
│   └─────────────────┘    └────────┬────────┘    └──────────┬──────────┘   │
│                                   │  Standard               │  Synthetic   │
│                                   ▼                         ▼              │
│   TIER 2: NEURAL ARCHITECTURE                                               │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│   │  2.5D Stacking  │───▶│ ResNet34 Encoder│───▶│  Skip Connections   │   │
│   │  (n-1, n, n+1)  │    │    Backbone     │    │   + CNN Decoder     │   │
│   └─────────────────┘    └─────────────────┘    └──────────┬──────────┘   │
│                                                             │  Features    │
│                                                             ▼              │
│   TIER 3: OUTPUT                                                            │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│   │ Tversky + CE    │───▶│   Test Time     │───▶│  Final 3D NIfTI     │   │
│   │    Loss         │    │  Augmentation   │    │      Masks          │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Standard path** uses preprocessed scans directly. **Synthetic path** uses CycleGAN-generated augmentations for cross-modality robustness.

---

## 📊 Datasets

### MM-WHS — Multi-Modality Whole Heart Segmentation
- Contains **paired MRI and CT scans**
- Used for standard cardiac anatomy evaluation
- [Request Access →](https://zmiclab.github.io/zxh/0/mmwhs/)

### HVSMR 2.0 — High-Resolution Cardiovascular MRI
- Focuses on **Congenital Heart Disease (CHD)** with severe structural defects
- High-resolution MRI specifically for complex pathology benchmarking
- [Request Access →](http://segchd.csail.mit.edu/)

> ⚠️ **Privacy Notice:** Due to medical data privacy policies (HIPAA), raw datasets are **not included** in this repository. Please request access from the respective dataset providers and place the downloaded files in the `data/` directory as described in the [Usage](#-usage) section.

---

## 🏆 Quantitative Results

All models were evaluated using:
- **DSC** — Dice Similarity Coefficient *(higher is better ↑)*
- **HD95** — 95th Percentile Hausdorff Distance in mm *(lower is better ↓)*

---

### Standard Anatomy — MM-WHS Dataset

| Model | Dice Score ↑ | Precision ↑ | HD95 (mm) ↓ |
|:---|:---:|:---:|:---:|
| 🥇 **Our 2.5D ResNet34** | **0.92** | **0.94** | 19.92 |
| SegResNet | 0.86 | 0.90 | 4.64 |
| Swin-UNETR | 0.84 | 0.88 | 6.64 |
| Dynamic U-Net | 0.82 | 0.85 | 5.92 |

---

### Congenital Defects — HVSMR 2.0 Dataset

| Model | Dice Score ↑ | Precision ↑ | HD95 (mm) ↓ |
|:---|:---:|:---:|:---:|
| 🥇 **Our HyperSeg Model** | **0.89** | **0.88** | **2.78** |
| SegResNet | 0.70 | 0.76 | 34.40 |
| Swin-UNETR | 0.62 | 0.69 | 41.72 |

> 💡 **Key Takeaway:** Our HyperSeg model achieves a **+27% Dice improvement** and reduces HD95 by over **92%** compared to Swin-UNETR on the challenging HVSMR 2.0 congenital defect benchmark.

---


### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | `>= 1.12.0` | Core deep learning framework |
| `monai` | `>= 1.0.0` | Medical image training utilities |
| `numpy` | latest | Numerical operations |
| `scipy` | latest | Scientific computing |
| `SimpleITK` | latest | Medical image I/O and resampling |
| `nibabel` | latest | NIfTI file reading/writing |

---

## 🚀 Usage

### Step 1 — Prepare Your Data

Place your downloaded dataset files inside `data/raw/` following this structure:
```
data/raw/
├── ct/
│   ├── patient_001.nii.gz
│   └── ...
└── mri/
    ├── patient_001.nii.gz
    └── ...
```

### Step 2 — Data Preprocessing

Run the preprocessing pipeline to apply **1.0mm³ isotropic resampling** and **CLAHE intensity normalization**:

```bash
python preprocess.py --data_dir ./data/raw --out_dir ./data/processed
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data/raw` | Path to raw NIfTI input files |
| `--out_dir` | `./data/processed` | Path to save preprocessed outputs |

---

### Step 3 — Model Training

**For Standard Anatomy (MM-WHS Dataset):**

```bash
python train.py \
  --model resnet34 \
  --dataset mmwhs \
  --epochs 300 \
  --lr 0.0001
```

**For Congenital Defects (HVSMR 2.0 Dataset):**

```bash
python train.py \
  --model hyperseg \
  --dataset hvsmr \
  --epochs 300 \
  --lr 0.0001
```

| Argument | Options | Description |
|---|---|---|
| `--model` | `resnet34`, `hyperseg` | Model architecture to train |
| `--dataset` | `mmwhs`, `hvsmr` | Target dataset |
| `--epochs` | integer | Number of training epochs |
| `--lr` | float | Initial learning rate |

---

### Step 4 — Inference & Evaluation

Run inference on the test set with **Test Time Augmentation (TTA)** enabled to generate final 3D NIfTI segmentation masks:

```bash
python test.py \
  --weights ./checkpoints/best_model.pth \
  --use_tta True \
  --out_dir ./predictions
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | required | Path to trained model checkpoint |
| `--use_tta` | `True` | Enable/disable Test Time Augmentation |
| `--out_dir` | `./predictions` | Directory to save output NIfTI masks |

---

## 👨‍💻 Authors & Acknowledgements

This work was conducted by the following researchers at the **Institute of Aeronautical Engineering (IARE)**:

| Name | Department | Role |
|---|---|---|
| **Nandala Nithin** | CSE (AI & ML), IARE | Lead Author |
| **Tangturi Jeshwanth Goud** | CSE (AI & ML), IARE | Co-Author |
| **Maddikunta Srithan** | CSE (AI & ML), IARE | Co-Author |
| **Dr. B. Padmaja** | Institute of Aeronautical Engineering | Faculty Advisor |

We gratefully acknowledge the providers of the **MM-WHS** and **HVSMR 2.0** datasets for making their data available to the research community.

---

## 📝 Citation

If you find this code or our methodology useful in your research, please cite our paper:

```bibtex
@inproceedings{nithin2024heartseg,
  title     = {An Advanced Deep Learning Framework for Robust Multi-Modality Whole Heart Segmentation},
  author    = {Nithin, Nandala and Goud, Tangturi Jeshwanth and Srithan, Maddikunta and Padmaja, B.},
  booktitle = {IEEE International Conference},
  year      = {2024}
}
```

---

<div align="center">

**⭐ If this work helps your research, please consider starring the repository! ⭐**

Made with ❤️ by the IARE AI & ML Research Team

</div>
