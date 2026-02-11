# 🧬 ProjectMetaData — Histopathology Cancer Detection System

<p align="center">
  <b>Research‑grade deep learning pipeline built from scratch.</b><br>
  Custom CNN • HDF5 Streaming • Apple Silicon Optimized • Desktop ONCO‑SCAN Interface
</p>

---

## ✦ Overview

**ProjectMetaData** is a from‑scratch deep learning system designed to analyze histopathology image patches and classify metastatic cancer presence.

Built entirely in PyTorch, this project simulates a real-world ML research workflow — from dataset ingestion and training to an interactive desktop inference interface.

> ⚠️ This project is for **research and learning purposes only** and is not intended for clinical diagnosis.

---

## ✦ Key Features

- 🧠 Custom CNN architecture (no pretrained backbones)
- ⚡ Apple Silicon (MPS) GPU acceleration
- 📦 HDF5 dataset streaming for massive datasets
- 💾 Automatic checkpoint saving & resume
- 🖥 ONCO‑SCAN desktop analysis interface
- 🧪 Modular structure for experimentation

---

## ✦ Installation

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/ProjectMetaData.git
cd ProjectMetaData
```

### 2. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```
pip3 install torch torchvision pillow numpy h5py
```

---

## ✦ Dataset Setup

Datasets are **not included** in this repository due to size.

You must download the dataset manually and place it inside the project folder.

### ➜ Download Dataset

```
[ https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon ]
```

### ➜ Required Folder Structure

After downloading, your project should look like this:

```
ProjectMetaData/
│
├── Training_Data/
│   ├── pcam/
│   │   └── test_split.h5
│   │
│   └── Labels/
│       └── camelyonpatch_level_2_split_test_y.h5
│
├── Model/
├── ONCO_SCAN_GUI.py
└── README.md
```

If filenames differ, update file paths inside the training script.

---

## ✦ Training the Model

Run:

```
python3 Model/train_model.py
```

Training pipeline:

- Input Size: 96×96 RGB
- Classes: 2 (Malignant / Healthy)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Checkpoints auto‑saved as `Cancter_Detector.pt`

The system streams images directly from `.h5` files without loading the entire dataset into RAM.

---

## ✦ Running Cancer Scanner

Launch the desktop inference interface:

```
python3 upload.py
```

Workflow:

1. Load trained model automatically
2. Upload a pathology image
3. Run analysis to view confidence scores

Interface includes:

- Animated scan visualization
- Confidence meters
- Metadata readout

---

## ✦ Project Structure

```
ProjectMetaData/
│
├── Model/
│   ├── dataset_loader.py
│   ├── training_pipeline.py
│   └── checkpoint_utils.py
│
├── Training_Data/
├── ONCO_SCAN_GUI.py
└── README.md
```

---

## ✦ Tech Stack

- Python
- PyTorch
- Torchvision
- Tkinter
- Pillow
- h5py

---

## ✦ Research Disclaimer

This repository demonstrates machine learning engineering concepts and experimental workflows.

It is **NOT** a medical device and must not be used for real diagnostic decisions.

---

## ✦ Future Roadmap

- Multi-class tumor classification
- Metadata-aware training pipeline
- CoreML / Swift deployment
- Performance optimizations for large-scale training

---

<p align="center">
  Built for learning. Built from scratch. Built to push boundaries.
</p>
