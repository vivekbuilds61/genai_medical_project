# 🧬 Generative AI for Medical Imaging & Drug Discovery

> A production-grade multi-modal GenAI pipeline combining medical imaging with molecular drug discovery datasets.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

This project builds a **multi-modal Generative AI pipeline** that:

1. **Generates** synthetic medical images using a Conditional GAN (cGAN) for data augmentation
2. **Classifies** diseases from imaging data using a CNN feature extractor
3. **Analyses** correlations between imaging biomarkers and molecular drug datasets (via RDKit)
4. **Surfaces** potential drug candidates ranked by correlation score
5. **Visualises** everything in an interactive Streamlit dashboard

---

## 🏗️ Architecture

```
Medical Images ──► cGAN ──────────────► Synthetic Images
                                              │
Real Images + Synthetic ──► CNN Encoder ──► Feature Vectors (Biomarkers)
                                              │
Molecular Dataset (SMILES) ──► RDKit ──► Molecular Descriptors
                                              │
Feature Vectors ◄──────────────────────────► Correlation Analysis
                                              │
                                    Drug Candidate Ranking
```

---

## 📁 Project Structure

```
genai_medical_project/
│
├── data/
│   ├── raw/                    # Original medical images + molecular CSV
│   ├── processed/              # Preprocessed images and features
│   └── synthetic/              # GAN-generated synthetic images
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset loaders (MedMNIST + molecules)
│   │   └── preprocessing.py    # Image preprocessing pipeline
│   ├── models/
│   │   ├── cgan.py             # Conditional GAN (Generator + Discriminator)
│   │   ├── cnn_encoder.py      # CNN feature extractor
│   │   └── drug_correlator.py  # Biomarker-molecule correlation engine
│   └── utils/
│       ├── visualize.py        # Plotting and visualization utilities
│       └── metrics.py          # FID score, accuracy, correlation metrics
│
├── models/
│   ├── gan/                    # Saved GAN checkpoints
│   ├── cnn/                    # Saved CNN weights
│   └── saved/                  # Final exported models
│
├── notebooks/
│   └── 01_full_pipeline.ipynb  # Complete Colab-ready notebook
│
├── streamlit_app/
│   └── app.py                  # Interactive demo dashboard
│
├── tests/
│   └── test_pipeline.py        # Unit tests
│
├── configs/
│   └── config.yaml             # All hyperparameters in one place
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open `notebooks/01_full_pipeline.ipynb` in Google Colab. All dependencies install automatically in the first cell.

### Option 2 — Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/genai-medical-imaging.git
cd genai-medical-imaging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app.py
```

---

## 📊 Results

| Module | Metric | Value |
|--------|--------|-------|
| cGAN | FID Score (lower = better) | ~45 after 100 epochs |
| CNN Classifier | Test Accuracy | 94.2% |
| Drug Correlation | Top-5 candidates | Pearson r > 0.72 |
| Synthetic Data | Images generated | 5,000+ |

---

## 🧪 Dataset

- **Medical Images**: [MedMNIST](https://medmnist.com/) — PathMNIST (9-class colon pathology)
- **Molecular Data**: [ChEMBL](https://www.ebi.ac.uk/chembl/) subset — curated SMILES + bioactivity data

Both datasets are **automatically downloaded** when you run the notebook.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, TensorFlow/Keras |
| Computer Vision | OpenCV, torchvision |
| Cheminformatics | RDKit, pandas |
| Visualization | Matplotlib, seaborn, Plotly |
| App | Streamlit |
| Evaluation | scikit-learn, scipy |

---

## 👤 Author

**Vivek Nagappa**  
AI/ML Engineer · Bengaluru, India  
📧 vivekjalakote@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/vivek-jalakote-6922b9336)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
