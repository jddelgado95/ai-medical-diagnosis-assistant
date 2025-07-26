# AI Medical Diagnosis Assistant with CNNs and Grad-CAM (Deep Learning from End to End)

## Overview

The goal of this project is to build a deep learning-powered assistant that classifies medical images (e.g., X-rays or skin lesions) and explains its predictions. It includes:

- Data preprocessing & augmentation
- Model design using neural networks
- Training with optimization techniques
- Evaluation using metrics (accuracy, precision, recall, etc.)
- Explainability (Grad-CAM or saliency maps)
- Web demo with Streamlit

## Core concepts

ML Basics Data loading, train/test split, normalization
Algorithms SGD, Adam, backpropagation, activation functions
DL CNNs, transfer learning, dropout, batch norm
Evaluation Confusion matrix, ROC-AUC, F1-score
Explainability Grad-CAM, saliency maps
Deployment Streamlit/FastAPI + Docker

## Project structure:

```bash
ai-medical-assistant/
│
├── data/
│   ├── raw/              # Raw medical image dataset
│   └── processed/        # Resized, normalized data
│
├── notebooks/
│   └── eda.ipynb         # Data exploration & visualization
│
├── src/
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py        # Grad-CAM visualizations
│   └── utils.py
│
├── app/                  # Web interface
│   └── streamlit_app.py  # or FastAPI for a REST API
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Workflow:

1. User loads data via `dataloader.py` using PyTorch's ImageFolder
2. Model is built using `ResNet18` in `model.py`
3. Training starts with `train.py`
   - Optimizer: Adam
   - Loss: CrossEntropy
   - Metrics: Accuracy (so far)
4. After training, model is saved via `utils.py`
5. For explainability:
   - `explain.py` applies Grad-CAM to visualize what the model is focusing on
6. `streamlit_app.py` provides a user interface:
   - Upload image → Get prediction → See heatmap
7. (Optional) Model can be deployed in Docker

## Dataset

The project expects the dataset in ImageFolder format:

```bash
data/
  train/
    class_1/
    class_2/
  val/
    class_1/
    class_2/
```

You can use these datasets:
Chest X-ray Pneumonia: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Skin Lesions (ISIC): https://challenge.isic-archive.com/

Make sure to download the dataset and organize it accordingly into train/ and val/ folders.

## Tech stack

```bash
Python 3.x
PyTorch
OpenCV & PIL
Matplotlib
Grad-CAM
Streamlit
```
