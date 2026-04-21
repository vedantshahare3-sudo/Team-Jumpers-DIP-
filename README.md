# 🌍 MAST-Net: Air Quality Prediction using Hybrid Deep Learning

This project is an implementation of the research paper:

**"Air quality prediction using multi-source remote sensing data integration with hybrid deep learning framework"**

It reproduces a simplified version of the proposed **MAST-Net (Multi-Modal Attention-based Spatio-Temporal Network)** using PyTorch.

---

## 🚀 Overview

Air pollution forecasting is a complex spatio-temporal problem. This project implements a hybrid deep learning model combining:

- CNN (for spatial feature extraction)
- Bi-LSTM (for temporal dependencies)
- Multi-head Attention (for feature fusion)
- Uncertainty Quantification (for prediction reliability)

The model predicts key pollutants:
- PM2.5
- PM10
- NO₂
- O₃

---

## 🧠 Architecture (MAST-Net)

The model follows the pipeline described in the paper:

### 1. Multi-Source Feature Extraction
- 4 parallel CNN branches simulate:
  - Sentinel-5P
  - MODIS
  - Landsat-8
  - Ground/Auxiliary data

### 2. Dynamic Feature Selection (DFS)
- Learns which features matter using a gating mechanism

### 3. Temporal Modeling
- Bi-directional LSTM captures time dependencies (7-day sequence)

### 4. Attention Fusion
- Multi-head attention (8 heads) integrates features

### 5. Uncertainty Output
- Outputs:
  - Mean predictions (4 pollutants)
  - Log variance (uncertainty)

---

## 📂 Project Structure
├── mastnet.py # Main model implementation
├── train.py # Training loop
├── data_simulator.py # Synthetic data generation
├── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/mast-net.git
cd mast-net
pip install torch matplotlib numpy

## Run training
python train.py

## Expected output
Epoch [1/10] | Loss: X.XXXX
...
Epoch [10/10] | Loss: X.XXXX
