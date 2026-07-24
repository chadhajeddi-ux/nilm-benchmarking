# BiWave-NILM: Benchmarking Deep Learning Models for Non-Intrusive Load Monitoring

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Chadha Jeddi  
**Institution:** ENI Tunis / ACTIA ES PowerLab  
**Type:** Engineering Internship Project  
**Target deployment:** STM32MP2 NPU (Neural-ART, INT8)

---

## Project Overview

Comprehensive benchmarking of deep learning models for **Non-Intrusive Load Monitoring (NILM)** — the task of disaggregating whole-house power consumption into individual appliance contributions using a single smart meter.

**Key contribution:** BiWave-NILM, a novel lightweight architecture designed for embedded deployment on the STM32MP2 Neural Processing Unit (NPU), combining:
- **DWT preprocessing** for physically invariant frequency signatures
- **Parallel BiGRU + BiTCN** branches for complementary temporal modeling
- **Lite-CBAM(T) attention** for channel and temporal focus
- **DyT normalization** for batch-size-independent embedded inference

---

## Baseline Models

| Model | Reference | Params | INT8 (MB) | Deployable |
|-------|-----------|--------|-----------|------------|
| CNN | Zhang et al., AAAI 2018 | 101K | 0.101 | ✅ |
| GRU | Krystalakos et al., 2018 | 184K | 0.184 | ✅ |
| BiGRU | Petralia et al., KDD 2025 | 465K | 0.465 | ✅ |
| LSTM | Hwang & Kang, IEEE TIM 2022 | 130K | 0.130 | ✅ |
| BiLSTM | Kelly & Knottenbelt, 2015 | 324K | 0.324 | ✅ |
| CNN-LSTM | Luo et al., 2023 | 272K | 0.272 | ✅ |
| NILMFormer | Petralia et al., KDD 2025 | 386K | 1.54 FP32 | ❌ GPU only |
| **BiWave-NILM** | **This work** | **354K** | **0.354** | **✅** |

---

## Datasets

| Dataset | Houses | Sampling | Appliances | Size |
|---------|--------|----------|------------|------|
| UK-DALE | 5 | 6s | Kettle, Fridge, Washer, Dishwasher, Microwave | 3.3 GB |
| REDD | 6 | 3s→6s | Fridge, Washer, Dishwasher, Microwave | 401 MB |
| AMPds2 | 1 | 60s→6s | Multiple | 560 MB |

**Train/test split (UK-DALE):** Houses 1,3,4,5 for training → House 2 for testing.
Matches NILMFormer (Petralia et al., KDD 2025) experimental protocol exactly.

---

## Project Structure

```
nilm-benchmarking/
├── src/
│   ├── config.py           # Central configuration
│   ├── dwt.py              # DWT decomposition (db2, level 3)
│   ├── preprocessing.py    # Data loading and cleaning
│   ├── dataset.py          # PyTorch Dataset with gap-aware windowing
│   ├── metrics.py          # 11 evaluation metrics + Focal Loss
│   ├── train.py            # Unified training loop (all 8 models)
│   └── evaluate.py         # Evaluation pipeline + cross-dataset
├── models/
│   ├── baselines/          # 7 baseline model implementations
│   │   ├── cnn_model.py
│   │   ├── gru_model.py
│   │   ├── bigru_model.py
│   │   ├── lstm_model.py
│   │   ├── bilstm_model.py
│   │   ├── cnn_lstm_model.py
│   │   ├── nilmformer_model.py
│   │   └── docs/           # README per model with references
│   └── proposed/
│       └── proposed_model.py   # BiWave-NILM
├── notebooks/
│   ├── local/              # Data exploration (runs locally)
│   └── colab/              # Training notebooks (runs on Colab GPU)
├── experiments/
│   ├── checkpoints/        # Best model weights (.pth)
│   └── results/            # Evaluation metrics (JSON)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/chadhajeddi-ux/nilm-benchmarking
cd nilm-benchmarking

# Create virtual environment
python3.10 -m venv nilm_env
source nilm_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Explore Data

```bash
jupyter notebook notebooks/local/01_data_exploration.ipynb
```

### 2. Train a Model (local quick test)

```bash
# Quick test — 3 epochs, 50K rows, CPU
python src/train.py --model biwave --quick

# Full training — 50 epochs (use Colab GPU)
python src/train.py --model biwave --dataset ukdale --epochs 50

# Train all models
python src/train.py --all --dataset ukdale --epochs 50
```

### 3. Evaluate

```bash
# Within-dataset evaluation (UK-DALE house 2)
python src/evaluate.py \
    --checkpoint experiments/checkpoints/biwave_best.pth \
    --model biwave --dataset ukdale --house 2

# Cross-dataset evaluation
python src/evaluate.py \
    --checkpoint experiments/checkpoints/biwave_best.pth \
    --model biwave --cross-dataset \
    --source ukdale --target redd

# Evaluate all trained models
python src/evaluate.py --all --dataset ukdale --house 2
```

---

## Evaluation Metrics

All metrics match NILMFormer (Petralia et al., KDD 2025) Table 2:

| Metric | Description | Direction |
|--------|-------------|-----------|
| MAE | Mean Absolute Error (Watts) | ↓ |
| RMSE | Root Mean Squared Error | ↓ |
| NDE | Normalized Disaggregation Error | ↓ |
| SAE | Signal Aggregate Error | ↓ |
| TECA | Total Energy Correctly Assigned | ↑ |
| MR | Matching Ratio (best overall) | ↑ |
| F1 | F1 Score (ON/OFF classification) | ↑ |
| Accuracy | Classification accuracy | ↑ |
| Balanced Acc | Class-balanced accuracy | ↑ |
| Precision | Positive prediction accuracy | ↑ |
| Recall | True positive rate | ↑ |

---

## Deployment Targets (STM32MP2 NPU)

| Constraint | Target | BiWave-NILM |
|------------|--------|-------------|
| F1 Score | ≥ 0.80 | TBD after training |
| MAE | ≤ 50 W | TBD after training |
| Latency | ≤ 15 ms | TBD via STEdgeAI |
| Energy | ≤ 500 mW | TBD on board |
| RAM | ≤ 2 MB | ✅ 0.354 MB |

---

## Key References

1. Zhang et al. "Sequence-to-Point Learning with Neural Networks for NILM." **AAAI 2018.**
2. Luo et al. "A Multi-Task Learning Model for NILM Based on DWT." **J.Supercomputing 2023.**
3. Petralia et al. "NILMFormer: NILM that Accounts for Non-Stationarity." **KDD 2025.**
4. Hur et al. "Semi-Supervised Domain Adaptation for NILM." **Sensors 2022.**
5. Woo et al. "CBAM: Convolutional Block Attention Module." **ECCV 2018.**
6. Ma et al. "Transformers without Normalization (DyT)." **arXiv:2503.10622 2025.**
7. Lin et al. "Focal Loss for Dense Object Detection." **ICCV 2017.**
8. Kelly & Knottenbelt. "Neural NILM." **ACM BuildSys 2015.**
9. Howard et al. "MobileNets: Efficient CNNs (DepthwiseSep)." **2017.**
10. Bai et al. "An Empirical Evaluation of TCN for Sequence Modeling." **2018.**
