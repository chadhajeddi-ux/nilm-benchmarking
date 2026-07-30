"""
config.py — Central Configuration for NILM Benchmarking Project
================================================================
All project constants live here. Every other file imports from this module.
Change a value here → it updates everywhere automatically.

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
Year    : 2026-2027
"""

import os
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================
# All paths are relative to the project root.
# Path() creates OS-independent paths (works on Linux, Mac, Windows).

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# __file__        = this file (config.py) → ~/nilm_project/src/config.py
# .parent         = go up one level       → ~/nilm_project/src/
# .parent.parent  = go up two levels      → ~/nilm_project/
# .resolve()      = convert to absolute path

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR    = PROJECT_ROOT / "experiments" / "checkpoints"
RESULTS_DIR        = PROJECT_ROOT / "experiments" / "results"

# Create directories if they don't exist yet.
# exist_ok=True means: don't crash if the folder already exists.
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================
# Three public NILM datasets used in this project.
# Each has different sampling rates, households, and appliances.

DATASETS = {
    "REDD": {
        "path": DATA_RAW_DIR / "REDD",
        "sampling_rate": 3,           # seconds between measurements
        "target_sampling_rate": 6,    # we resample everything to 6 seconds
        "houses": [1, 2, 3, 4, 5, 6],
        "test_house": 1,              # Leave-One-House-Out: house 1 for testing
    },
    "UK-DALE": {
        "path": DATA_RAW_DIR / "UK-DALE",
        "sampling_rate": 6,           # already at 6 seconds
        "target_sampling_rate": 6,
        "houses": [1, 2, 3, 4, 5],
        "test_house": 2,              # Following Mamba-ECA-UNet: house 2 for testing
    },
    "AMPds2": {
        "path": DATA_RAW_DIR / "AMPds2",
        "sampling_rate": 60,          # 1-minute intervals
        "target_sampling_rate": 6,    # upsample to 6 seconds (interpolation)
        "houses": [1],                # AMPds2 has only 1 house
        "test_house": 1,
    },
}


# ============================================================
# 3. APPLIANCE CONFIGURATION
# ============================================================
# Target appliances that exist across all three datasets.
# Each appliance has:
#   - power_threshold: Watts above which the appliance is considered ON
#   - max_power: clipping value to remove outliers
#   - type: "binary" (on/off like kettle) or "multi" (complex cycles like washer)

APPLIANCES = {
    "kettle": {
        "power_threshold": 2000,      # ON when power > 2000W
        "max_power": 3100,            # clip outliers above 3100W
        "type": "binary",             # simple on/off behavior
    },
    "fridge": {
        "power_threshold": 50,        # ON when power > 50W
        "max_power": 300,
        "type": "binary",             # compressor on/off (simplified)
    },
    "washing_machine": {
        "power_threshold": 20,        # ON when power > 20W
        "max_power": 2500,
        "type": "multi",              # complex cycle: heat → wash → rinse → spin
    },
    "dishwasher": {
        "power_threshold": 10,        # ON when power > 10W
        "max_power": 2500,
        "type": "multi",
    },
    "microwave": {
        "power_threshold": 200,       # ON when power > 200W
        "max_power": 3000,
        "type": "binary",
    },
}

# List of appliance names for iteration
APPLIANCE_NAMES = list(APPLIANCES.keys())
N_APPLIANCES = len(APPLIANCE_NAMES)


# ============================================================
# 4. DWT (Discrete Wavelet Transform) CONFIGURATION
# ============================================================
# Based on Luo et al. (2023) — "A multi-task learning model
# for NILM based on discrete wavelet transform"
# Validated empirically: db2 wavelet, level 3 decomposition.

DWT_WAVELET = "db2"         # Daubechies-2 (= D4 = 4 filter coefficients)
DWT_LEVEL = 3               # 3-level decomposition → 4 sub-bands
DWT_N_SUBBANDS = DWT_LEVEL + 1  # Always level + 1 = 4 sub-bands

# Sub-band names for clarity in code and logs:
# After pywt.wavedec(signal, 'db2', level=3) → [cA3, cD3, cD2, cD1]
# Mapping to paper notation:
#   cA3 = LP3 (very slow baseline trend)
#   cD3 = HP3 (medium-slow variations)
#   cD2 = HP2 (medium-fast variations)
#   cD1 = HP1 (fast transients — appliance switching events)
DWT_SUBBAND_NAMES = ["LP3", "HP3", "HP2", "HP1"]


# ============================================================
# 5. WINDOWING CONFIGURATION
# ============================================================
# Sliding window parameters for seq2point learning.
# Window of 480 points at 6-second sampling = 48 minutes.
# Validated by Luo et al. Figure 2: window sizes 400-600 are optimal.

WINDOW_SIZE = 480            # number of timesteps per window
SAMPLING_RATE = 6            # seconds between measurements
WINDOW_DURATION_MIN = (WINDOW_SIZE * SAMPLING_RATE) / 60  # = 48.0 minutes

# Stride = how many steps to slide the window between consecutive examples.
# stride=1 → maximum overlap → most training examples (recommended for training)
# stride=WINDOW_SIZE → no overlap → fewest examples (faster but less data)
TRAIN_STRIDE = 30           # stride=120: 12min between windows, fast training
TEST_STRIDE = 480            # non-overlapping windows for evaluation (NILMFormer protocol)

# Gap filling: forward-fill missing values if gap < MAX_GAP_SECONDS
MAX_GAP_SECONDS = 180        # 3 minutes, following Mamba-ECA-UNet paper


# ============================================================
# 6. MODEL HYPERPARAMETERS
# ============================================================
# These are starting values. Tuned via hyperparameter search on Day 12.

# --- Shared across all models ---
INPUT_CHANNELS = DWT_N_SUBBANDS + 2  # 4 DWT sub-bands + 2 temporal (sin/cos hour)
SEQ_LENGTH = WINDOW_SIZE         # 480 timesteps

# --- Your proposed model: DWT-BiGRU-Lite-CBAM(T)-DyT ---
PROPOSED_MODEL = {
    "name": "DWT-BiGRU-CBAM-DyT",

    # DepthwiseSep-Conv1D block
    "conv_out_channels": 64,      # output channels after pointwise conv
    "conv_kernel_size": 3,        # temporal kernel size
    "conv_padding": 1,            # same padding (output length = input length)

    # BiGRU
    "gru_hidden_size": 128,       # hidden units per direction
    "gru_num_layers": 2,          # stacked BiGRU layers
    "gru_dropout": 0.2,           # dropout between GRU layers (not on last layer)
    "gru_bidirectional": True,    # always True for NILM offline processing

    # Lite-CBAM(T) attention
    "cbam_reduction_ratio": 16,   # channel attention: 128 → 8 → 128
    "cbam_temporal_kernel": 7,    # temporal attention: Conv1D kernel size

    # DyT normalization: y = gamma * tanh(alpha * x) + beta
    # alpha, gamma, beta are learned per layer during training.
    # No batch dependency — works perfectly at batch=1 inference.
    "dyt_init_alpha": 1.0,        # initial value of alpha (learnable)
}

# --- Baseline models ---
BASELINE_MODELS = {
    "CNN": {
        "n_filters": [64, 128, 256],    # 3 Conv1D layers with increasing filters
        "kernel_sizes": [3, 3, 3],
        "dropout": 0.2,
    },
    "GRU": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
    },
    "LSTM": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
    },
    "BiLSTM": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
    },
    "CNN-LSTM": {
        "n_filters": [64, 128],
        "kernel_sizes": [3, 3],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
    },
}


# ============================================================
# 7. TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 256             # larger batches for class imbalance (more ON events per batch)
LEARNING_RATE = 1e-3         # Adam optimizer starting learning rate
WEIGHT_DECAY = 1e-5          # L2 regularization to prevent overfitting
NUM_EPOCHS = 50              # maximum training epochs
EARLY_STOPPING_PATIENCE = 10 # stop if validation loss doesn't improve for 10 epochs

# Learning rate scheduler: reduce LR when validation loss plateaus
LR_SCHEDULER = {
    "type": "ReduceLROnPlateau",
    "factor": 0.5,            # multiply LR by 0.5 when triggered
    "patience": 5,            # wait 5 epochs before reducing
    "min_lr": 1e-6,           # never go below this LR
}

# Random seed for reproducibility — same seed = same results every time
SEED = 42


# ============================================================
# 8. LOSS FUNCTION CONFIGURATION
# ============================================================
# Composite loss from Mamba-ECA-UNet with Focal Loss from Luo et al.
#
# L_total = L_state + L_power + L_gated
#
# L_state = Focal Loss (for ON/OFF classification — handles class imbalance)
# L_power = Smooth L1 (for power regression — robust to outliers)
# L_gated = Smooth L1 (for gated output: y_hat = p_hat * sigmoid(s_logit))

LOSS = {
    # Focal Loss parameters (classification head)
    "focal_gamma": 2.0,       # focus parameter: higher = more focus on hard examples
    "focal_alpha": 0.25,      # class weight for positive (ON) class

    # Smooth L1 beta (regression heads)
    "smooth_l1_beta": 1.0,    # transition point between L1 and L2 behavior

    # Task weights (if using weighted sum instead of equal weights)
    "weight_state": 1.0,      # weight for L_state
    "weight_power": 1.0,      # weight for L_power
    "weight_gated": 1.0,      # weight for L_gated
}


# ============================================================
# 9. EVALUATION METRICS
# ============================================================
# Four metrics required by the internship proposal:
#   F1-score, Accuracy, MAE, Recall
# Plus embedded metrics for holistic benchmarking:
#   Latency (ms), RAM (KB), Energy (mW), Parameters, MACs

# ON/OFF classification threshold:
# State probability > threshold → predicted as ON
STATE_THRESHOLD = 0.5        # default; Mamba-ECA-UNet uses 0.2

# Metrics to compute and report
CLASSIFICATION_METRICS = ["f1_score", "accuracy", "recall", "precision"]
REGRESSION_METRICS = ["mae", "mre"]  # MAE in Watts, MRE = mean relative error

# Embedded deployment targets (Stage 1 requirements vector r)
DEPLOYMENT_TARGETS = {
    "max_latency_ms": 15.0,       # inference must complete in ≤ 15ms
    "max_energy_mw": 500.0,       # power consumption ≤ 500mW
    "max_ram_kb": 2048.0,         # RAM usage ≤ 2MB
    "min_f1_score": 0.80,         # F1 must be ≥ 0.80
    "max_mae_watts": 50.0,        # MAE must be ≤ 50W
    "target_platform": "STM32MP2",
    "quantization": "INT8",
    "toolchain": "X-LINUX-AI",
}


# ============================================================
# 10. DEVICE CONFIGURATION
# ============================================================
# Automatically detect if GPU is available.
# Training: GPU (Kaggle/Colab) or CPU (local).
# Inference/deployment: STM32MP2 NPU via STEdgeAI.

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STEdgeAI path (discovered on your machine)
STEDGEAI_PATH = Path("/opt/ST/STEdgeAI/2.2")


# ============================================================
# 11. LOGGING & REPRODUCIBILITY
# ============================================================

# Experiment naming convention: {model}_{dataset}_{timestamp}
# Example: "BiGRU_UKDALE_20260721_143000"
EXPERIMENT_NAME_FORMAT = "{model}_{dataset}_{timestamp}"

# Print configuration summary when imported
def print_config():
    """Print a clean summary of the current configuration."""
    print("=" * 50)
    print("NILM Benchmarking — Configuration Summary")
    print("=" * 50)
    print(f"  Window:      {WINDOW_SIZE} points ({WINDOW_DURATION_MIN:.0f} min)")
    print(f"  DWT:         {DWT_WAVELET} level {DWT_LEVEL} → {DWT_N_SUBBANDS} sub-bands")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LEARNING_RATE}")
    print(f"  Epochs:      {NUM_EPOCHS}")
    print(f"  Device:      {DEVICE}")
    print(f"  Appliances:  {APPLIANCE_NAMES}")
    print(f"  Datasets:    {list(DATASETS.keys())}")
    print(f"  Target:      {DEPLOYMENT_TARGETS['target_platform']} "
          f"({DEPLOYMENT_TARGETS['quantization']})")
    print("=" * 50)


# ============================================================
# ENTRY POINT — Run this file directly to verify configuration
# ============================================================
# Usage: python src/config.py

if __name__ == "__main__":
    print_config()
