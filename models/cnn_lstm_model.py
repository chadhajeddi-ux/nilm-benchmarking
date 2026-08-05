"""
cnn_lstm_model.py — CNN-LSTM with Residual Blocks + Attention for NILM
=======================================================================
Architecture inspired by:
    "Development of hybrid CNN-LSTM for NILM" ResearchGate 2025
    "A Novel Hybrid DL Approach for NILM Based on LSTM and CNN"
    Naderian, arXiv:2104.07809, 2021

Improvements over baseline CNN-LSTM:
    1. Residual blocks (RB) between conv layers — mitigates vanishing gradient
    2. Channel attention (CA) after residual — selects important feature maps
    3. Larger LSTM (128 hidden, 2 layers) — more temporal capacity
    4. Stronger regularization (dropout=0.2, BN after each conv)

Architecture:
    Conv Block 1 (32 filters) → Residual Block → Conv Block 2 (64 filters)
    → Channel Attention → Conv Block 3 (128 filters) → MaxPool(8)
    → LSTM(128, 2 layers) → Center → Dense → MultiTaskHeads

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for time series.
    Adds skip connection from input to output — mitigates vanishing gradient.
    Reference: He et al. 'Deep Residual Learning' CVPR 2016.
    """
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))  # skip connection


class ChannelAttention1D(nn.Module):
    """
    Channel Attention for 1D feature maps.
    Selectively focuses on most relevant frequency/feature channels.
    Inspired by CBAM (Woo et al. ECCV 2018) simplified for 1D.
    Paper calls this 'Attention Layer (AL)'.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # global average pooling
        self.gmp = nn.AdaptiveMaxPool1d(1)  # global max pooling
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        avg = self.fc(self.gap(x).squeeze(-1))  # (B, C)
        mx  = self.fc(self.gmp(x).squeeze(-1))  # (B, C)
        scale = self.sigmoid(avg + mx).unsqueeze(-1)  # (B, C, 1)
        return x * scale  # channel-weighted features


class CNNLSTMBaseline(nn.Module):
    """
    CNN-LSTM with Residual Blocks + Channel Attention for NILM.

    Improvements over vanilla CNN-LSTM:
    - Residual block prevents vanishing gradients in deeper network
    - Channel attention selects most discriminative frequency features
    - Filters: 32 → 64 → 128 (progressive, matches paper)
    - LSTM: 128 hidden, 2 layers (stronger temporal modeling)

    Architecture:
        (B,6,480) → Conv32 → ResBlock(32) → Conv64 → ChannelAttn(64)
        → Conv128 → MaxPool(8) → (B,128,60)
        → LSTM(128,2L) → (B,60,128)
        → Center → Dense(128→64) → MultiTaskHeads → (B,5)

    Parameters
    ----------
    in_channels : int   Default 6
    window_size : int   Default 480
    n_appliances: int   Default 5
    dropout     : float Default 0.2
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.2,
        pool_size: int = 8,
        lstm_hidden: int = 128,
    ):
        super().__init__()
        self.window_size = window_size
        self.pool_size = pool_size
        self.center_compressed = (window_size // 2) // pool_size

        # ── Block 1: Conv(6→32) ──────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Residual Block (32→32) — mitigates vanishing gradient ─
        self.res_block = ResidualBlock1D(32, dropout=dropout)

        # ── Block 2: Conv(32→64) ─────────────────────────────────
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Channel Attention (64) — selects important features ───
        self.channel_attn = ChannelAttention1D(64, reduction=4)

        # ── Block 3: Conv(64→128) ────────────────────────────────
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── MaxPool: 480 → 60 ────────────────────────────────────
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        # ── LSTM: temporal sequence modeling ─────────────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # ── Dense + MultiTaskHeads ────────────────────────────────
        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 6, 480)
        Returns: powers, states, gated — each (B, N_app)
        """
        # Conv1 + Residual
        x = self.conv1(x)       # (B, 32, 480)
        x = self.res_block(x)   # (B, 32, 480) — skip connection

        # Conv2 + Channel Attention
        x = self.conv2(x)           # (B, 64, 480)
        x = self.channel_attn(x)    # (B, 64, 480) — weighted channels

        # Conv3 + MaxPool
        x = self.conv3(x)   # (B, 128, 480)
        x = self.pool(x)    # (B, 128, 60)

        # LSTM
        x = x.permute(0, 2, 1)     # (B, 60, 128)
        x, _ = self.lstm(x)         # (B, 60, 128)

        # Center timestep
        center = x[:, self.center_compressed, :]  # (B, 128)

        out = self.dense(center)    # (B, 64)
        return self.heads(out)      # (B, 5) each

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-LSTM + ResidualBlocks + ChannelAttention — Self-Test")
    print("=" * 60)

    model = CNNLSTMBaseline()
    B = 4
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    assert powers.shape == (B, N_APPLIANCES)
    print(f"[Input]  {tuple(x.shape)}")
    print(f"[Output] powers={tuple(powers.shape)}")
    print(f"[Params] {model.count_parameters():,} | "
          f"{model.count_parameters()/1e6:.3f} MB INT8")
    print()
    print("Architecture:")
    print("  Conv(6→32) → ResBlock(32) → Conv(32→64)")
    print("  → ChannelAttn(64) → Conv(64→128) → MaxPool(8)")
    print("  → LSTM(128, 2L) → Center → Dense(128→64) → Heads")
    print()
    print("All tests passed!")
