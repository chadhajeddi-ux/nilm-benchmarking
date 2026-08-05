"""
cnn_lstm_2d_model.py — Kurtogram-based 2D CNN-LSTM for NILM
============================================================
Paper approach: convert 1D power signal to 2D Kurtogram image,
then apply 2D CNN + LSTM for classification.

Reference:
    "Development of hybrid CNN-LSTM for NILM" ResearchGate 2025
    Algorithm: CNN-LSTM with Residual Blocks and Attention Layer

Architecture:
    Power Signal (480,) → Kurtogram (8, 24) → 2D CNN
    → ResidualBlock → ChannelAttention → MaxPool2D
    → Flatten → LSTM → Dense → MultiTaskHeads

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from scipy import signal as scipy_signal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


def compute_kurtogram_tensor(power_signal: np.ndarray,
                              n_levels: int = 8,
                              n_time: int = 24,
                              fs: float = 1/6) -> np.ndarray:
    """
    Convert 1D power signal to 2D Kurtogram image.
    Output: (n_levels, n_time) float32 array
    """
    from scipy.interpolate import interp1d
    kurtogram = np.zeros((n_levels, n_time), dtype=np.float32)
    window_sizes = [2**i for i in range(4, 4+n_levels)]

    for i, nperseg in enumerate(window_sizes):
        nperseg_safe = min(nperseg, len(power_signal)//2)
        if nperseg_safe < 4:
            continue
        noverlap = nperseg_safe // 2
        _, _, Sxx = scipy_signal.spectrogram(
            power_signal, fs=fs,
            nperseg=nperseg_safe, noverlap=noverlap)
        if Sxx.shape[1] < 2:
            continue
        mean_s = Sxx.mean(axis=1, keepdims=True)
        std_s  = Sxx.std(axis=1, keepdims=True) + 1e-8
        normalized = (Sxx - mean_s) / std_s
        kurt = (normalized**4).mean(axis=1) - 3
        t_new = np.linspace(0, 1, n_time)
        t_old = np.linspace(0, 1, len(kurt))
        f_i = interp1d(t_old, kurt, kind='linear', fill_value='extrapolate')
        kurtogram[i, :] = np.clip(f_i(t_new), -10, 50).astype(np.float32)

    # Normalize to [0, 1]
    mn, mx = kurtogram.min(), kurtogram.max()
    kurtogram = (kurtogram - mn) / (mx - mn + 1e-8)
    return kurtogram


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.gap(x).view(x.size(0), -1))
        mx  = self.fc(self.gmp(x).view(x.size(0), -1))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class CNNLSTMKurtogram(nn.Module):
    """
    2D CNN-LSTM using Kurtogram images as input.
    Follows the paper's approach exactly.

    Input: 6 × (8, 24) Kurtogram images — one per DWT channel
    (We use all 6 input channels as separate Kurtogram images)

    Architecture:
        (B, 6, 8, 24) → Conv2D(32) → ResBlock(32)
        → Conv2D(64) → ChannelAttn(64) → Conv2D(128) → MaxPool2D
        → Flatten → Reshape for LSTM → LSTM(64) → Dense → Heads
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.2,
        lstm_hidden: int = 64,
        n_levels: int = 8,
        n_time: int = 24,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_time = n_time

        # 2D CNN blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout))

        self.res_block = ResidualBlock2D(32, dropout)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout))

        self.channel_attn = ChannelAttention2D(64, reduction=4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout))

        self.pool = nn.MaxPool2d(kernel_size=2)
        # After pool: (B, 128, 4, 12) → flatten last dim for LSTM

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128*4,  # flattened freq dim
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x_kurtogram: torch.Tensor):
        """
        x_kurtogram: (B, 6, 8, 24) — 6 channel Kurtogram images
        Returns: powers, states, gated — each (B, N_app)
        """
        # 2D CNN
        x = self.conv1(x_kurtogram)     # (B, 32, 8, 24)
        x = self.res_block(x)           # (B, 32, 8, 24)
        x = self.conv2(x)               # (B, 64, 8, 24)
        x = self.channel_attn(x)        # (B, 64, 8, 24)
        x = self.conv3(x)               # (B, 128, 8, 24)
        x = self.pool(x)                # (B, 128, 4, 12)

        # Reshape for LSTM: treat time dim as sequence
        B, C, H, W = x.shape
        x = x.view(B, C*H, W)          # (B, 512, 12)
        x = x.permute(0, 2, 1)         # (B, 12, 512)

        # LSTM
        x, _ = self.lstm(x)             # (B, 12, 64)
        center = x[:, x.shape[1]//2, :] # center timestep (B, 64)

        out = self.dense(center)
        return self.heads(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("="*60)
    print("CNN-LSTM 2D Kurtogram — Self-Test")
    print("="*60)

    model = CNNLSTMKurtogram()
    B = 4
    # Simulate Kurtogram input
    x = torch.randn(B, INPUT_CHANNELS, 8, 24)
    powers, states, gated = model(x)
    assert powers.shape == (B, N_APPLIANCES)
    print(f"[Input]  {tuple(x.shape)} (B, channels, n_levels, n_time)")
    print(f"[Output] powers={tuple(powers.shape)}")
    print(f"[Params] {model.count_parameters():,}")
    print("All tests passed!")
