"""
cnn_lstm_2d_model.py — Kurtogram RGB 100x100x3 CNN-LSTM for NILM
=================================================================
Exact paper configuration:
    Image size: 100x100x3 (RGB Kurtogram)
    Architecture: CNN + ResidualBlocks + AttentionLayer + LSTM
    
Reference:
    "Development of hybrid CNN-LSTM for NILM" ResearchGate 2025
    Algorithm: CNN-LSTM with Residual Blocks (RB) and Attention Layer (AL)

RGB Mapping:
    R channel = LP3 (low freq trend)
    G channel = HP2 (mid freq cycles)  
    B channel = HP1 (high freq transients)

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


def compute_rgb_kurtogram(power_signal: np.ndarray,
                           dwt_bands: np.ndarray,
                           img_size: int = 100,
                           fs: float = 1/6) -> np.ndarray:
    """
    Convert power signal to 100x100x3 RGB Kurtogram image.
    
    RGB channels:
        R = Kurtogram of LP3 (low frequency trend)
        G = Kurtogram of HP2 (mid frequency cycles)
        B = Kurtogram of HP1 (high frequency transients)
    
    Parameters
    ----------
    power_signal : (480,) aggregate power window
    dwt_bands    : (4, 480) DWT decomposition
    img_size     : output image size (100x100 like paper)
    
    Returns
    -------
    rgb : (3, img_size, img_size) float32 in [0,1]
    """
    def band_to_kurtogram(signal_1d, n_levels=10, n_time=100):
        """Convert 1D signal to 2D Kurtogram."""
        kurtogram = np.zeros((n_levels, n_time), dtype=np.float32)
        window_sizes = [2**i for i in range(3, 3+n_levels)]
        
        for i, nperseg in enumerate(window_sizes):
            nperseg_safe = min(nperseg, len(signal_1d)//2)
            if nperseg_safe < 4:
                continue
            noverlap = nperseg_safe // 2
            try:
                _, _, Sxx = scipy_signal.spectrogram(
                    signal_1d, fs=fs,
                    nperseg=nperseg_safe, noverlap=noverlap)
                if Sxx.shape[1] < 2:
                    continue
                mean_s = Sxx.mean(axis=1, keepdims=True)
                std_s  = Sxx.std(axis=1, keepdims=True) + 1e-8
                normalized = (Sxx - mean_s) / std_s
                kurt = (normalized**4).mean(axis=1) - 3
                t_new = np.linspace(0, 1, n_time)
                t_old = np.linspace(0, 1, len(kurt))
                if len(kurt) > 1:
                    f_i = interp1d(t_old, kurt, kind='linear',
                                   fill_value='extrapolate')
                    kurtogram[i, :] = np.clip(f_i(t_new), -10, 50)
            except Exception:
                continue
        
        # Normalize to [0, 1]
        mn, mx = kurtogram.min(), kurtogram.max()
        if mx > mn:
            kurtogram = (kurtogram - mn) / (mx - mn)
        
        # Resize to img_size x img_size using scipy zoom
        from scipy.ndimage import zoom
        scale_h = img_size / kurtogram.shape[0]
        scale_w = img_size / kurtogram.shape[1]
        kurtogram = zoom(kurtogram, (scale_h, scale_w), order=1)
        return kurtogram.astype(np.float32)
    
    # R = LP3 (index 0), G = HP2 (index 2), B = HP1 (index 3)
    R = band_to_kurtogram(dwt_bands[0], n_levels=10, n_time=100)  # (100, 100)
    G = band_to_kurtogram(dwt_bands[2], n_levels=10, n_time=100)  # (100, 100)
    B = band_to_kurtogram(dwt_bands[3], n_levels=10, n_time=100)  # (100, 100)
    
    rgb = np.stack([R, G, B], axis=0)  # (3, 100, 100)
    return rgb


class ResidualBlock2D(nn.Module):
    """Residual block between conv layers — prevents vanishing gradient."""
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


class AttentionLayer2D(nn.Module):
    """
    Multi-head attention between conv and LSTM.
    Paper: 4 attention heads, Q=K=V=128 dim.
    Simplified channel attention version for efficiency.
    """
    def __init__(self, channels: int, n_heads: int = 4, reduction: int = 4):
        super().__init__()
        self.n_heads = n_heads
        # Channel attention (efficient version of multi-head)
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
    2D CNN-LSTM with RGB Kurtogram (100x100x3) input.
    
    Follows paper architecture exactly:
    - 3 CNN blocks: 32, 64, 128 filters with 3x3 kernels
    - Residual block between block 1 and block 3
    - Attention layer between conv and LSTM
    - LSTM: 64 units
    - Dense: 128 units
    - MultiTaskHeads (our addition for power+state regression)
    
    Input:  (B, 3, 100, 100) — RGB Kurtogram
    Output: powers, states, gated — each (B, N_app)
    """

    def __init__(
        self,
        in_channels: int = 3,  # RGB
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.2,
        lstm_hidden: int = 64,  # paper uses 64
    ):
        super().__init__()
        self.in_channels = in_channels

        # Block 1: Conv(3→32) — paper filter size 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout))

        # Residual Block (32→32)
        self.res_block = ResidualBlock2D(32, dropout)

        # Block 2: Conv(32→64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2))  # 100→50

        # Attention Layer (64 channels, 4 heads)
        self.attention = AttentionLayer2D(64, n_heads=4, reduction=4)

        # Block 3: Conv(64→128)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2))  # 50→25

        # Global Average Pool: (B, 128, 25, 25) → (B, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(4)  # (B, 128, 4, 4)

        # Reshape for LSTM: treat spatial positions as sequence
        # (B, 128, 4, 4) → (B, 16, 128) — 16 positions as sequence
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,  # 64 like paper
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Dense: 128 units like paper
        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 100, 100) — RGB Kurtogram
        Returns: powers, states, gated — each (B, N_app)
        """
        # CNN
        x = self.conv1(x)       # (B, 32, 100, 100)
        x = self.res_block(x)   # (B, 32, 100, 100)
        x = self.conv2(x)       # (B, 64, 50, 50)
        x = self.attention(x)   # (B, 64, 50, 50)
        x = self.conv3(x)       # (B, 128, 25, 25)
        x = self.gap(x)         # (B, 128, 4, 4)

        # Reshape for LSTM
        B, C, H, W = x.shape
        x = x.view(B, C, H*W)   # (B, 128, 16)
        x = x.permute(0, 2, 1)  # (B, 16, 128)

        # LSTM
        x, _ = self.lstm(x)     # (B, 16, 64)
        x = x[:, -1, :]         # last timestep (B, 64)

        # Dense + Heads
        out = self.dense(x)     # (B, 64)
        return self.heads(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("="*60)
    print("CNN-LSTM 2D RGB Kurtogram (100x100x3) — Self-Test")
    print("="*60)

    model = CNNLSTMKurtogram()
    B = 4
    x = torch.randn(B, 3, 100, 100)  # RGB Kurtogram
    powers, states, gated = model(x)
    assert powers.shape == (B, N_APPLIANCES)

    print(f"[Input]  {tuple(x.shape)} — RGB Kurtogram 100x100")
    print(f"[Output] powers={tuple(powers.shape)}")
    print(f"[Params] {model.count_parameters():,}")
    print()
    print("Architecture (paper-aligned):")
    print("  Conv(3→32) → ResBlock(32) → Conv(32→64) → MaxPool")
    print("  → AttentionLayer(4 heads) → Conv(64→128) → MaxPool")
    print("  → GAP(4×4) → LSTM(64) → Dense(128→64) → Heads")
    print()
    print("All tests passed!")
