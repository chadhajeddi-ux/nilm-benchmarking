"""
cnn_lstm_model.py :  CNN-LSTM Baseline for NILM (seq2point)
============================================================
Architecture based on:
    Luo et al. "A Multi-Task Learning Model for NILM Based on
    Discrete Wavelet Transform." Journal of Supercomputing, 2023.
    DOI: https://doi.org/10.1007/s11227-022-05000-6

    This is the closest baseline to our proposed model , it uses
    the same Conv+LSTM pattern but without DWT preprocessing,
    bidirectionality, attention, or DyT normalization.

    Also validated by:
    Kim et al. "Convolutional sequence to sequence with attention
    for NILM." IEEE TIE, 2021.

Architecture:
    3× Conv1D blocks (feature extraction)
    MaxPool1D (temporal compression)
    LSTM (temporal modeling)
    Center slice → Dense → MultiTaskHeads

The CNN-LSTM hybrid is the most natural extension of both CNN and
LSTM baselines ,  CNN extracts local features, LSTM models temporal
dependencies across compressed representations.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


class CNNLSTMBaseline(nn.Module):
    """
    CNN + LSTM hybrid baseline for NILM.

    CNN blocks extract multi-scale local features from DWT sub-bands.
    MaxPool compresses temporal dimension before LSTM — reduces
    sequence length from 480 to 60, making LSTM processing 8× faster
    while maintaining sufficient temporal resolution for NILM.
    LSTM then models temporal dependencies across compressed features.

    This architecture directly precedes our proposed model in
    complexity — adding bidirectionality (BiGRU), attention (CBAM),
    and depthwise separable Conv gives the proposed model.

    Ablation value:
        CNN only  → CNN baseline (no temporal modeling)
        LSTM only → LSTM baseline (no local feature extraction)
        CNN+LSTM  → this model (both, but no attention/bidirectionality)
        Proposed  → CNN+BiGRU+CBAM+DyT (full model)

    Parameters
    ----------
    in_channels : int
        Default 6 (4 DWT + 2 temporal).
    window_size : int
        Default 480.
    n_appliances : int
        Default 5.
    conv_channels : list
        Output channels for each Conv block. Default [32, 64, 64].
    pool_size : int
        MaxPool kernel size for temporal compression. Default 8.
        480 / 8 = 60 positions fed to LSTM.
    lstm_hidden : int
        LSTM hidden size. Default 128.
    dropout : float
        Default 0.2.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        conv_channels: list = [32, 64, 64],
        pool_size: int = 8,
        lstm_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.window_size = window_size
        self.pool_size = pool_size
        # After MaxPool: center position in compressed sequence
        self.center_compressed = (window_size // 2) // pool_size

        # ---- 3× Conv1D blocks ----
        # Progressive feature extraction from DWT sub-bands
        conv_blocks = []
        ch_in = in_channels
        for ch_out in conv_channels:
            conv_blocks.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(ch_out),
                nn.Dropout(dropout),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*conv_blocks)

        # ---- Temporal compression ----
        # MaxPool reduces 480 → 60 before LSTM
        # This makes LSTM 8× faster without significant information loss
        # (NILM events span multiple timesteps, not single points)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        # ---- LSTM temporal modeling ----
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,  # unidirectional for this baseline
        )

        # ---- Dense + heads ----
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
        # CNN: (B, 6, 480) → (B, 64, 480)
        x = self.conv(x)

        # MaxPool: (B, 64, 480) → (B, 64, 60)
        x = self.pool(x)

        # LSTM: (B, 60, 64) → (B, 60, 128)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # Center of compressed sequence
        center = x[:, self.center_compressed, :]  # (B, 128)

        out = self.dense(center)
        return self.heads(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 55)
    print("CNN-LSTM Baseline — Self-Test")
    print("Reference: Luo et al. J.Supercomputing 2023")
    print("=" * 55)

    model = CNNLSTMBaseline()
    B = 8
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    assert powers.shape == (B, N_APPLIANCES)
    assert (powers >= 0).all()
    assert (gated <= powers + 1e-6).all()

    with torch.no_grad():
        after_conv = model.conv(x)
        after_pool = model.pool(after_conv)
        after_lstm, _ = model.lstm(after_pool.permute(0, 2, 1))

    print(f"[Input]  {tuple(x.shape)}")
    print(f"[Output] powers={tuple(powers.shape)}")
    print(f"[Shapes] conv={tuple(after_conv.shape)}, "
          f"pool={tuple(after_pool.shape)}, "
          f"lstm={tuple(after_lstm.shape)}")
    print(f"[Center] compressed index = {model.center_compressed} "
          f"(of {WINDOW_SIZE // model.pool_size} positions)")
    print(f"[Params] {model.count_parameters():,} | "
          f"{model.count_parameters()/1e6:.3f} MB INT8")

    print(f"\n[Ablation position in baseline hierarchy]")
    print(f"  CNN only   → local features, no temporal modeling")
    print(f"  LSTM only  → temporal modeling, no local features")
    print(f"  CNN+LSTM   → both, no attention, no bidirectionality")
    print(f"  Proposed   → CNN(dws)+BiGRU+CBAM(T)+DyT — full model")

    print(f"\n{'=' * 55}")
    print("All tests passed — CNN-LSTM baseline ready!")
    print(f"{'=' * 55}")
