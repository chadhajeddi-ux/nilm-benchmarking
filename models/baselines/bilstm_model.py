"""
bilstm_model.py — BiLSTM Baseline for NILM (seq2point)
========================================================
EXACT architecture from NILMFormer official source code:
    Kelly & Knottenbelt. "Neural NILM: Deep Neural Networks Applied
    to Energy Disaggregation." ACM BuildSys 2015.
    As implemented by:
    Petralia et al. NILMFormer (KDD 2025)
    Source: src/baselines/nilm/bilstm.py (©2025 EDF, Adrien Petralia)
    GitHub: https://github.com/adrienpetralia/NILMFormer

Reference numbers from NILMFormer Table 2 (UK-DALE, BiLSTM row):
    MAE: 130.864W | MR: 0.217 | F1: 0.229 | Accuracy: 0.430
    Recall: 0.921 | Precision: 0.165

Original architecture:
    Conv1d(c_in→16, k=4, same)
    BiLSTM(16→64, bidirectional) → 128
    BiLSTM(128→128, bidirectional) → 256
    Flatten(window × 256) → Linear → Linear → output

Our adaptations:
    1. Input: (B, 6, 480) DWT+temporal vs (B, 1, T) raw power
    2. Seq2point: center timestep vs seq2seq all timesteps
    3. Multi-task N=5 appliances vs single appliance output
    4. Gated output: ŷ = p̂ × σ(ŝ)

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


class BiLSTMBaseline(nn.Module):
    """
    Conv1D + Bidirectional LSTM for multi-task NILM.

    Reproduces the official NILMFormer BiLSTM baseline (bilstm.py)
    with adaptations for 6-channel DWT input and multi-task output.

    Key difference from BiGRU:
        LSTM has 3 gates (forget, input, output) + cell state c_t
        GRU  has 2 gates (reset, update) — no separate cell state
        LSTM has ~33% more parameters than GRU for the same hidden size
        Both achieve comparable NILM accuracy (BiGRU slightly better
        per NILMFormer Table 2: BiGRU MAE=100.8W vs BiLSTM MAE=130.9W)

    Parameters
    ----------
    in_channels : int
        Default 6 (4 DWT + 2 temporal).
    window_size : int
        Default 480.
    n_appliances : int
        Default 5.
    dropout : float
        Default 0.1 (matches NILMFormer).
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.center = window_size // 2
        self.drop = nn.Dropout(dropout)

        # Conv1d — matching NILMFormer bilstm.py (k=4 not k=5)
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=4,
                              stride=1, padding='same', bias=True)

        # BiLSTM layers — identical to NILMFormer bilstm.py
        self.lstm1 = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

        # Dense + multi-task heads
        self.dense = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 6, 480) → powers, states, gated each (B, N_app)
        """
        x = self.conv(x)                           # (B, 16, 480)
        x = self.drop(x).permute(0, 2, 1)          # (B, 480, 16)
        x, _ = self.lstm1(x)                       # (B, 480, 128)
        x, _ = self.lstm2(self.drop(x))            # (B, 480, 256)
        center = x[:, self.center, :]              # (B, 256)
        out = self.dense(center)                   # (B, 64)
        return self.heads(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 55)
    print("BiLSTM Baseline — Self-Test")
    print("Source: NILMFormer bilstm.py (Petralia et al. KDD 2025)")
    print("=" * 55)

    model = BiLSTMBaseline()
    B = 8
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    print(f"\n[Input]  {tuple(x.shape)}")
    print(f"[Output] powers={tuple(powers.shape)}, "
          f"states={tuple(states.shape)}, gated={tuple(gated.shape)}")

    assert powers.shape == (B, N_APPLIANCES)
    assert (powers >= 0).all()
    assert (gated <= powers + 1e-6).all()

    with torch.no_grad():
        c = model.conv(x)
        p = model.drop(c).permute(0, 2, 1)
        l1, _ = model.lstm1(p)
        l2, _ = model.lstm2(model.drop(l1))
    print(f"\n[Shapes] conv={tuple(c.shape)}, "
          f"lstm1={tuple(l1.shape)}, lstm2={tuple(l2.shape)}")
    print(f"[Params] {model.count_parameters():,} | "
          f"{model.count_parameters()/1e6:.3f} MB INT8")

    print(f"\n[NILMFormer Table 2 — BiLSTM reference]")
    print(f"  MAE=130.9W | MR=0.217 | F1=0.229 | Acc=0.430")
    print(f"  Note: BiGRU (100.8W) outperforms BiLSTM (130.9W)")
    print(f"  → BiGRU better choice for deployable model")

    print(f"\n{'=' * 55}")
    print("All tests passed — BiLSTM baseline ready!")
    print(f"{'=' * 55}")
