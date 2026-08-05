"""
bigru_model.py — BiGRU Baseline for NILM
==========================================
EXACT architecture from NILMFormer official source code:
    Petralia et al. "NILMFormer: Non-Intrusive Load Monitoring
    that Accounts for Non-Stationarity." KDD 2025.
    arXiv: https://arxiv.org/abs/2506.05880
    GitHub: https://github.com/adrienpetralia/NILMFormer
    Source: nilmformer/baselines/BiGRU.py (c)2025 EDF, Adrien Petralia)

Original architecture (faithfully reproduced):
    Conv1d(c_in->16, k=5, same) -> Dropout
    Conv1d(16->8, k=5, same)    -> Dropout
    BiGRU(8->64, bidirectional) -> output 128
    BiGRU(128->128, bidirectional) -> output 256
    Dense(256->64)
    Separate heads for power regression and state classification

Our adaptations for fair benchmarking:
    1. Input: (B, 6, 480) with 4 DWT sub-bands + 2 temporal features
       instead of (B, 1, T) raw power only
    2. Seq2point: predict CENTER timestep only (position 240)
       instead of seq2seq (all timesteps)
    3. Multi-task heads: power + state per appliance (N=5)
       instead of single appliance output
    4. Gated output: y_hat = p_hat x sigmoid(s_logit)
    5. Focal Loss compatible: raw logits returned for state head

Reference numbers from NILMFormer Table 2 (UK-DALE, BiGRU row):
    MAE: 100.8W | MR: 0.314 | F1: 0.253 | Accuracy: 0.491
    Recall: 0.962 | Precision: 0.177
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
)

from cnn_model import MultiTaskHeads


class BiGRUBaseline(nn.Module):
    """
    Bidirectional GRU baseline for multi-task NILM.

    Faithfully implements the BiGRU from NILMFormer (Petralia et al.,
    KDD 2025) with three adaptations: 6-channel DWT input, seq2point
    center extraction, and N-appliance multi-task output.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 6 (4 DWT + 2 temporal).
    window_size : int
        Number of timesteps. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    dropout : float
        Dropout probability. Default 0.1 (matches NILMFormer).
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

        # Conv1D layers - matching NILMFormer exactly
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5,
                               stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=5,
                               stride=1, padding='same', bias=True)

        # BiGRU layers - identical to NILMFormer
        self.gru1 = nn.GRU(8, 64, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)

        # Dense projection - matching NILMFormer
        self.dense = nn.Linear(256, 64)

        # Multi-task heads - our adaptation
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 6, 480)
        Returns: powers, states, gated — each (B, N_app)
        """
        # Conv1D feature extraction
        x = self.conv1(x)                          # (B, 16, 480)
        x = self.conv2(self.drop(x))               # (B, 8, 480)

        # Permute for GRU
        x = self.drop(x).permute(0, 2, 1)          # (B, 480, 8)

        # BiGRU temporal modeling
        x = self.gru1(x)[0]                        # (B, 480, 128)
        x = self.gru2(self.drop(x))[0]             # (B, 480, 256)

        # Seq2point: center timestep only
        center = x[:, self.center, :]              # (B, 256)

        # Dense projection
        out = self.drop(self.dense(self.drop(center)))  # (B, 64)

        # Multi-task heads
        powers, states, gated = self.heads(out)

        return powers, states, gated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("BiGRU Baseline - Self-Test")
    print("Architecture: NILMFormer official (Petralia et al. KDD 2025)")
    print("=" * 60)

    model = BiGRUBaseline(
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
        dropout=0.1,
    )

    B = 8
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    print(f"\n[Input]")
    print(f"  x shape:      {tuple(x.shape)}")
    print(f"\n[Output]")
    print(f"  powers shape: {tuple(powers.shape)}")
    print(f"  states shape: {tuple(states.shape)}")
    print(f"  gated  shape: {tuple(gated.shape)}")

    print(f"\n[Checks]")
    assert powers.shape == (B, N_APPLIANCES)
    assert states.shape == (B, N_APPLIANCES)
    assert gated.shape  == (B, N_APPLIANCES)
    assert (powers >= 0).all()
    assert (gated >= 0).all()
    assert (gated <= powers + 1e-6).all()
    print(f"  All shapes correct")
    print(f"  powers >= 0")
    print(f"  gated <= powers")

    print(f"\n[Model complexity]")
    n_params = model.count_parameters()
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Approx. size (INT8):  {n_params / 1e6:.3f} MB")

    print(f"\n[Per-layer output shapes]")
    with torch.no_grad():
        c1 = model.conv1(x)
        c2 = model.conv2(model.drop(c1))
        p  = model.drop(c2).permute(0, 2, 1)
        g1 = model.gru1(p)[0]
        g2 = model.gru2(model.drop(g1))[0]
        ct = g2[:, WINDOW_SIZE // 2, :]
        d  = model.drop(model.dense(model.drop(ct)))
    print(f"  After conv1:   {tuple(c1.shape)}")
    print(f"  After conv2:   {tuple(c2.shape)}")
    print(f"  After permute: {tuple(p.shape)}")
    print(f"  After BiGRU1:  {tuple(g1.shape)}")
    print(f"  After BiGRU2:  {tuple(g2.shape)}")
    print(f"  After center:  {tuple(ct.shape)}")
    print(f"  After dense:   {tuple(d.shape)}")

    print(f"\n[NILMFormer Table 2 - BiGRU reference numbers]")
    print(f"  MAE: 100.8W | MR: 0.314 | F1: 0.253 | Acc: 0.491")
    print(f"  Recall: 0.962 (high) | Precision: 0.177 (low)")
    print(f"  -> Focal Loss will improve precision and F1")

    print(f"\n[Our improvements over original BiGRU]")
    print(f"  Input: 6ch DWT+temporal vs 1ch raw power")
    print(f"  Strategy: seq2point vs seq2seq")
    print(f"  Output: {N_APPLIANCES} appliances vs 1")
    print(f"  Loss: Focal+SmoothL1+Gated vs MSE+BCE")

    print(f"\n{'=' * 60}")
    print("All tests passed - BiGRU baseline ready!")
    print(f"{'=' * 60}")