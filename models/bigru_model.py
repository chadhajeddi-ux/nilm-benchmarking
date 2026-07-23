"""
bigru_model.py — BiGRU Baseline for NILM (seq2point)
======================================================
Architecture based on:
    Precioso Garcelán, D.; Gomez-Ullate, D.
    "Bidirectional GRU with Convolutional Layers for NILM."
    Referenced as BiGRU baseline in:
    Petralia et al. "NILMFormer: Non-Intrusive Load Monitoring
    that Accounts for Non-Stationarity." KDD 2025.
    arXiv: https://arxiv.org/abs/2506.05880
    GitHub: https://github.com/adrienpetralia/NILMFormer

Supporting reference:
    Xuan et al. "Load Energy Decomposition Algorithm Based on
    Improved Bidirectional Transformer Combined With Time-Sensing
    Self-Attention." IEEE Access, Vol. 12, pp. 75625-75639, 2024.
    DOI: https://doi.org/10.1109/ACCESS.2024.3373801

Why BiGRU as a separate baseline from GRU:
    BiGRU processes each window in BOTH directions simultaneously:
        Forward  GRU: h_fwd  [t=0 → t=480] — past context
        Backward GRU: h_bwd  [t=480 → t=0] — future context
    At the center timestep (t=240), the concatenated hidden state
    contains information from ALL 480 positions — the full bilateral
    context that seq2point was designed to exploit.

    Unidirectional GRU at t=240 only sees positions 0→240.
    BiGRU at t=240 sees positions 0→480 (full window).
    This is the key improvement over the GRU baseline.

    NILMFormer Table 2 reports BiGRU MAE values on UK-DALE:
        Dishwasher: 27.2W, Fridge: 36.7W, Kettle: 11.6W,
        Microwave: 12.4W, Washing Machine: 21.1W
    These are your reference benchmark numbers.

Architecture overview:
    Input (B, 6, 480)
        ↓  Conv1D(64, k=5) + ReLU + Dropout
        ↓  Permute → (B, 480, 64)
        ↓  BiGRU(hidden=128, layers=2, bidirectional=True)
           Output: (B, 480, 256)  ← 128 fwd + 128 bwd concatenated
        ↓  Take center timestep [:, 240, :]  → (B, 256)
        ↓  Linear(256→64) + ReLU + Dropout
        ↓  N × [power head | state head] → gated output

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
)

from cnn_model import MultiTaskHeads


# ============================================================
# BiGRU MODEL
# ============================================================

class BiGRUBaseline(nn.Module):
    """
    Conv1D + Bidirectional GRU seq2point model for multi-task NILM.

    The critical difference from GRUBaseline:
        GRU  → at center t=240: sees only positions 0..240 (past only)
        BiGRU → at center t=240: sees ALL positions 0..480 (full window)

    This bilateral context is exactly what seq2point learning requires
    — the model has equal access to past AND future around each target
    prediction point. At the center of a 48-minute window, BiGRU knows
    what happened 24 minutes before AND 24 minutes after — critical
    for disambiguating appliances with similar power signatures but
    different temporal contexts.

    The forward pass reads t=0→480. The backward pass reads t=480→0.
    Their hidden states at each position t are concatenated:
        h_t = [h_fwd_t ; h_bwd_t]  shape: (B, 256) at center

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 6 (4 DWT + 2 temporal).
    window_size : int
        Number of timesteps. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    conv_out : int
        Conv1D output channels. Default 64.
    conv_kernel : int
        Conv1D kernel size. Default 5.
    gru_hidden : int
        GRU hidden size PER DIRECTION. Default 128.
        Total output = 2 × gru_hidden = 256.
    gru_layers : int
        Number of stacked BiGRU layers. Default 2.
    dropout : float
        Dropout probability. Default 0.2.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        conv_out: int = 64,
        conv_kernel: int = 5,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.window_size = window_size
        self.center = window_size // 2
        self.gru_hidden = gru_hidden

        # ---- Local feature extraction ----
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_out, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Bidirectional temporal modeling ----
        # bidirectional=True: runs TWO GRU instances in parallel
        #   forward GRU:  processes x[0], x[1], ..., x[479]
        #   backward GRU: processes x[479], x[478], ..., x[0]
        # Output at each t: [h_fwd_t ; h_bwd_t] → size 2 × gru_hidden
        self.bigru = nn.GRU(
            input_size=conv_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,              # KEY DIFFERENCE from GRU
        )
        # Output shape: (B, T, 2 × gru_hidden) = (B, 480, 256)

        # ---- Dense compression ----
        # 256 → 64: reduce before multi-task heads
        self.dense = nn.Sequential(
            nn.Linear(2 * gru_hidden, 64),   # 256 → 64
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Multi-task output heads ----
        self.heads = MultiTaskHeads(
            in_features=64,
            n_appliances=n_appliances,
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 6, 480)

        Returns
        -------
        powers : (B, N_app) — normalized power at center timestep
        states : (B, N_app) — raw logits at center timestep
        gated  : (B, N_app) — p̂ × σ(ŝ) gated prediction
        """
        # (B, 6, 480) → (B, 64, 480)
        conv_out = self.conv(x)

        # (B, 64, 480) → (B, 480, 64) for BiGRU
        conv_out = conv_out.permute(0, 2, 1)

        # (B, 480, 64) → (B, 480, 256)
        # Both directions process simultaneously — full bilateral context
        bigru_out, _ = self.bigru(conv_out)

        # Seq2point: center position has FULL bilateral context
        # (B, 480, 256) → (B, 256)
        center_features = bigru_out[:, self.center, :]

        # (B, 256) → (B, 64)
        dense_out = self.dense(center_features)

        # (B, 64) → powers, states, gated
        powers, states, gated = self.heads(dense_out)

        return powers, states, gated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# ENTRY POINT — Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BiGRU Baseline — Self-Test")
    print("Reference: Precioso Garcelán & Gomez-Ullate 2023")
    print("Benchmarked by: NILMFormer (Petralia et al. KDD 2025)")
    print("=" * 55)

    model = BiGRUBaseline(
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
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
    print(f"  All shapes correct ✓")
    print(f"  powers >= 0: ✓")
    print(f"  gated <= powers: ✓")

    print(f"\n[Model complexity]")
    n_params = model.count_parameters()
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Approx. size (INT8):  {n_params / 1e6:.3f} MB")

    print(f"\n[Per-layer output shapes]")
    with torch.no_grad():
        after_conv  = model.conv(x)
        after_perm  = after_conv.permute(0, 2, 1)
        after_bigru, _ = model.bigru(after_perm)
        center      = after_bigru[:, WINDOW_SIZE // 2, :]
        after_dense = model.dense(center)
    print(f"  After Conv1D:     {tuple(after_conv.shape)}")
    print(f"  After permute:    {tuple(after_perm.shape)}")
    print(f"  After BiGRU:      {tuple(after_bigru.shape)}")
    print(f"  After center:     {tuple(center.shape)}")
    print(f"  After dense:      {tuple(after_dense.shape)}")

    print(f"\n[BiGRU vs GRU — key difference]")
    print(f"  GRU  center hidden: 128  (past only: positions 0→240)")
    print(f"  BiGRU center hidden: 256 (full: positions 0→480)")
    print(f"  Bilateral context advantage: ✓")

    print(f"\n[NILMFormer reference numbers — UK-DALE]")
    ref = {
        "Dishwasher": (27.2, 0.408),
        "Fridge": (36.7, 0.364),
        "Kettle": (11.6, 0.618),
        "Microwave": (12.4, 0.040),
        "Washing Machine": (21.1, 0.086),
    }
    print(f"  {'Appliance':<18} {'MAE (W)':<10} {'MR':<8}")
    print(f"  {'-'*36}")
    for app, (mae, mr) in ref.items():
        print(f"  {app:<18} {mae:<10.1f} {mr:<8.3f}")
    print(f"  (Your BiGRU+DWT model should beat these numbers)")

    print(f"\n{'=' * 55}")
    print("All tests passed — BiGRU baseline ready!")
    print(f"{'=' * 55}")
