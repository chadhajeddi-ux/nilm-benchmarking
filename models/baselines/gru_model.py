"""
gru_model.py — GRU Baseline for NILM (seq2point)
==================================================
Architecture based on Window-GRU (WGRU):
    Krystalakos et al. "Sliding Window Approach for Online Energy
    Disaggregation Using Artificial Neural Networks." HellAI 2018.

    Validated and benchmarked by:
    Virtsionis et al. "SAED: Self-Attentive Energy Disaggregation."
    Machine Learning, Springer, 2021.
    DOI: https://doi.org/10.1007/s10994-021-06106-3
    GitHub: https://github.com/Virtsionis/SelfAttentiveEnergyDisaggregator

    WGRU is the most widely used GRU baseline in NILM literature.
    It serves as one of three standard baselines (alongside S2P-CNN
    and SAED) in the Torch-NILM benchmarking framework.

Modifications from original WGRU:
    - Input: (6, 480) with 4 DWT sub-bands + 2 temporal features
      instead of (1, 50-100) raw power only
    - Unidirectional GRU (not bidirectional) — this file is the
      GRU baseline. BiLSTM baseline is a separate file.
    - Multi-task output: power + state per appliance (N heads)
    - Gated output: ŷ = p̂ × σ(ŝ_logit) for physical consistency
    - Seq2point center extraction instead of last hidden state

Architecture overview:
    Input (B, 6, 480)
        ↓  Conv1D(64, k=5) + ReLU + Dropout
        ↓  Permute → (B, 480, 64)
        ↓  GRU(hidden=128, layers=2, dropout=0.2)
        ↓  Take center timestep [:, 240, :]
        ↓  Linear(128→64) + ReLU + Dropout
        ↓  N × [power head | state head] → gated output

Why GRU instead of LSTM:
    GRU has 2 gates (reset, update) vs LSTM's 3 gates (forget,
    input, output) + cell state. Fewer parameters, comparable
    performance on NILM tasks (validated by SAED paper Table 1:
    WGRU 270K params achieves similar F1 to larger models).
    GRU is also more NPU-friendly due to simpler gate structure.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
)

# Reuse the same multi-task heads from CNN baseline
from cnn_model import MultiTaskHeads


# ============================================================
# GRU MODEL — WGRU architecture, adapted
# ============================================================

class GRUBaseline(nn.Module):
    """
    Conv1D + GRU seq2point model for multi-task NILM.

    The Conv1D layer extracts local features from the multi-channel
    DWT input. The GRU layers model temporal dependencies across
    the full 480-timestep window unidirectionally (past → future).
    The center timestep prediction combines past context from the
    forward GRU pass with the locally extracted features.

    This is the UNIDIRECTIONAL GRU baseline. For bidirectional
    temporal modeling, see bilstm_model.py or the proposed model.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 6 (4 DWT + 2 temporal).
    window_size : int
        Number of timesteps per window. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    conv_out : int
        Number of Conv1D output channels. Default 64.
    conv_kernel : int
        Conv1D kernel size. Default 5.
    gru_hidden : int
        GRU hidden units. Default 128.
    gru_layers : int
        Number of stacked GRU layers. Default 2.
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
        # Conv1D learns local patterns in the DWT sub-bands
        # before feeding into the GRU. This follows the WGRU
        # architecture where Conv precedes GRU for feature extraction.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_out, kernel_size=conv_kernel,
                      padding=conv_kernel // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Temporal modeling ----
        # GRU processes the sequence position by position,
        # maintaining a hidden state that accumulates temporal context.
        #
        # At each timestep t, the GRU computes:
        #   z_t = σ(W_z · [h_{t-1}, x_t])    — update gate
        #   r_t = σ(W_r · [h_{t-1}, x_t])    — reset gate
        #   h̃_t = tanh(W · [r_t ⊗ h_{t-1}, x_t])  — candidate
        #   h_t = (1 - z_t) ⊗ h_{t-1} + z_t ⊗ h̃_t  — output
        #
        # The update gate z_t decides how much of the past to keep.
        # The reset gate r_t decides how much of the past to forget.
        # These gates are INPUT-DEPENDENT — they adapt based on the
        # current power reading, unlike fixed Conv filters.
        self.gru = nn.GRU(
            input_size=conv_out,      # 64 features from Conv1D
            hidden_size=gru_hidden,   # 128 hidden units
            num_layers=gru_layers,    # 2 stacked GRU layers
            batch_first=True,         # input: (B, T, features)
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,      # unidirectional for this baseline
        )
        # Output: (B, T, 128) — hidden state at every timestep

        # ---- Dense compression ----
        # Reduce 128 → 64 before multi-task heads
        self.dense = nn.Sequential(
            nn.Linear(gru_hidden, 64),
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
        x : (B, 6, 480) — batch of DWT windows with temporal features

        Returns
        -------
        powers : (B, N_app) — normalized power at center timestep
        states : (B, N_app) — state logits at center timestep
        gated  : (B, N_app) — gated final prediction
        """
        # ---- Conv1D: local feature extraction ----
        # (B, 6, 480) → (B, 64, 480)
        conv_out = self.conv(x)

        # ---- Permute for GRU ----
        # GRU expects (B, T, features) but Conv1D outputs (B, features, T)
        # (B, 64, 480) → (B, 480, 64)
        conv_out = conv_out.permute(0, 2, 1)

        # ---- GRU: temporal modeling ----
        # (B, 480, 64) → (B, 480, 128)
        gru_out, _ = self.gru(conv_out)
        # _ is the final hidden state — we don't need it for seq2point

        # ---- Seq2point: take center timestep ----
        # (B, 480, 128) → (B, 128)
        center_features = gru_out[:, self.center, :]

        # ---- Dense compression ----
        # (B, 128) → (B, 64)
        dense_out = self.dense(center_features)

        # ---- Multi-task heads ----
        # (B, 64) → powers (B,5), states (B,5), gated (B,5)
        powers, states, gated = self.heads(dense_out)

        return powers, states, gated

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# ENTRY POINT — Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("GRU Baseline — Self-Test")
    print("Architecture: WGRU (Krystalakos et al. 2018)")
    print("=" * 55)

    model = GRUBaseline(
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
    )

    # ---- Forward pass with dummy batch ----
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
        after_conv = model.conv(x)
        after_perm = after_conv.permute(0, 2, 1)
        after_gru, _ = model.gru(after_perm)
        center = after_gru[:, WINDOW_SIZE // 2, :]
        after_dense = model.dense(center)
    print(f"  After Conv1D:     {tuple(after_conv.shape)}")
    print(f"  After permute:    {tuple(after_perm.shape)}")
    print(f"  After GRU:        {tuple(after_gru.shape)}")
    print(f"  After center:     {tuple(center.shape)}")
    print(f"  After dense:      {tuple(after_dense.shape)}")

    print(f"\n{'=' * 55}")
    print("All tests passed — GRU baseline ready!")
    print(f"{'=' * 55}")
