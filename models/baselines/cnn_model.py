"""
cnn_model.py : CNN Baseline for NILM (seq2point)
==================================================
Architecture adapted from:
    Zhang et al. "Sequence-to-Point Learning with Neural Networks
    for Non-Intrusive Load Monitoring." AAAI 2018.
    DOI: https://doi.org/10.1609/aaai.v32i1.11873
    Official code: https://github.com/MingjunZhong/seq2point-nilm

Modifications from original:
    - Input: (6, 480) with 4 DWT sub-bands + 2 temporal features
      instead of (1, 599) raw power only
    - Multi-task output: power + state per appliance (N heads)
      instead of single scalar output per model
    - Gated output: ŷ = p̂ × σ(ŝ_logit) for physical consistency
    - Focal Loss compatible: raw logits returned for state head
    - Conv1D instead of Conv2D (cleaner for 1D time series in PyTorch)

Architecture overview (your input):
    Input (B, 6, 480)
        ↓  Conv1D(30, k=10) + ReLU
        ↓  Conv1D(30, k=8)  + ReLU
        ↓  Conv1D(40, k=6)  + ReLU
        ↓  Conv1D(50, k=5)  + ReLU
        ↓  Conv1D(50, k=5)  + ReLU
        ↓  Take center timestep [:, :, 240]
        ↓  Linear(50→1024)  + ReLU + Dropout
        ↓  N × [Linear(1024→1) power | Linear(1024→1) state]
    Output:
        powers : (B, N_app)   — scaled power [0, 1]
        states : (B, N_app)   — raw logits for Focal Loss
        gated  : (B, N_app)   — p̂ × σ(ŝ) physically consistent

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
    APPLIANCE_NAMES,
)


# ============================================================
# HELPER — Shared Multi-Task Output Heads
# ============================================================

class MultiTaskHeads(nn.Module):
    """
    N parallel pairs of output heads — one pair per appliance.

    Each pair:
        power_head : Linear → ReLU → scalar in [0, 1] (normalized Watts)
        state_head : Linear → scalar logit (no activation — use with
                     BCEWithLogitsLoss or Focal Loss)

    Gated output:
        ŷ = p̂ × sigmoid(ŝ_logit)
        When model predicts OFF (ŝ_logit << 0), sigmoid → 0,
        so ŷ → 0 regardless of p̂. Physical consistency enforced.

    Parameters
    ----------
    in_features : int
        Size of the input feature vector per sample.
    n_appliances : int
        Number of target appliances (default from config).
    """

    def __init__(self, in_features: int, n_appliances: int = N_APPLIANCES):
        super().__init__()
        self.n_appliances = n_appliances

        # One power head + one state head per appliance
        # Using ModuleList so PyTorch tracks all parameters correctly
        self.power_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 1),
                nn.ReLU(),              # power is always ≥ 0
            )
            for _ in range(n_appliances)
        ])

        self.state_heads = nn.ModuleList([
            nn.Linear(in_features, 1)  # raw logit — no activation
            for _ in range(n_appliances)
        ])

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, in_features)

        Returns
        -------
        powers : (B, N_app)  — normalized power predictions [0, 1]
        states : (B, N_app)  — raw state logits for loss computation
        gated  : (B, N_app)  — p̂ × σ(ŝ) final output
        """
        powers = torch.cat(
            [head(x) for head in self.power_heads], dim=1
        )  # (B, N_app)

        states = torch.cat(
            [head(x) for head in self.state_heads], dim=1
        )  # (B, N_app)

        # Physical gating — from Mamba-ECA-UNet (Fan et al., PSETC 2025)
        gated = powers * torch.sigmoid(states)  # (B, N_app)

        return powers, states, gated


# ============================================================
# CNN MODEL — Zhang et al. AAAI 2018, adapted
# ============================================================

class CNNBaseline(nn.Module):
    """
    5-layer Conv1D seq2point model for multi-task NILM.

    Follows the architecture of Zhang et al. (AAAI 2018) exactly,
    with the modifications documented in the module docstring above.

    The CNN learns hierarchical local features from the aggregate
    power signal — from broad patterns (kernel=10, ~1 minute) to
    fine-grained details (kernel=5, ~30 seconds). The seq2point
    strategy gives the model bilateral context: 240 timesteps of
    past AND 240 timesteps of future for each center prediction.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 6 (4 DWT + 2 temporal).
    window_size : int
        Number of timesteps per window. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    dropout : float
        Dropout probability after the dense layer. Default 0.5.
        Higher than typical (0.2-0.3) because the dense layer
        has 1024 units — significant risk of overfitting.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.window_size = window_size
        self.center = window_size // 2  # seq2point: predict center only

        # ---- 5 Conv1D blocks (Zhang et al. architecture) ----
        # padding='same' keeps temporal length = 480 throughout
        # This is cleaner than manual padding calculation and
        # matches the original paper's "same" padding behavior.
        self.conv_blocks = nn.Sequential(
            # Block 1: broad local patterns (~1 minute view)
            nn.Conv1d(in_channels, 30, kernel_size=10, padding='same'),
            nn.ReLU(),

            # Block 2: medium-broad patterns
            nn.Conv1d(30, 30, kernel_size=8, padding='same'),
            nn.ReLU(),

            # Block 3: medium patterns
            nn.Conv1d(30, 40, kernel_size=6, padding='same'),
            nn.ReLU(),

            # Block 4: fine patterns (~30 second view)
            nn.Conv1d(40, 50, kernel_size=5, padding='same'),
            nn.ReLU(),

            # Block 5: finest patterns (same as block 4 — depth matters)
            nn.Conv1d(50, 50, kernel_size=5, padding='same'),
            nn.ReLU(),
        )
        # Output shape after conv_blocks: (B, 50, 480)

        # ---- Dense block ----
        # Take center timestep → flatten 50 features → expand to 1024
        self.dense = nn.Sequential(
            nn.Linear(50, 1024),    # 50 features from center timestep
            nn.ReLU(),
            nn.Dropout(dropout),    # 0.5 dropout as in original paper
        )

        # ---- Multi-task output heads ----
        self.heads = MultiTaskHeads(
            in_features=1024,
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
        # (B, 6, 480) → (B, 50, 480) through 5 conv blocks
        features = self.conv_blocks(x)

        # Seq2point: take ONLY the center timestep
        # (B, 50, 480) → (B, 50)
        center_features = features[:, :, self.center]

        # Dense expansion: (B, 50) → (B, 1024)
        dense_out = self.dense(center_features)

        # Multi-task heads: (B, 1024) → powers, states, gated
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
    print("CNN Baseline — Self-Test")
    print("Architecture: Zhang et al. AAAI 2018 (adapted)")
    print("=" * 55)

    model = CNNBaseline(
        in_channels=INPUT_CHANNELS,    # 6
        window_size=WINDOW_SIZE,       # 480
        n_appliances=N_APPLIANCES,     # 5
        dropout=0.5,
    )

    # ---- Forward pass with dummy batch ----
    B = 8  # batch size
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    print(f"\n[Input]")
    print(f"  x shape:      {tuple(x.shape)}")

    print(f"\n[Output]")
    print(f"  powers shape: {tuple(powers.shape)}")
    print(f"  states shape: {tuple(states.shape)}")
    print(f"  gated  shape: {tuple(gated.shape)}")

    print(f"\n[Checks]")
    assert powers.shape == (B, N_APPLIANCES), "powers shape wrong"
    assert states.shape == (B, N_APPLIANCES), "states shape wrong"
    assert gated.shape  == (B, N_APPLIANCES), "gated shape wrong"
    assert (powers >= 0).all(), "powers must be >= 0 (ReLU)"
    assert (gated >= 0).all(),  "gated must be >= 0"
    assert (gated <= powers + 1e-6).all(), \
        "gated must be <= powers (sigmoid gate ≤ 1)"
    print(f"  All output shapes correct ✓")
    print(f"  powers >= 0: ✓")
    print(f"  gated <= powers: ✓")

    print(f"\n[Model complexity]")
    n_params = model.count_parameters()
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Approx. size (INT8):  {n_params / 1e6:.3f} MB")

    print(f"\n[Per-layer output shapes]")
    # trace manually through the model
    with torch.no_grad():
        after_conv = model.conv_blocks(x)
        center = after_conv[:, :, WINDOW_SIZE // 2]
        after_dense = model.dense(center)
    print(f"  After conv_blocks:    {tuple(after_conv.shape)}")
    print(f"  After center slice:   {tuple(center.shape)}")
    print(f"  After dense:          {tuple(after_dense.shape)}")

    print(f"\n{'=' * 55}")
    print("All tests passed — CNN baseline ready!")
    print(f"{'=' * 55}")
