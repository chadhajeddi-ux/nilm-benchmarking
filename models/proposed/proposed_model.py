"""
proposed_model.py  : BiWaveNILM  for NILM
====================================================================
The proposed multi-task NILM model for embedded deployment on
STM32MP2 NPU (Neural-ART, INT8 quantization).

Novel contributions:
    1. DWT preprocessing: 4 sub-bands capture physically invariant
       frequency signatures for cross-dataset generalization
       (Luo et al., J.Supercomputing 2023)
    2. DepthwiseSeparable Conv1D: 3.5× fewer MACs than standard Conv
       (Howard et al., MobileNet 2017)
    3. Parallel BiGRU ‖ BiTCN: complementary temporal modeling —
       BiGRU for adaptive memory, BiTCN for multi-scale local patterns
    4. Lite-CBAM(T): lightweight channel + temporal attention
       (Woo et al., ECCV 2018)
    5. DyT normalization: batch-size-independent, replaces BatchNorm
       (Ma et al., arXiv:2503.10622, 2025)
    6. Global residual skip: gradient highway from input to deep layers
    7. Gated multi-task output: ŷ = p̂ × σ(ŝ) for physical consistency
       (Fan et al., Mamba-ECA-UNet, IEEE PSETC 2025)

Architecture:
    Input (B, 6, 480) — 4 DWT sub-bands + 2 temporal features
        ↓ Dual DWSConv (6→32→64) + DyT + GELU
        ↓ ┌─ BiGRU(64→128, bidir) → (B, 480, 256) + DyT
        ↓ └─ BiTCN(d=[1,2,4,8]) + residual → (B, 480, 256) + DyT
        ↓ Concatenate → (B, 480, 512) + Global residual skip
        ↓ Lite-CBAM: Channel(512→32→512) + Temporal(k=7) + DyT
        ↓ Seq2point center [:, 240, :] → (B, 512)
        ↓ Dense(512→128) + ReLU + Dropout
        ↓ MultiTaskHeads ×5 → powers, states, gated

Target deployment constraints (r):
    Latency  ≤ 15 ms    on STM32MP2 NPU
    Energy   ≤ 500 mW
    RAM      ≤ 2 MB
    F1       ≥ 0.80
    MAE      ≤ 50 W

Estimated complexity:
    Parameters: ~355K
    INT8 size:  ~0.355 MB (well within 2 MB budget)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "models" / "baselines"))
sys.path.insert(0, str(_root / "models"))
from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
)
from cnn_model import MultiTaskHeads


# ============================================================
# COMPONENT 1: Dynamic Tanh Normalization (DyT)
# ============================================================
# Reference: Ma et al. "Transformers without Normalization"
# arXiv:2503.10622, 2025.
#
# Replaces BatchNorm/LayerNorm with a learned tanh-based
# normalization that has ZERO batch dependency.
# At inference with batch=1 on STM32MP2, DyT works identically
# to training — unlike BatchNorm which uses running statistics.
#
# Formula: y = γ · tanh(α · x) + β
#   α: learned scalar controlling input scale (init=0.5)
#   γ: learned per-channel scale (init=1.0)
#   β: learned per-channel shift (init=0.0)

class DyT(nn.Module):
    """
    Dynamic Tanh normalization — batch-size-independent.

    Parameters
    ----------
    num_features : int
        Number of features (channels) to normalize.
    channel_dim : int
        Which dimension contains the channels.
        Use 1 for Conv1D output (B, C, T).
        Use -1 for GRU output (B, T, C).
    init_alpha : float
        Initial value for the learnable scale parameter α.
    """

    def __init__(self, num_features: int, channel_dim: int = 1,
                 init_alpha: float = 0.5):
        super().__init__()
        self.channel_dim = channel_dim
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape gamma and beta to broadcast correctly
        shape = [1] * x.ndim
        shape[self.channel_dim] = -1
        g = self.gamma.view(*shape)
        b = self.beta.view(*shape)
        return g * torch.tanh(self.alpha * x) + b


# ============================================================
# COMPONENT 2: Depthwise Separable Conv1D
# ============================================================
# Reference: Howard et al. "MobileNets: Efficient Convolutional
# Neural Networks for Mobile Vision Applications." 2017.
#
# Splits a standard Conv1D into two steps:
#   Step 1 (Depthwise): one filter per input channel (groups=in_ch)
#   Step 2 (Pointwise): 1×1 Conv mixing across channels
# Result: ~3.5× fewer multiply-accumulate operations (MACs)
# Critical for STM32MP2 NPU where MAC budget is limited.

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for 1D time series.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size for the depthwise convolution. Default 5.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2

        # Depthwise: each input channel gets its own filter
        # groups=in_channels means no cross-channel interaction
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # KEY: one filter per channel
            bias=False,
        )
        # Pointwise: 1×1 conv mixes information across channels
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, T) → depthwise → (B, in_ch, T) → pointwise → (B, out_ch, T)
        return self.pointwise(self.depthwise(x))


# ============================================================
# COMPONENT 3: TCN Residual Block (with GELU + DyT)
# ============================================================
# Reference: Bai et al. "An Empirical Evaluation of Generic
# Convolutional and Recurrent Networks for Sequence Modeling." 2018.
#
# Non-causal dilated convolution with symmetric padding —
# each block sees both past and future context.
# Residual skip connection ensures stable gradient flow
# through 4 stacked dilated layers.

class TCNResBlock(nn.Module):
    """
    Single TCN block: dilated Conv1D + GELU + DyT + residual skip.

    The dilation parameter controls the temporal receptive field:
        d=1: sees 3 consecutive timesteps (18 seconds at 6s sampling)
        d=2: sees timesteps 2 apart (36 seconds)
        d=4: sees timesteps 4 apart (72 seconds)
        d=8: sees timesteps 8 apart (144 seconds = 2.4 minutes)

    Parameters
    ----------
    channels : int
        Number of channels (input = output for residual skip).
    kernel_size : int
        Convolution kernel size. Default 3.
    dilation : int
        Dilation factor. Default 1.
    dropout : float
        Dropout rate. Default 0.1.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        # Symmetric padding for non-causal (sees both past and future)
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dyt = DyT(channels, channel_dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        out = self.conv(x)          # (B, C, T) — dilated conv
        out = self.dyt(out)         # (B, C, T) — DyT normalization
        out = F.gelu(out)           # GELU: preserves near-zero gradients
        out = self.dropout(out)
        return out + residual       # Residual skip connection


# ============================================================
# COMPONENT 4: Lite-CBAM(T) — Channel + Temporal Attention
# ============================================================
# Reference: Woo et al. "CBAM: Convolutional Block Attention
# Module." ECCV 2018.
#
# Two sequential attention mechanisms:
#   Channel attention: "WHICH features matter?"
#     GAP across time → FC bottleneck → sigmoid weights per channel
#   Temporal attention: "WHEN in the window matters?"
#     Pool across channels → Conv1D → sigmoid weights per timestep
#
# "Lite" because we use a single-path GAP (not GAP+GMP as in
# original CBAM) and a smaller bottleneck , reducing parameters
# for embedded deployment.

class LiteCBAM(nn.Module):
    """
    Lightweight CBAM with channel and temporal attention.

    Parameters
    ----------
    channels : int
        Number of input channels (feature dimension).
    reduction : int
        Bottleneck reduction ratio for channel attention.
        512 / 16 = 32 hidden units. Default 16.
    temporal_kernel : int
        Kernel size for temporal attention Conv1D.
        k=7 covers ~42 seconds of context. Default 7.
    """

    def __init__(self, channels: int, reduction: int = 16,
                 temporal_kernel: int = 7):
        super().__init__()

        # ---- Channel attention: "which features matter?" ----
        self.channel_attn = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

        # ---- Temporal attention: "when in the window matters?" ----
        # MaxPool + AvgPool across channels → 2 channels
        # Conv1D(2→1) → sigmoid weights per timestep
        self.temporal_conv = nn.Conv1d(
            2, 1,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C) — sequence format from BiGRU/BiTCN

        Returns
        -------
        (B, T, C) — same shape, attention-weighted
        """
        # ---- Channel attention ----
        # GAP across time: (B, T, C) → (B, C)
        gap = x.mean(dim=1)
        # FC bottleneck + sigmoid: (B, C) → (B, C)
        ch_weights = self.channel_attn(gap)
        # Apply: (B, T, C) × (B, 1, C)
        x = x * ch_weights.unsqueeze(1)

        # ---- Temporal attention ----
        # Pool across channels: max and mean
        t_max = x.max(dim=-1, keepdim=True).values   # (B, T, 1)
        t_avg = x.mean(dim=-1, keepdim=True)          # (B, T, 1)
        # Concatenate: (B, T, 2)
        t_cat = torch.cat([t_max, t_avg], dim=-1)
        # Conv1D expects (B, C, T): permute → conv → permute back
        t_cat = t_cat.permute(0, 2, 1)                # (B, 2, T)
        t_weights = torch.sigmoid(self.temporal_conv(t_cat))  # (B, 1, T)
        t_weights = t_weights.permute(0, 2, 1)        # (B, T, 1)
        # Apply: (B, T, C) × (B, T, 1)
        x = x * t_weights

        return x


# ============================================================
# MAIN MODEL: BiWaveNILM
# ============================================================

class BiWaveNILM(nn.Module):
    """
    Proposed multi-task NILM model for embedded NPU deployment.

    Combines complementary temporal modeling (BiGRU for adaptive
    memory + BiTCN for multi-scale local patterns) with efficient
    depthwise-separable convolution, lightweight attention, and
    batch-independent normalization ,  all designed to satisfy
    STM32MP2 NPU deployment constraints.

    Parameters
    ----------
    in_channels : int
        Input channels. Default 6 (4 DWT + 2 temporal).
    window_size : int
        Sequence length. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    dws_channels : list
        Output channels for dual DWSConv. Default [32, 64].
    gru_hidden : int
        BiGRU hidden size per direction. Default 128.
        Total BiGRU output = 2 × gru_hidden = 256.
    tcn_channels : int
        BiTCN internal channels. Default 64.
    tcn_dilations : list
        Dilation factors for BiTCN blocks. Default [1, 2, 4, 8].
    cbam_reduction : int
        CBAM channel attention bottleneck ratio. Default 16.
    dropout : float
        Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        dws_channels: list = [32, 64],
        gru_hidden: int = 128,
        tcn_channels: int = 64,
        tcn_dilations: list = [1, 2, 4, 8],
        cbam_reduction: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.window_size = window_size
        self.center = window_size // 2  # seq2point index
        self.gru_hidden = gru_hidden

        # Total output of each parallel branch
        branch_out = 2 * gru_hidden  # 256 (BiGRU bidirectional)
        merged_dim = 2 * branch_out  # 512 (after concatenation)

        # ====================================================
        # Block 1: Dual Depthwise Separable Conv1D
        # ====================================================
        # Progressive feature enrichment: 6 → 32 → 64
        self.dws_conv1 = DepthwiseSeparableConv1d(
            in_channels, dws_channels[0], kernel_size=5)
        self.dyt1 = DyT(dws_channels[0], channel_dim=1)

        self.dws_conv2 = DepthwiseSeparableConv1d(
            dws_channels[0], dws_channels[1], kernel_size=5)
        self.dyt2 = DyT(dws_channels[1], channel_dim=1)

        self.drop1 = nn.Dropout(dropout)

        # ====================================================
        # Global Residual Skip: 64 → 512 projection
        # ====================================================
        # Gradient highway from input features to deep layers.
        # 1×1 Conv projects 64 channels to 512 to match the
        # concatenated branch output dimension.
        self.skip_proj = nn.Conv1d(
            dws_channels[1], merged_dim, kernel_size=1)

        # ====================================================
        # Block 2A: BiGRU Branch — Adaptive Temporal Memory
        # ====================================================
        # Single-layer BiGRU with 128 hidden per direction.
        # At center t=240: full bilateral context from all 480
        # positions via forward + backward passes.
        self.bigru = nn.GRU(
            input_size=dws_channels[1],   # 64
            hidden_size=gru_hidden,       # 128
            batch_first=True,
            bidirectional=True,           # output: 256
        )
        self.dyt_gru = DyT(branch_out, channel_dim=-1)  # (B, T, 256)

        # ====================================================
        # Block 2B: BiTCN Branch — Multi-Scale Local Patterns
        # ====================================================
        # 4 dilated Conv blocks with residual skip connections.
        # Non-causal symmetric padding: each block sees both
        # past and future — no directional bias.
        # Dilations [1,2,4,8] give receptive fields of
        # 18s, 36s, 72s, 144s at 6-second sampling.
        self.tcn_blocks = nn.ModuleList([
            TCNResBlock(tcn_channels, kernel_size=3,
                        dilation=d, dropout=dropout)
            for d in tcn_dilations
        ])
        # Project 64 → 256 to match BiGRU output dimension
        self.tcn_proj = nn.Conv1d(
            tcn_channels, branch_out, kernel_size=1)
        self.dyt_tcn = DyT(branch_out, channel_dim=1)  # (B, 256, T)

        # ====================================================
        # Block 3: Lite-CBAM(T) Attention
        # ====================================================
        # Channel attention: "which of the 512 features matter?"
        # Temporal attention: "which of the 480 timesteps to focus on?"
        self.cbam = LiteCBAM(
            channels=merged_dim,
            reduction=cbam_reduction,
            temporal_kernel=7,
        )
        self.dyt_cbam = DyT(merged_dim, channel_dim=-1)  # (B, T, 512)

        # ====================================================
        # Block 4: Dense + Multi-Task Output Heads
        # ====================================================
        self.dense = nn.Sequential(
            nn.Linear(merged_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(
            in_features=128,
            n_appliances=n_appliances,
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 6, 480) — DWT sub-bands + temporal features

        Returns
        -------
        powers : (B, N_app) — normalized power [0, 1]
        states : (B, N_app) — raw logits for Focal Loss
        gated  : (B, N_app) — p̂ × σ(ŝ) final prediction
        """
        # ---- Block 1: Dual DWSConv ----
        # (B, 6, 480) → (B, 32, 480) → (B, 64, 480)
        x = F.gelu(self.dyt1(self.dws_conv1(x)))
        x = self.drop1(F.gelu(self.dyt2(self.dws_conv2(x))))

        # ---- Save for global residual skip ----
        # (B, 64, 480) → project → (B, 512, 480) → (B, 480, 512)
        skip = self.skip_proj(x).permute(0, 2, 1)

        # ---- Branch A: BiGRU ----
        # (B, 64, 480) → permute → (B, 480, 64) → BiGRU → (B, 480, 256)
        gru_in = x.permute(0, 2, 1)
        gru_out, _ = self.bigru(gru_in)
        gru_out = self.dyt_gru(gru_out)  # (B, 480, 256)

        # ---- Branch B: BiTCN ----
        # (B, 64, 480) → 4× dilated Conv + residual → (B, 64, 480)
        tcn_out = x
        for block in self.tcn_blocks:
            tcn_out = block(tcn_out)
        # Project → (B, 256, 480) → DyT → permute → (B, 480, 256)
        tcn_out = self.dyt_tcn(self.tcn_proj(tcn_out))
        tcn_out = tcn_out.permute(0, 2, 1)

        # ---- Concatenate branches ----
        # (B, 480, 256) + (B, 480, 256) → (B, 480, 512)
        merged = torch.cat([gru_out, tcn_out], dim=-1)

        # ---- Add global residual skip ----
        # (B, 480, 512) + (B, 480, 512) → (B, 480, 512)
        merged = merged + skip

        # ---- Lite-CBAM(T) attention ----
        # Channel: which features matter
        # Temporal: which timesteps matter
        merged = self.cbam(merged)
        merged = self.dyt_cbam(merged)  # (B, 480, 512)

        # ---- Seq2point: center timestep ----
        # (B, 480, 512) → (B, 512)
        center = merged[:, self.center, :]

        # ---- Dense + Multi-task heads ----
        # (B, 512) → (B, 128) → powers, states, gated
        out = self.dense(center)
        powers, states, gated = self.heads(out)

        return powers, states, gated

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_macs_estimate(self) -> int:
        """Rough MAC estimate for NPU budget planning."""
        T = self.window_size
        # DWSConv1: 6×5×T + 6×32×T
        # DWSConv2: 32×5×T + 32×64×T
        # BiGRU: 3×(64×128 + 128×128)×2×T
        # TCN: 4×(64×64×3)×T
        # TCN proj: 64×256×T
        # Skip proj: 64×512×T
        # CBAM channel: 512×32 + 32×512
        # CBAM temporal: 2×1×7×T
        # Dense: 512×128
        dws1 = (6 * 5 + 6 * 32) * T
        dws2 = (32 * 5 + 32 * 64) * T
        gru = 3 * (64 * 128 + 128 * 128) * 2 * T
        tcn = 4 * (64 * 64 * 3) * T
        proj = (64 * 256 + 64 * 512) * T
        cbam = 512 * 32 + 32 * 512 + 2 * 7 * T
        dense = 512 * 128
        return dws1 + dws2 + gru + tcn + proj + cbam + dense


# ============================================================
# ENTRY POINT — Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Proposed Model — Self-Test")
    print("DWT-DWSConv-BiGRU‖BiTCN-CBAM(T)-DyT")
    print("Target: STM32MP2 NPU (Neural-ART, INT8)")
    print("=" * 65)

    model = BiWaveNILM(
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
    )

    # ---- Forward pass ----
    B = 4
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    print(f"\n[Input]")
    print(f"  x shape:        {tuple(x.shape)}")

    print(f"\n[Output]")
    print(f"  powers shape:   {tuple(powers.shape)}")
    print(f"  states shape:   {tuple(states.shape)}")
    print(f"  gated  shape:   {tuple(gated.shape)}")

    print(f"\n[Output checks]")
    assert powers.shape == (B, N_APPLIANCES), f"Expected ({B},{N_APPLIANCES}), got {powers.shape}"
    assert states.shape == (B, N_APPLIANCES)
    assert gated.shape  == (B, N_APPLIANCES)
    assert (powers >= 0).all(), "powers must be >= 0 (ReLU)"
    assert (gated >= 0).all(), "gated must be >= 0"
    assert (gated <= powers + 1e-6).all(), "gated must be <= powers"
    print(f"  All shapes correct ✓")
    print(f"  powers >= 0 ✓")
    print(f"  gated <= powers ✓")

    # ---- Layer-by-layer trace ----
    print(f"\n[Layer-by-layer shapes]")
    with torch.no_grad():
        # Block 1: Dual DWSConv
        after_dws1 = F.gelu(model.dyt1(model.dws_conv1(x)))
        after_dws2 = model.drop1(F.gelu(model.dyt2(model.dws_conv2(after_dws1))))
        print(f"  After DWSConv 1:   {tuple(after_dws1.shape)}")
        print(f"  After DWSConv 2:   {tuple(after_dws2.shape)}")

        # Skip projection
        skip = model.skip_proj(after_dws2).permute(0, 2, 1)
        print(f"  Skip projection:   {tuple(skip.shape)}")

        # BiGRU branch
        gru_in = after_dws2.permute(0, 2, 1)
        gru_out, _ = model.bigru(gru_in)
        gru_out = model.dyt_gru(gru_out)
        print(f"  After BiGRU:       {tuple(gru_out.shape)}")

        # BiTCN branch
        tcn_out = after_dws2
        for block in model.tcn_blocks:
            tcn_out = block(tcn_out)
        tcn_out = model.dyt_tcn(model.tcn_proj(tcn_out))
        tcn_out = tcn_out.permute(0, 2, 1)
        print(f"  After BiTCN:       {tuple(tcn_out.shape)}")

        # Concatenate + skip
        merged = torch.cat([gru_out, tcn_out], dim=-1)
        merged = merged + skip
        print(f"  After concat+skip: {tuple(merged.shape)}")

        # CBAM
        merged = model.cbam(merged)
        merged = model.dyt_cbam(merged)
        print(f"  After CBAM+DyT:    {tuple(merged.shape)}")

        # Seq2point
        center = merged[:, WINDOW_SIZE // 2, :]
        print(f"  After seq2point:   {tuple(center.shape)}")

        # Dense
        out = model.dense(center)
        print(f"  After dense:       {tuple(out.shape)}")

    # ---- Model complexity ----
    print(f"\n[Model complexity]")
    n_params = model.count_parameters()
    print(f"  Trainable params:  {n_params:,}")
    print(f"  INT8 size:         {n_params / 1e6:.3f} MB")
    print(f"  MACs estimate:     {model.count_macs_estimate():,}")
    print(f"  Target RAM:        ≤ 2.0 MB → {'✓ PASS' if n_params / 1e6 < 2.0 else '✗ FAIL'}")

    # ---- Component parameter breakdown ----
    print(f"\n[Parameter breakdown]")
    components = {
        "DWSConv 1 (6→32)": sum(p.numel() for p in model.dws_conv1.parameters()),
        "DyT 1": sum(p.numel() for p in model.dyt1.parameters()),
        "DWSConv 2 (32→64)": sum(p.numel() for p in model.dws_conv2.parameters()),
        "DyT 2": sum(p.numel() for p in model.dyt2.parameters()),
        "Skip proj (64→512)": sum(p.numel() for p in model.skip_proj.parameters()),
        "BiGRU (64→128×2dir)": sum(p.numel() for p in model.bigru.parameters()),
        "DyT GRU": sum(p.numel() for p in model.dyt_gru.parameters()),
        "BiTCN (4× dilated)": sum(p.numel() for n, p in model.named_parameters() if "tcn_blocks" in n),
        "TCN proj (64→256)": sum(p.numel() for p in model.tcn_proj.parameters()),
        "DyT TCN": sum(p.numel() for p in model.dyt_tcn.parameters()),
        "Lite-CBAM": sum(p.numel() for p in model.cbam.parameters()),
        "DyT CBAM": sum(p.numel() for p in model.dyt_cbam.parameters()),
        "Dense (512→128)": sum(p.numel() for p in model.dense.parameters()),
        "MultiTask heads ×5": sum(p.numel() for p in model.heads.parameters()),
    }
    for name, count in components.items():
        pct = count / n_params * 100
        print(f"  {name:<24} {count:>8,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<24} {sum(components.values()):>8,}")

    # ---- Deployment targets ----
    print(f"\n[Deployment targets (r)]")
    print(f"  F1       ≥ 0.80    → to be verified after training")
    print(f"  MAE      ≤ 50 W    → to be verified after training")
    print(f"  Latency  ≤ 15 ms   → to be verified via STEdgeAI")
    print(f"  Energy   ≤ 500 mW  → to be verified on board")
    print(f"  RAM      ≤ 2.0 MB  → {n_params/1e6:.3f} MB INT8 ✓")

    # ---- Comparison with baselines ----
    print(f"\n[Comparison with baselines]")
    baselines = {
        "CNN (Zhang 2018)": 101_374,
        "GRU (Krystalakos 2018)": 184_458,
        "BiGRU (NILMFormer)": 464_522,
        "LSTM (Hwang 2022)": 129_626,
        "BiLSTM (Kelly 2015)": 323_674,
        "CNN-LSTM (Luo 2023)": 272_490,
        "NILMFormer (GPU only)": 386_204,
    }
    print(f"  {'Model':<28} {'Params':>10}  {'INT8 MB':>8}")
    print(f"  {'-'*50}")
    for name, params in baselines.items():
        print(f"  {name:<28} {params:>10,}  {params/1e6:>8.3f}")
    print(f"  {'-'*50}")
    print(f"  {'>>> PROPOSED (this) <<<':<28} {n_params:>10,}  {n_params/1e6:>8.3f}")

    print(f"\n{'=' * 65}")
    print("All tests passed — Proposed model ready!")
    print(f"{'=' * 65}")
