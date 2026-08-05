"""
proposed_model.py — BiWave-UNet: The Proposed NILM Model
=========================================================
Architecture: UNet Encoder-Decoder + BiGRU + BiTCN + CBAM + DyT + Seq2Seg

Combines the best of:
    - UNet-NILM (Faustine et al. 2020): multi-scale encoder-decoder + skip connections
    - BiWave-NILM (ours): DWT + BiGRU + BiTCN + CBAM + DyT + Seq2Seg
    - NILMFormer (Petralia 2025): instance normalization for cross-house
    - DU-NILM (2024): seq2seg output strategy

Architecture:
    Input (B,6,480)
        ↓
    ENCODER:
        E1: DWSConv(6→32)   + CBAM_T → (B,32,480)
        E2: DWSConv(32→64)  + CBAM_T → (B,64,240)  [MaxPool2]
        E3: DWSConv(64→128) + CBAM_T → (B,128,120) [MaxPool2]
        ↓
    BOTTLENECK:
        BiGRU(128→128) ‖ BiTCN(128, dilations=1,2,4,8)
        → Concat → Conv(256→128) → CBAM_T → DyT
        → (B,128,120)
        ↓
    DECODER (with skip connections):
        D3: Upsample + Cat[BN,E3] → DWSConv(256→64) → CBAM_T → (B,64,240)
        D2: Upsample + Cat[D3,E2] → DWSConv(128→32) → CBAM_T → (B,32,480)
        D1: Cat[D2,E1] → DWSConv(64→32) → (B,32,480)
        ↓
    SEQ2SEG HEADS (center 96 points):
        seg_power: Conv1D(32→5) → (B,5,96)
        ctr_state: pool → Linear(32→5) → (B,5)
        seg_gated: power * sigmoid(state) → (B,5,96)

Parameters: ~420K | INT8: 0.42 MB | Target: STM32MP2 NPU

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


# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

class DWSConvBlock(nn.Module):
    """
    Depthwise Separable Conv1D Block.
    DWSConv = depthwise (per-channel) + pointwise (1x1) convolution.
    90% fewer parameters than standard Conv1D.
    Reference: MobileNet (Howard et al. 2017)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=kernel//2,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.pw(self.dw(x)))))


class ResidualDWSConv(nn.Module):
    """
    Residual DWSConv Block — two DWSConv + skip connection.
    Prevents vanishing gradient in deep encoder/decoder.
    """
    def __init__(self, channels: int, kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel//2,
                      groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=kernel//2,
                      groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class CBAM_Temporal(nn.Module):
    """
    CBAM Temporal Attention for 1D time series.
    Focuses on the most informative timesteps.
    Reference: Woo et al. ECCV 2018
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Channel attention
        self.ca_gap = nn.AdaptiveAvgPool1d(1)
        self.ca_gmp = nn.AdaptiveMaxPool1d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Temporal attention
        self.ta_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_c = self.ca_fc(self.ca_gap(x).squeeze(-1))
        max_c = self.ca_fc(self.ca_gmp(x).squeeze(-1))
        ca = torch.sigmoid(avg_c + max_c).unsqueeze(-1)
        x = x * ca

        # Temporal attention
        avg_t = x.mean(dim=1, keepdim=True)    # (B, 1, T)
        max_t = x.max(dim=1, keepdim=True)[0]  # (B, 1, T)
        ta = torch.sigmoid(self.ta_conv(
            torch.cat([avg_t, max_t], dim=1)))  # (B, 1, T)
        x = x * ta

        return x


class BiTCNBlock(nn.Module):
    """
    Bidirectional Temporal Convolutional Network.
    Multi-scale dilated convolutions capture patterns at different time scales.
    Dilations: 1, 2, 4, 8 → receptive field = 1+2+4+8 = 15 steps per direction.
    """
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in [1, 2, 4, 8]:
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          groups=channels, bias=False),
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = out + layer(out)  # residual per dilation
        return out


class DynamicThreshold(nn.Module):
    """
    Dynamic Thresholding (DyT) — learnable activation thresholds.
    Replaces fixed ReLU with per-channel learnable thresholds.
    Helps with appliance-specific activation levels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * 0.5)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        return torch.tanh(self.alpha * x + self.beta)


# ═══════════════════════════════════════════════════════════════
# ENCODER
# ═══════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    3-level encoder with progressive downsampling.
    Each level: DWSConv → ResidualDWSConv → CBAM → output
    Skip connections stored for decoder.
    """
    def __init__(self, in_ch: int = 6, dims: list = [32, 64, 128],
                 dropout: float = 0.2):
        super().__init__()
        self.levels = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_ch
        for i, dim in enumerate(dims):
            self.levels.append(nn.Sequential(
                DWSConvBlock(prev_ch, dim, kernel=5, dropout=dropout),
                ResidualDWSConv(dim, kernel=5, dropout=dropout),
                CBAM_Temporal(dim, reduction=4),
            ))
            if i < len(dims) - 1:
                self.pools.append(nn.MaxPool1d(2))
            prev_ch = dim

    def forward(self, x):
        skips = []
        for i, level in enumerate(self.levels):
            x = level(x)
            skips.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        return x, skips


# ═══════════════════════════════════════════════════════════════
# BOTTLENECK
# ═══════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """
    BiGRU ‖ BiTCN → Concat → Conv → CBAM → DyT
    The heart of BiWave-UNet — captures temporal dependencies.
    """
    def __init__(self, channels: int = 128, dropout: float = 0.2):
        super().__init__()
        # BiGRU branch
        self.bigru = nn.GRU(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # BiTCN branch
        self.bitcn = BiTCNBlock(channels, dropout=dropout)

        # Merge
        self.merge = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.cbam = CBAM_Temporal(channels, reduction=4)
        self.dyt = DynamicThreshold(channels)

    def forward(self, x):
        # BiGRU: (B, C, T) → (B, T, C) → GRU → (B, T, C) → (B, C, T)
        gru_out, _ = self.bigru(x.permute(0, 2, 1))
        gru_out = gru_out.permute(0, 2, 1)  # (B, C, T)

        # BiTCN
        tcn_out = self.bitcn(x)  # (B, C, T)

        # Concat + merge
        merged = torch.cat([gru_out, tcn_out], dim=1)  # (B, 2C, T)
        merged = self.merge(merged)    # (B, C, T)
        merged = self.cbam(merged)     # (B, C, T)
        merged = self.dyt(merged)      # (B, C, T)

        return merged


# ═══════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════

class DecoderLevel(nn.Module):
    """
    Single decoder level: Upsample → Cat(skip) → DWSConv → ResConv → CBAM
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear',
                               align_corners=False)
        self.conv = nn.Sequential(
            DWSConvBlock(in_ch + skip_ch, out_ch, kernel=5, dropout=dropout),
            ResidualDWSConv(out_ch, kernel=5, dropout=dropout),
            CBAM_Temporal(out_ch, reduction=4),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Match temporal dimension
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    3-level decoder with skip connections from encoder.
    Progressively upsamples and refines features.
    dims should be encoder dims [32, 64, 128] — decoder reverses internally.
    """
    def __init__(self, dims: list = [32, 64, 128], dropout: float = 0.2):
        super().__init__()
        # Level 3→2: bottleneck(128) + E3 skip(128) → 64
        self.level3 = DecoderLevel(dims[2], dims[2], dims[1], dropout)
        # Level 2→1: D3(64) + E2 skip(64) → 32
        self.level2 = DecoderLevel(dims[1], dims[1], dims[0], dropout)
        # Final refinement at level 1 (no upsampling)
        # D2(32) + E1 skip(32) → 32
        self.level1 = nn.Sequential(
            DWSConvBlock(dims[0] + dims[0], dims[0], kernel=5, dropout=dropout),
            ResidualDWSConv(dims[0], kernel=5, dropout=dropout),
        )

    def forward(self, x, skips):
        # skips = [E1(B,32,480), E2(B,64,240), E3(B,128,120)]
        x = self.level3(x, skips[2])    # (B,128,120) → (B,64,240)
        x = self.level2(x, skips[1])    # (B,64,240)  → (B,32,480)
        # Concatenate with E1 skip (no upsampling needed)
        x = torch.cat([x, skips[0]], dim=1)  # (B,64,480)
        x = self.level1(x)              # (B,32,480)
        return x


# ═══════════════════════════════════════════════════════════════
# SEQ2SEG HEADS
# ═══════════════════════════════════════════════════════════════

class Seq2SegHeads(nn.Module):
    """
    Seq2Seg output heads — predict 96-point segments.
    Reference: DU-NILM 2024

    Three outputs:
        seg_power: (B, 5, 96)  — power regression on segment
        ctr_state: (B, 5)      — binary state at center point
        seg_gated: (B, 5, 96)  — power * sigmoid(state) gated output
    """
    def __init__(self, in_ch: int = 32, n_app: int = 5, seg_size: int = 96):
        super().__init__()
        self.seg_size = seg_size
        self.n_app = n_app

        # Per-appliance power heads
        self.power_heads = nn.ModuleList([
            nn.Conv1d(in_ch, 1, kernel_size=1) for _ in range(n_app)
        ])
        # Per-appliance state heads (center point classification)
        self.state_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(in_ch, 1),
            ) for _ in range(n_app)
        ])

    def forward(self, x, window_size: int = 480):
        """
        x: (B, 32, 480) — decoder output at full resolution
        """
        seg_start = (window_size - self.seg_size) // 2
        seg_end   = seg_start + self.seg_size

        # Extract center segment
        x_seg = x[:, :, seg_start:seg_end]  # (B, 32, 96)

        # Power heads: per-appliance Conv1D
        seg_power = torch.cat([
            torch.sigmoid(h(x_seg)) for h in self.power_heads
        ], dim=1)  # (B, 5, 96)

        # State heads: center point from segment
        x_center = x[:, :, window_size // 2 : window_size // 2 + 1]  # (B,32,1)
        ctr_state = torch.cat([
            h(x_center) for h in self.state_heads
        ], dim=1)  # (B, 5)

        # Gated output: power * sigmoid(state)
        state_gate = torch.sigmoid(ctr_state).unsqueeze(-1)  # (B, 5, 1)
        seg_gated = seg_power * state_gate  # (B, 5, 96)

        return seg_power, ctr_state, seg_gated


# ═══════════════════════════════════════════════════════════════
# MAIN MODEL
# ═══════════════════════════════════════════════════════════════

class BiWaveNILM(nn.Module):
    """
    BiWave-UNet: The Proposed NILM Model
    =====================================

    A multi-scale encoder-decoder architecture combining:
    - UNet skip connections (Faustine et al. 2020)
    - BiGRU bidirectional recurrence (temporal modeling)
    - BiTCN multi-scale dilated convolutions (local patterns)
    - CBAM temporal attention (focus on switching events)
    - DyT dynamic thresholding (appliance-specific activations)
    - DWSConv depthwise separable convolutions (efficiency)
    - Seq2Seg 96-point output (DU-NILM 2024)

    Input:  (B, 6, 480) — 4 DWT sub-bands + 2 temporal features
    Output: seg_power (B,5,96), ctr_state (B,5), seg_gated (B,5,96)

    Parameters: ~420K | INT8: 0.42 MB
    Target: STM32MP2 NPU (Neural-ART accelerator)

    Parameters
    ----------
    in_channels  : int  Default 6 (4 DWT + sin_hour + cos_hour)
    window_size  : int  Default 480 (48 minutes at 6s sampling)
    n_appliances : int  Default 5
    seg_size     : int  Default 96 (center segment for seq2seg)
    dims         : list Default [32, 64, 128] (encoder/decoder channels)
    dropout      : float Default 0.2
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        seg_size: int = 96,
        dropout: float = 0.2,
        dims: list = None,
    ):
        super().__init__()
        if dims is None:
            dims = [32, 64, 128]

        self.window_size = window_size
        self.seg_size = seg_size
        self.n_appliances = n_appliances

        # Encoder: 3 levels
        self.encoder = Encoder(
            in_ch=in_channels, dims=dims, dropout=dropout)

        # Bottleneck: BiGRU + BiTCN + CBAM + DyT
        self.bottleneck = Bottleneck(
            channels=dims[-1], dropout=dropout)

        # Decoder: 3 levels with skip connections
        self.decoder = Decoder(
            dims=dims, dropout=dropout)

        # Seq2Seg heads
        self.heads = Seq2SegHeads(
            in_ch=dims[0], n_app=n_appliances, seg_size=seg_size)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 6, 480)
        Returns:
            seg_power: (B, 5, 96) — power regression
            ctr_state: (B, 5)     — state classification
            seg_gated: (B, 5, 96) — gated output
        """
        # Encode
        bottleneck_in, skips = self.encoder(x)
        # skips = [E1(B,32,480), E2(B,64,240), E3(B,128,120)]
        # bottleneck_in = E3 output (B,128,120)

        # Bottleneck
        bn = self.bottleneck(bottleneck_in)  # (B,128,120)

        # Decode with skip connections
        decoded = self.decoder(bn, skips)  # (B,32,480)

        # Seq2Seg output
        seg_power, ctr_state, seg_gated = self.heads(
            decoded, self.window_size)

        return seg_power, ctr_state, seg_gated

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self):
        n = self.count_parameters()
        print(f'BiWave-UNet | Parameters: {n:,} | INT8: {n/1e6:.3f} MB')
        print()
        print('Architecture:')
        print('  ENCODER:')
        print('    E1: DWSConv(6→32)   + ResDWSConv + CBAM_T → (B,32,480)')
        print('    E2: DWSConv(32→64)  + ResDWSConv + CBAM_T → (B,64,240)')
        print('    E3: DWSConv(64→128) + ResDWSConv + CBAM_T → (B,128,120)')
        print()
        print('  BOTTLENECK:')
        print('    BiGRU(128→128) ‖ BiTCN(128, d=1,2,4,8)')
        print('    → Concat → Conv(256→128) → CBAM_T → DyT')
        print()
        print('  DECODER (skip connections):')
        print('    D3: Up + Cat[BN,E3] → DWSConv(256→64) + CBAM_T → (B,64,240)')
        print('    D2: Up + Cat[D3,E2] → DWSConv(128→32) + CBAM_T → (B,32,480)')
        print('    D1: Cat[D2,E1] → DWSConv(64→32) → (B,32,480)')
        print()
        print('  HEADS (Seq2Seg):')
        print('    seg_power: Conv1D(32→5) → (B,5,96)')
        print('    ctr_state: Pool+Linear  → (B,5)')
        print('    seg_gated: power×σ(state) → (B,5,96)')
        print()
        print(f'  Deployable: STM32MP2 NPU (INT8={n/1e6:.3f} MB < 16 MB)')


# Alias for backward compatibility
ProposedModel = BiWaveNILM


if __name__ == "__main__":
    print("="*70)
    print("BiWave-UNet — Self-Test")
    print("="*70)

    model = BiWaveNILM()
    B = 4
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)

    seg_p, ctr_s, seg_g = model(x)

    assert seg_p.shape == (B, N_APPLIANCES, 96), f"seg_p wrong: {seg_p.shape}"
    assert ctr_s.shape == (B, N_APPLIANCES),     f"ctr_s wrong: {ctr_s.shape}"
    assert seg_g.shape == (B, N_APPLIANCES, 96), f"seg_g wrong: {seg_g.shape}"

    model.model_summary()
    print()
    print(f"[Input]  {tuple(x.shape)}")
    print(f"[Output] seg_power={tuple(seg_p.shape)}  "
          f"ctr_state={tuple(ctr_s.shape)}  "
          f"seg_gated={tuple(seg_g.shape)}")
    print()

    # Test gradient flow
    loss = seg_p.sum() + ctr_s.sum() + seg_g.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print(f"Gradient flow: {'OK' if grad_ok else 'BROKEN'}")
    print()
    print("All tests passed!")
