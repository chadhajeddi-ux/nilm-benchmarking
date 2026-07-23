"""
nilmformer_model.py  : NILMFormer (GPU only, non-deployable on NPU)
===================================================================
Adapted from official NILMFormer source code:
    Petralia et al. "NILMFormer: Non-Intrusive Load Monitoring
    that Accounts for Non-Stationarity." KDD 2025.
    Copyright ©2025 EDF, Adrien Petralia
    GitHub: https://github.com/adrienpetralia/NILMFormer
    License: See NILMFormer/LICENSE

IMPORTANT — DEPLOYMENT STATUS:
    ❌ NOT DEPLOYABLE on STM32MP2 NPU
    Reason 1: Self-attention O(T²) complexity , 480×480 matrix
               requires ~900KB RAM at FP32 (exceeds 2MB budget with
               other activations)
    Reason 2: xformers.ops.memory_efficient_attention uses a custom
               CUDA kernel that cannot export to ONNX
    Reason 3: Dynamic attention matrix shape (T,T) breaks static
               computation graph required by X-LINUX-AI
    Status: GPU-only upper bound reference

    This model is included ONLY as a non-deployable state-of-the-art
    reference. Its published results (MAE=78.5W, F1=0.377 on UK-DALE)
    represent the accuracy ceiling for GPU-based NILM.

Modifications from original:
    1. Replaced xformers efficient attention with standard PyTorch
       scaled dot-product attention (for CPU/GPU without xformers)
    2. Added multi-task heads (power + state per appliance, N=5)
       instead of single regression output
    3. Added gated output: ŷ = p̂ × σ(ŝ) for physical consistency
    4. Adapted input to (B, 6, 480) with DWT+temporal channels
    5. Seq2point center extraction instead of seq2seq

Architecture overview:
    Input (B, 6+e, 480)
        ↓  Instance normalization
        ↓  DilatedBlock: 4× ResUnit(Conv1D+GELU+BN, dilations=[1,2,4,8])
        ↓  TimeRPE: sinusoidal timestamp encoding → projected
        ↓  Concatenate DilatedBlock + TimeRPE → (B, 480, d_model)
        ↓  Append TokenStats (mean+std) → (B, 481, d_model)
        ↓  N× EncoderLayer: DiagMaskedSelfAttention + PFFN + LayerNorm
        ↓  Remove stats token → (B, 480, d_model)
        ↓  Conv1D head → (B, 480, c_out)
        ↓  Reverse instance normalization via ProjStats2
        ↓  Take center timestep → N multi-task heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
)

from cnn_model import MultiTaskHeads


# ============================================================
# EMBEDDING BLOCK — from NILMFormer embedding.py (unchanged)
# ============================================================

class ResUnit(nn.Module):
    """Residual Conv1D unit with GELU activation and BatchNorm."""
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dilation,
                      stride=stride, bias=bias, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        self.match_residual = (c_in > 1 and c_in != c_out)
        if self.match_residual:
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        if self.match_residual:
            return torch.add(self.conv(x), self.layers(x))
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    """Stack of ResUnits with increasing dilation — the embedding block."""
    def __init__(self, c_in=32, c_out=32, kernel_size=8,
                 dilation_list=[1, 2, 4, 8], bias=True):
        super().__init__()
        layers = []
        for i, dilation in enumerate(dilation_list):
            in_ch = c_in if i == 0 else c_out
            layers.append(ResUnit(in_ch, c_out, k=kernel_size,
                                  dilation=dilation, bias=bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK — adapted from NILMFormer transformer.py
# Standard PyTorch attention replaces xformers (for portability)
# ============================================================

class DiagonallyMaskedSelfAttention(nn.Module):
    """
    Self-attention with diagonal mask — from NILMFormer transformer.py.

    The diagonal mask sets attention scores at position (i,i) to -inf
    before softmax, preventing each token from attending to itself.
    This encourages the model to aggregate context from neighbors.

    ADAPTATION: replaced xformers.ops.memory_efficient_attention with
    standard PyTorch scaled_dot_product_attention for portability.
    This runs on CPU and GPU without the xformers library.

    NOTE: This is still O(T²) — 480×480 = 230K elements per head.
    This is why NILMFormer cannot be deployed on STM32MP2 NPU.
    """
    def __init__(self, dim, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape

        Q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Diagonal mask: prevent self-attention at each position
        diag_mask = torch.eye(L, dtype=torch.bool, device=x.device)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(diag_mask, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)  # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.dropout(self.wo(out))


class PositionWiseFeedForward(nn.Module):
    """Two-layer FFN with GELU activation — from NILMFormer."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-Norm Transformer encoder layer — from NILMFormer."""
    def __init__(self, d_model, n_head, pffn_ratio=4, dp_rate=0.1):
        super().__init__()
        head_dim = d_model // n_head
        self.attn = DiagonallyMaskedSelfAttention(d_model, n_head, head_dim, dp_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_model * pffn_ratio, dp_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ============================================================
# NILMFORMER MAIN MODEL
# ============================================================

class NILMFormerModel(nn.Module):
    """
    NILMFormer for multi-task NILM — adapted from Petralia et al. KDD 2025.

    GPU-only model included as non-deployable upper bound reference.
    Published accuracy: MAE=78.5W, F1=0.377 on UK-DALE (Table 2).

    Parameters
    ----------
    in_channels : int
        Input channels. Default 6 (4 DWT + 2 temporal).
        Original uses 1 (raw power) + e (TimeRPE channels).
    window_size : int
        Sequence length. Default 480.
    n_appliances : int
        Number of target appliances. Default 5.
    d_model : int
        Transformer inner dimension. Must be divisible by 4.
        Default 96 (matches NILMFormer paper).
    n_encoder_layers : int
        Number of Transformer encoder layers. Default 2.
    n_head : int
        Number of attention heads. Default 4.
    kernel_size : int
        DilatedBlock kernel size. Default 8.
    dilations : list
        Dilation factors for DilatedBlock. Default [1,2,4,8].
    dp_rate : float
        Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        d_model: int = 96,
        n_encoder_layers: int = 2,
        n_head: int = 4,
        kernel_size: int = 8,
        dilations: list = [1, 2, 4, 8],
        dp_rate: float = 0.1,
    ):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"

        self.window_size = window_size
        self.center = window_size // 2
        self.d_model = d_model

        # d_model split: 3/4 from DilatedBlock, 1/4 from TimeRPE
        d_model_embed = 3 * d_model // 4   # e.g. 72 if d_model=96
        d_model_rpe   = d_model // 4        # e.g. 24 if d_model=96

        # ---- Embedding Block ----
        # DilatedBlock processes DWT sub-bands (first 4 channels)
        n_dwt_channels = max(1, in_channels - 2)  # all except temporal
        self.embed_block = DilatedBlock(
            c_in=n_dwt_channels,
            c_out=d_model_embed,
            kernel_size=kernel_size,
            dilation_list=dilations,
        )

        # ---- TimeRPE projection ----
        # Projects 2 temporal channels (sin/cos hour) to d_model//4
        # n_temporal = in_channels - n_dwt_channels = 2
        self.n_dwt_channels = n_dwt_channels
        self.proj_rpe = nn.Conv1d(in_channels - n_dwt_channels if in_channels > n_dwt_channels else 2,
                                   d_model_rpe, kernel_size=1)

        # ---- Stats token (instance normalization tokens) ----
        self.proj_stats1 = nn.Linear(2, d_model)   # encode mean+std
        self.proj_stats2 = nn.Linear(d_model, 2)   # decode mean+std

        # ---- Transformer Encoder ----
        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model, n_head, pffn_ratio=4, dp_rate=dp_rate)
              for _ in range(n_encoder_layers)],
            nn.LayerNorm(d_model),
        )

        # ---- Downstream Head ----
        self.head_conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                   padding=1, padding_mode='replicate')

        # ---- Multi-task output heads ----
        self.dense = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dp_rate),
        )
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 6, 480) — DWT sub-bands + temporal features
        Returns: powers, states, gated — each (B, N_app)
        """
        B, C, L = x.shape

        # Separate DWT channels (first n_dwt) and temporal (last 2)
        n_dwt = self.n_dwt_channels
        dwt_x = x[:, :n_dwt, :]                          # (B, n_dwt, L)
        rpe_x = x[:, n_dwt:, :] if C > n_dwt else \
                torch.zeros(B, 2, L, device=x.device)    # (B, 2, L)

        # ---- Instance normalization ----
        proxy = dwt_x.mean(dim=1, keepdim=True)  # (B, 1, L)
        inst_mean = proxy.mean(dim=-1, keepdim=True).detach()   # (B, 1, 1)
        inst_std  = proxy.std(dim=-1, keepdim=True).detach() + 1e-6

        dwt_norm = (dwt_x - inst_mean) / inst_std  # (B, 4, L)

        # ---- Embedding: DilatedBlock ----
        embed = self.embed_block(dwt_norm)  # (B, d_model*3//4, L)

        # ---- TimeRPE projection ----
        rpe = self.proj_rpe(rpe_x)  # (B, d_model//4, L)

        # ---- Concatenate → (B, L, d_model) ----
        x_seq = torch.cat([embed, rpe], dim=1).permute(0, 2, 1)

        # ---- Stats token ----
        stats = torch.cat([inst_mean, inst_std], dim=1).squeeze(-1)  # (B, 2)
        stats_token = self.proj_stats1(stats).unsqueeze(1)  # (B, 1, d_model)
        x_seq = torch.cat([x_seq, stats_token], dim=1)     # (B, L+1, d_model)

        # ---- Transformer Encoder ----
        x_seq = self.encoder(x_seq)      # (B, L+1, d_model)
        x_seq = x_seq[:, :-1, :]        # remove stats token (B, L, d_model)

        # ---- Conv head ----
        x_seq = self.head_conv(x_seq.permute(0, 2, 1))  # (B, d_model, L)

        # ---- Reverse instance normalization ----
        stats_out = self.proj_stats2(stats_token)   # (B, 1, 2)
        out_mean  = stats_out[:, :, 0]              # (B, 1)
        out_std   = stats_out[:, :, 1]              # (B, 1)
        x_seq = x_seq * out_std.unsqueeze(-1) + out_mean.unsqueeze(-1)

        # ---- Seq2point: center timestep ----
        center = x_seq[:, :, self.center]   # (B, d_model)

        # ---- Dense + multi-task heads ----
        out = self.dense(center)
        powers, states, gated = self.heads(out)
        return powers, states, gated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# ENTRY POINT — Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NILMFormer — Self-Test")
    print("Source: Petralia et al. KDD 2025 (©2025 EDF)")
    print("Status: GPU-only | NOT deployable on STM32MP2 NPU")
    print("=" * 60)

    model = NILMFormerModel(
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
        d_model=96,
        n_encoder_layers=2,
        n_head=4,
    )

    B = 4
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
    print(f"  All shapes correct ✓")
    print(f"  powers >= 0 ✓")

    print(f"\n[Model complexity]")
    n_params = model.count_parameters()
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Approx. size (FP32):  {n_params * 4 / 1e6:.2f} MB")
    print(f"  Attention matrix:     480×480 = {480*480:,} elements per head")
    print(f"  → O(T²) — incompatible with STM32MP2 NPU static graph")

    print(f"\n[Published results — NILMFormer Table 2, UK-DALE]")
    print(f"  MAE:      78.511 W  ← best GPU result")
    print(f"  MR:       0.399")
    print(f"  F1:       0.377     ← best GPU result")
    print(f"  Accuracy: 0.749")
    print(f"  Recall:   0.866")
    print(f"  Precision:0.297")
    print(f"  → Used as GPU upper bound in comparison table")
    print(f"  → Your deployable model targets F1 > 0.35 on STM32MP2")

    print(f"\n[Deployment verdict]")
    print(f"  STM32MP2 NPU:  ❌ NOT DEPLOYABLE")
    print(f"  STM32MP2 CPU:  ❌ Too slow (O(T²) attention)")
    print(f"  GPU (Kaggle):  ✅ Works for training and evaluation")

    print(f"\n{'=' * 60}")
    print("All tests passed — NILMFormer ready for GPU evaluation!")
    print(f"{'=' * 60}")
