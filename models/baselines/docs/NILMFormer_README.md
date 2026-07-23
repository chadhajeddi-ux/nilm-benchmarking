# NILMFormer — GPU-Only Upper Bound Reference

## ⚠️ DEPLOYMENT STATUS: NOT DEPLOYABLE ON STM32MP2 NPU

| Target | Status | Reason |
|---|---|---|
| STM32MP2 NPU | ❌ | O(T²) attention, dynamic shapes |
| STM32MP2 CPU | ❌ | Too slow for real-time inference |
| GPU (Kaggle/Colab) | ✅ | Works for training and evaluation |

Included ONLY as a non-deployable accuracy upper bound.
Its published results define the ceiling for GPU-based NILM.

---

## Reference

**Paper:** NILMFormer: Non-Intrusive Load Monitoring that Accounts
for Non-Stationarity
**Authors:** Adrien Petralia, Philippe Charpentier, Youssef Kadhi,
Themis Palpanas
**Conference:** ACM SIGKDD 2025
**DOI:** https://doi.org/10.1145/3711896.3737251
**arXiv:** https://arxiv.org/abs/2506.05880
**Official GitHub:** https://github.com/adrienpetralia/NILMFormer
**Copyright:** ©2025 EDF (Électricité De France)

**Industrial deployment:** NILMFormer is deployed as the backbone
algorithm for EDF's consumption monitoring service, delivering
detailed insights to millions of customers about individual appliance
power consumption via the Mon Suivi Conso mobile application.

---

## Published Results — NILMFormer Table 2 (UK-DALE)

| Metric | NILMFormer | BiGRU | BiLSTM | Gap (our vs SOTA) |
|---|---|---|---|---|
| MAE (W) ↓ | **78.511** | 100.772 | 130.864 | target: <80W |
| MR ↑ | **0.399** | 0.314 | 0.217 | target: >0.38 |
| F1 ↑ | **0.377** | 0.253 | 0.229 | target: >0.35 |
| Accuracy ↑ | **0.749** | 0.491 | 0.430 | target: >0.70 |
| Recall ↑ | 0.866 | **0.962** | 0.921 | — |
| Precision ↑ | **0.297** | 0.177 | 0.165 | — |

NILMFormer best on almost all metrics.
BiGRU has highest Recall (0.962) but lowest Precision (0.177).
Our Focal Loss targets this precision-recall imbalance.

---

## Architecture Overview

```
Input (B, 1+e, T)
    │   1 channel: aggregate power
    │   e channels: TimeRPE (sin/cos timestamp encoding)
    │
    ▼
Instance Normalization
    │   Subtract μ and σ per window — handles non-stationarity
    │   TokenStats = [μ, σ] projected to d_model
    │
    ▼
DilatedBlock Embedding
    │   4× ResUnit(Conv1D + GELU + BN, dilations=[1,2,4,8])
    │   → (B, 3×d_model/4, T)
    │
    ▼
TimeRPE Projection
    │   Timestamp sin/cos → projected to d_model/4
    │   Concatenate with DilatedBlock → (B, T, d_model)
    │
    ▼
Append TokenStats → (B, T+1, d_model)
    │
    ▼
N× EncoderLayer
    │   DiagonallyMaskedSelfAttention(d_model, n_heads)
    │   ← O(T²): 480×480 = 230,400 elements per head ⚠️
    │   PositionWiseFeedForward(d_model, 4×d_model)
    │   LayerNorm (Pre-Norm)
    │
    ▼
Remove TokenStats → (B, T, d_model)
    │
    ▼
Conv1D Head → (B, T, c_out)
    │
    ▼
Reverse Instance Normalization via ProjStats2
    │
    ▼
Center timestep → MultiTaskHeads → powers, states, gated
```

---

## Why NILMFormer Cannot Be Deployed on STM32MP2

### Reason 1  :  O(T²) Attention Memory

For T=480 timesteps, self-attention computes a (T×T) matrix:
- FP32: 480 × 480 × 4 bytes = **921 KB per head**
- With 4 heads: 3.7 MB ,  exceeds 2MB NPU RAM budget
- INT8: still 230 KB per head + other activations → impossible

### Reason 2  :  Dynamic Computation Graph

The attention matrix shape (T×T) changes with input length T.
STEdgeAI/X-LINUX-AI requires **static** computation graphs
with fixed tensor shapes known at compile time.
A dynamic (T×T) matrix breaks this requirement.

### Reason 3  :  Custom CUDA Kernel Dependency

The original code uses `xformers.ops.memory_efficient_attention` —
a custom CUDA kernel unavailable in standard ONNX operator sets.
Our adaptation replaces this with standard PyTorch attention
for portability, but the O(T²) complexity remains.

### Reason 4  : Softmax Over Long Sequences

Softmax over 480 positions requires computing 480 exponentials
per position  , not efficiently vectorizable in INT8 on Neural-ART.

---

## Our Adaptation

```python
# Original NILMFormer (xformers required):
output = xops.memory_efficient_attention(xq, xk, xv, ...)

# Our adaptation (standard PyTorch, no xformers):
scores = torch.matmul(Q, K.transpose(-2,-1)) * scale
attn = torch.softmax(scores.masked_fill(diag_mask, -inf), dim=-1)
output = torch.matmul(attn, V)
```

All other components (DilatedBlock, TimeRPE, EncoderLayer,
instance normalization) are faithfully reproduced from
the official source code.

---

## Parameters (Our Adaptation)

| Component | Parameters |
|---|---|
| DilatedBlock (6→72, 4 ResUnits) | ~55K |
| TimeRPE projection (2→24) | ~48 |
| ProjStats1 (2→96) | ~192 |
| ProjStats2 (96→2) | ~192 |
| 2× EncoderLayer (d=96, h=4) | ~150K |
| LayerNorm | ~192 |
| Conv1D Head | ~27K |
| Dense + MultiTaskHeads | ~16K |
| **Total** | **~386K** |

FP32 size: ~1.54 MB. INT8: ~0.386 MB (if quantizable ,  it is not
for deployment but useful for size estimation).

---

## How to Use in Comparison Table

```python
# Use published numbers directly (same experimental setup)
NILMFORMER_RESULTS = {
    "model": "NILMFormer",
    "source": "Petralia et al. KDD 2025, Table 2",
    "dataset": "UK-DALE",
    "MAE_W": 78.511,
    "MR": 0.399,
    "F1": 0.377,
    "Accuracy": 0.749,
    "Recall": 0.866,
    "Precision": 0.297,
    "deployable_NPU": False,
    "deployable_CPU": False,
    "note": "GPU only — O(T²) attention incompatible with STM32MP2",
}
```

---

## Thesis Sentence

"NILMFormer [ref, KDD 2025] achieves state-of-the-art NILM
accuracy with MAE=78.5W and F1=0.377 on UK-DALE, deployed by EDF
serving millions of customers, but requires GPU inference due to
O(T²) self-attention complexity. We include it as a non-deployable
accuracy upper bound. Our proposed model targets competitive F1
while satisfying all STM32MP2 NPU deployment constraints."

---

## Citation

```bibtex
@inproceedings{petralia2025nilmformer,
  title={NILMFormer: Non-Intrusive Load Monitoring that
         Accounts for Non-Stationarity},
  author={Petralia, Adrien and Charpentier, Philippe
          and Kadhi, Youssef and Palpanas, Themis},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference
             on Knowledge Discovery and Data Mining},
  year={2025},
  publisher={ACM},
  doi={10.1145/3711896.3737251}
}
```
