# BiLSTM Baseline :  NILM Seq2Point

## References

### Primary
**Paper:** Neural NILM: Deep Neural Networks Applied to Energy Disaggregation
**Authors:** Jack Kelly, William Knottenbelt
**Conference:** ACM BuildSys 2015
**URL:** https://dl.acm.org/doi/10.1145/2821650.2821672

### Implementation Source
**Paper:** NILMFormer: Non-Intrusive Load Monitoring that Accounts
for Non-Stationarity
**Authors:** Petralia et al.
**Conference:** KDD 2025
**Source file:** src/baselines/nilm/bilstm.py (©2025 EDF)
**GitHub:** https://github.com/adrienpetralia/NILMFormer

### Reference Numbers
From NILMFormer Table 2 (UK-DALE, BiLSTM row):
MAE=130.864W | MR=0.217 | F1=0.229 | Accuracy=0.430
Recall=0.921 | Precision=0.165

---

## What This Model Does

Bidirectional LSTM baseline for NILM using seq2point.
Processes each window in BOTH directions simultaneously.
At center t=240: sees ALL 480 positions , full bilateral context.

This is the LSTM equivalent of BiGRU. Compare both to understand
whether the LSTM cell state brings benefit over GRU gating when
bidirectionality is added.

---

## Architecture

```
Input (B, 6, 480)
    │   6 channels = 4 DWT sub-bands + 2 temporal features
    │
    ▼
Conv1d(6→16, k=4, same) + Dropout(0.1)
    │   → (B, 16, 480)
    │   Matches NILMFormer bilstm.py exactly (k=4)
    │
    ▼
Permute → (B, 480, 16)
    │
    ▼
BiLSTM(16→64, bidirectional)   → (B, 480, 128)
    │   Forward LSTM:  h_fwd[t=0→480]
    │   Backward LSTM: h_bwd[t=480→0]
    │   Output: [h_fwd_t ; h_bwd_t] = 128 at each position
    │
    ▼
BiLSTM(128→128, bidirectional) → (B, 480, 256)
    │   Further temporal abstraction with full bilateral context
    │
    ▼
Take center timestep [:, 240, :] → (B, 256)
    │   FULL bilateral context: positions 0..480
    │
    ▼
Linear(256→64) + ReLU + Dropout → (B, 64)
    │
    ▼
For each appliance k (k = 1..5):
    Power head : Linear(64→1) + ReLU → p̂_k ∈ [0,1]
    State head : Linear(64→1)        → ŝ_logit,k
    Gated:       ŷ_k = p̂_k × σ(ŝ_logit,k)
```

---

## BiLSTM vs BiGRU — NILMFormer Comparison

From NILMFormer Table 2 (UK-DALE):

| Metric | BiGRU | BiLSTM | Winner |
|---|---|---|---|
| MAE (W) ↓ | 100.8 | 130.9 | BiGRU |
| MR ↑ | 0.314 | 0.217 | BiGRU |
| F1 ↑ | 0.253 | 0.229 | BiGRU |
| Accuracy ↑ | 0.491 | 0.430 | BiGRU |
| Recall ↑ | 0.962 | 0.921 | BiGRU |
| Params | 244K | 324K | BiGRU |

BiGRU consistently outperforms BiLSTM on NILM with fewer parameters.
This validates the choice of BiGRU (not BiLSTM) as the recurrent
component of the proposed model.

---

## Parameters

| Component | Parameters |
|---|---|
| Conv1d (6→16, k=4) | 400 |
| BiLSTM layer 1 (16→64×2) | 41,984 |
| BiLSTM layer 2 (128→128×2) | 394,752 |
| Dense (256→64) | 16,448 |
| Power heads (64→1) × 5 | 325 |
| State heads (64→1) × 5 | 325 |
| **Total** | **~323K** |

INT8 size: ~0.323 MB ,  deployable on STM32MP2 NPU.

---

## Key Differences From BiGRU

| Aspect | BiGRU | BiLSTM |
|---|---|---|
| Recurrent unit | GRU (2 gates) | LSTM (3 gates + cell) |
| Parameters | 244K | 323K |
| NILM performance | Better (NILMFormer) | Slightly worse |
| Conv kernel | k=5 | k=4 (Kelly et al.) |
| Deployment | ✅ | ✅ |

---

## Usage

```python
from models.baselines.bilstm_model import BiLSTMBaseline

model = BiLSTMBaseline(
    in_channels=6,
    window_size=480,
    n_appliances=5,
    dropout=0.1,
)

x = torch.randn(32, 6, 480)
powers, states, gated = model(x)
```

---

## Citations

```bibtex
@inproceedings{kelly2015neural,
  title={Neural NILM: Deep neural networks applied to
         energy disaggregation},
  author={Kelly, Jack and Knottenbelt, William},
  booktitle={Proceedings of the 2nd ACM International Conference
             on Embedded Systems for Energy-Efficient Built
             Environments},
  pages={55--64},
  year={2015}
}

@inproceedings{petralia2025nilmformer,
  title={NILMFormer: Non-Intrusive Load Monitoring that
         Accounts for Non-Stationarity},
  author={Petralia, Adrien and others},
  booktitle={KDD 2025},
  doi={10.1145/3711896.3737251},
  year={2025}
}
```
