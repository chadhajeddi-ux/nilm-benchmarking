# BiGRU Baseline : NILM Seq2Point

## References

### Primary : BiGRU NILM Architecture
**Paper:** Bidirectional GRU with Convolutional Layers for NILM
**Authors:** Precioso Garcelán, D.; Gomez-Ullate, D.
**Year:** 2023
**Used as baseline in:** NILMFormer (Petralia et al., KDD 2025)
**arXiv:** https://arxiv.org/abs/2506.05880
**GitHub:** https://github.com/adrienpetralia/NILMFormer

### Supporting  : GRU Bidirectional Transformer for NILM
**Paper:** Load Energy Decomposition Algorithm Based on Improved
Bidirectional Transformer Combined With Time-Sensing Self-Attention
**Authors:** Yang Xuan, Chengxin Pang, Haimeng Yu, Xinhua Zeng, Yongbo Chen
**Journal:** IEEE Access, Vol. 12, pp. 75625-75639, 2024
**DOI:** https://doi.org/10.1109/ACCESS.2024.3373801

---

## What This Model Does

BiGRU (Bidirectional GRU) baseline for NILM using seq2point.
Identical to GRU baseline but processes the sequence in BOTH
directions simultaneously — forward (past→future) and backward
(future→past). At the center prediction point, the model has
complete bilateral context from all 480 timesteps.

---

## Architecture

```
Input (B, 6, 480)
    │   6 channels = 4 DWT sub-bands + 2 temporal features
    │
    ▼
Conv1D(in=6, out=64, kernel=5, padding=2) + ReLU + Dropout(0.2)
    │   → (B, 64, 480)
    │
    ▼
Permute → (B, 480, 64)
    │
    ▼
BiGRU(input=64, hidden=128/dir, layers=2, bidirectional=True)
    │   Forward GRU:  h_fwd[t=0→480]
    │   Backward GRU: h_bwd[t=480→0]
    │   Concatenate:  [h_fwd_t ; h_bwd_t] → 256 per timestep
    │   → (B, 480, 256)
    │
    ▼
Take center timestep [:, 240, :] → (B, 256)
    │   ← Full bilateral context: positions 0..480 both directions
    │
    ▼
Linear(256→64) + ReLU + Dropout(0.2) → (B, 64)
    │
    ▼
For each appliance k (k = 1..5):
    Power head k : Linear(64→1) + ReLU  → p̂_k ∈ [0, 1]
    State head k : Linear(64→1)         → ŝ_logit,k
    Gated output : ŷ_k = p̂_k × σ(ŝ_logit,k)
```

---

## BiGRU vs GRU :  The Critical Difference

```
                    Position 0    Center (240)    Position 480
                         │              │               │
GRU (forward only):  ───────────────►  │
    Hidden at 240 sees: positions 0..240 only (PAST ONLY)
    Missing: positions 241..480 (FUTURE)

BiGRU (bidirectional):
    Forward:         ───────────────►  │  ──────────────►
    Backward:        ◄──────────────   │  ◄──────────────
    Hidden at 240 sees: ALL 480 positions (FULL BILATERAL)
```

For seq2point with window=480, the center point is position 240.
BiGRU makes FULL use of the entire window ,  the key theoretical
advantage that seq2point learning was designed to exploit.

---

## Mathematical Formulation

```
Forward GRU at timestep t:
    z_t^f = σ(W_z · [h_{t-1}^f, x_t])
    h_t^f = GRU_forward(x_0, x_1, ..., x_t)

Backward GRU at timestep t:
    z_t^b = σ(W_z · [h_{t+1}^b, x_t])
    h_t^b = GRU_backward(x_T, x_{T-1}, ..., x_t)

BiGRU output at center t=240:
    h_240 = [h_240^f ; h_240^b]  ∈ R^{256}
    h_240^f integrates: x_0, x_1, ..., x_240  (past)
    h_240^b integrates: x_480, x_479, ..., x_240 (future)
```

---

## Parameters

| Component | Parameters |
|---|---|
| Conv1D (6→64, k=5) | 1,984 |
| BiGRU layer 1 (64→128×2) | 148,992 |
| BiGRU layer 2 (256→128×2) | 198,144 |
| Dense (256→64) | 16,448 |
| Power heads (64→1) × 5 | 325 |
| State heads (64→1) × 5 | 325 |
| **Total** | **~366K** |

INT8 size: ~0.366 MB , deployable on STM32MP2 NPU.
Note: ~2× more parameters than GRU (366K vs 184K) due to
bidirectional processing , each GRU direction has its own weights.

---

## Reference Numbers : NILMFormer Table 2

From Petralia et al. (KDD 2025), BiGRU baseline on UK-DALE:

| Appliance | MAE (W) | MR |
|---|---|---|
| Dishwasher | 27.2 | 0.408 |
| Fridge | 36.7 | 0.364 |
| Kettle | 11.6 | 0.618 |
| Microwave | 12.4 | 0.040 |
| Washing Machine | 21.1 | 0.086 |

Your BiGRU+DWT model should improve on these numbers.
Your proposed DWT-BiGRU-CBAM-DyT model should improve further.

---

## Differences From GRU Baseline

| Aspect | GRU | BiGRU |
|---|---|---|
| Direction | Unidirectional (forward) | Bidirectional (fwd + bwd) |
| Hidden at center | 128 (past only) | 256 (full bilateral) |
| Context at t=240 | Positions 0→240 only | All 480 positions |
| Parameters | ~184K | ~366K |
| INT8 size | 0.184 MB | 0.366 MB |
| Theoretical advantage | Partial context | Full seq2point context |

---

## Usage

```python
from models.bigru_model import BiGRUBaseline

model = BiGRUBaseline(
    in_channels=6,
    window_size=480,
    n_appliances=5,
    gru_hidden=128,
    gru_layers=2,
    dropout=0.2,
)

x = torch.randn(32, 6, 480)
powers, states, gated = model(x)
```

---

## Citations

```bibtex
@inproceedings{petralia2025nilmformer,
  title={NILMFormer: Non-Intrusive Load Monitoring that
         Accounts for Non-Stationarity},
  author={Petralia, Adrien and others},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference
             on Knowledge Discovery and Data Mining},
  year={2025},
  doi={10.1145/3711896.3737251}
}

@article{xuan2024load,
  title={Load Energy Decomposition Algorithm Based on Improved
         Bidirectional Transformer Combined With Time-Sensing
         Self-Attention},
  author={Xuan, Yang and Pang, Chengxin and Yu, Haimeng
          and Zeng, Xinhua and Chen, Yongbo},
  journal={IEEE Access},
  volume={12},
  pages={75625--75639},
  year={2024},
  doi={10.1109/ACCESS.2024.3373801}
}
```
