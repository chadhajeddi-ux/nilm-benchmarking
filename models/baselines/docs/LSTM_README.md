# LSTM Baseline :  NILM Seq2Point

## References

### Primary
**Paper:** Nonintrusive Load Monitoring Using an LSTM with Feedback Structure
**Authors:** Hyeontaek Hwang, Sanggil Kang
**Journal:** IEEE Transactions on Instrumentation and Measurement, Vol. 71, 2022
**DOI:** https://doi.org/10.1109/TIM.2022.3169536

### Supporting
**Paper:** A New Approach for Supervised Power Disaggregation by Using
a Deep Recurrent LSTM Network
**Authors:** Mauch L., Yang B.
**Conference:** IEEE GlobalSIP 2015

---

## What This Model Does

Unidirectional LSTM baseline for NILM using seq2point.
Processes the window forward only (past → future).
At the center timestep t=240, the hidden state contains
information from positions 0..240 only  , no future context.

Compare with BiLSTM to quantify the value of bidirectionality.

---

## Architecture

```
Input (B, 6, 480)
    │   6 channels = 4 DWT sub-bands + 2 temporal features
    │
    ▼
Conv1d(6→16, k=4, same) + Dropout(0.1)
    │   → (B, 16, 480)
    │   k=4 matches NILMFormer bilstm.py convention
    │
    ▼
Permute → (B, 480, 16)
    │
    ▼
LSTM(16→64, unidirectional)   → (B, 480, 64)
    │   forget gate: f_t = σ(W_f · [h_{t-1}, x_t])
    │   input  gate: i_t = σ(W_i · [h_{t-1}, x_t])
    │   cell update: c̃_t = tanh(W_c · [h_{t-1}, x_t])
    │   cell state:  c_t = f_t ⊗ c_{t-1} + i_t ⊗ c̃_t
    │   output gate: o_t = σ(W_o · [h_{t-1}, x_t])
    │   hidden:      h_t = o_t ⊗ tanh(c_t)
    │
    ▼
LSTM(64→128, unidirectional)  → (B, 480, 128)
    │
    ▼
Take center timestep [:, 240, :] → (B, 128)
    │   PAST context only (positions 0..240)
    │   Missing: future context (positions 241..480)
    │
    ▼
Linear(128→64) + ReLU + Dropout → (B, 64)
    │
    ▼
For each appliance k (k = 1..5):
    Power head : Linear(64→1) + ReLU → p̂_k ∈ [0,1]
    State head : Linear(64→1)        → ŝ_logit,k
    Gated:       ŷ_k = p̂_k × σ(ŝ_logit,k)
```

---

## LSTM vs GRU — Gate Comparison

| Property | GRU | LSTM |
|---|---|---|
| Gates | 2 (reset, update) | 3 (forget, input, output) |
| Memory | h_t only | h_t + c_t (cell state) |
| Parameters | fewer | ~33% more |
| Key advantage | Simpler, faster | Explicit long-term memory |
| NILM difference | Minimal in practice | Minimal in practice |

The separate cell state c_t in LSTM provides dedicated long-term
memory independent of the output h_t. In theory this helps for
very long sequences. In practice for NILM at 480 timesteps, the
difference is small  , NILMFormer Table 2 shows BiGRU and BiLSTM
perform similarly when bidirectional.

---

## Parameters

| Component | Parameters |
|---|---|
| Conv1d (6→16, k=4) | 400 |
| LSTM layer 1 (16→64) | 20,992 |
| LSTM layer 2 (64→128) | 98,816 |
| Dense (128→64) | 8,256 |
| Power heads (64→1) × 5 | 325 |
| State heads (64→1) × 5 | 325 |
| **Total** | **~129K** |

INT8 size: ~0.129 MB  ,  smallest recurrent baseline.

---

## Ablation Value

LSTM vs GRU comparison isolates the effect of the cell state:
- If LSTM significantly outperforms GRU → cell state matters
- If similar → GRU is the better choice (fewer params, faster)

LSTM vs BiLSTM comparison isolates bidirectionality:
- Expected: BiLSTM clearly better (full bilateral context)
- This is confirmed by NILMFormer Table 2

---

## Usage

```python
from models.baselines.lstm_model import LSTMBaseline

model = LSTMBaseline(
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
@article{hwang2022nonintrusive,
  title={Nonintrusive load monitoring using an LSTM
         with feedback structure},
  author={Hwang, Hyeontaek and Kang, Sanggil},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--11},
  year={2022},
  doi={10.1109/TIM.2022.3169536}
}

@inproceedings{mauch2015new,
  title={A new approach for supervised power disaggregation
         by using a deep recurrent LSTM network},
  author={Mauch, Lukas and Yang, Bin},
  booktitle={IEEE GlobalSIP},
  pages={63--67},
  year={2015}
}
```
