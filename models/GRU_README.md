# GRU Baseline — NILM Seq2Point

## References

### Primary — WGRU Architecture
**Paper:** Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks
**Authors:** Odysseas Krystalakos, Christoforos Nalmpantis, Dimitris Vrakas
**Conference:** HellAI 2018 (10th Hellenic Conference on Artificial Intelligence)
**GitHub (original):** https://github.com/Virtsionis/SelfAttentiveEnergyDisaggregator

### Validation — SAED Benchmark
**Paper:** SAED: Self-Attentive Energy Disaggregation
**Authors:** Nikolaos Virtsionis Gkalinikis, Christoforos Nalmpantis, Dimitris Vrakas
**Journal:** Machine Learning, Springer, 2021
**DOI:** https://doi.org/10.1007/s10994-021-06106-3
**GitHub:** https://github.com/Virtsionis/SelfAttentiveEnergyDisaggregator

### Supplementary — LSTM Feedback Structure
**Paper:** Nonintrusive Load Monitoring Using an LSTM with Feedback Structure
**Authors:** Hyeontaek Hwang, Sanggil Kang
**Journal:** IEEE Transactions on Instrumentation and Measurement, 2022
**DOI:** https://doi.org/10.1109/TIM.2022.3169536

---

## What This Model Does

GRU (Gated Recurrent Unit) baseline for NILM using seq2point.
The Conv1D layer extracts local patterns from DWT sub-bands.
The GRU layers model temporal dependencies across the full window
by maintaining a hidden state that adapts to the input content.

This is the UNIDIRECTIONAL version (forward pass only).
BiLSTM baseline uses bidirectional processing.

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
    │   GRU expects (batch, time, features)
    │
    ▼
GRU(input=64, hidden=128, layers=2, dropout=0.2, bidirectional=False)
    │   → (B, 480, 128)
    │   Hidden state h_t carries temporal context from all past timesteps
    │
    ▼
Take center timestep [:, 240, :] → (B, 128)
    │   Seq2point: bilateral context (positions 0-239 inform via
    │   Conv1D features, positions 0-240 inform via GRU state)
    │
    ▼
Linear(128→64) + ReLU + Dropout(0.2) → (B, 64)
    │
    ▼
For each appliance k (k = 1..5):
    Power head k : Linear(64→1) + ReLU  → p̂_k ∈ [0, 1]
    State head k : Linear(64→1)         → ŝ_logit,k
    Gated output : ŷ_k = p̂_k × σ(ŝ_logit,k)
```

---

## GRU vs LSTM — Why GRU is a Valid Separate Baseline

| Property | GRU | LSTM |
|---|---|---|
| Gates | 2 (reset, update) | 3 (forget, input, output) |
| States | 1 (hidden h_t) | 2 (hidden h_t + cell c_t) |
| Parameters | ~396K for our config | ~525K for same config |
| Speed | Faster (fewer operations) | Slower |
| NILM performance | Comparable (SAED paper) | Comparable |
| NPU deployment | Simpler gate structure | More complex |

GRU and LSTM are separate baselines because they have fundamentally
different internal mechanisms. GRU merges the forget and input gates
into a single update gate and has no separate cell state — making it
simpler but potentially losing the explicit long-term memory that
LSTM's cell state provides.

---

## GRU Gate Equations

```
Update gate:  z_t = σ(W_z · [h_{t-1}, x_t])
    Controls how much of the past to keep vs how much to update.
    z_t ≈ 1 → keep past hidden state (stable period, no change)
    z_t ≈ 0 → replace with new candidate (event detected)

Reset gate:   r_t = σ(W_r · [h_{t-1}, x_t])
    Controls how much of the past to use when computing the candidate.
    r_t ≈ 0 → ignore past completely (fresh start after a gap)
    r_t ≈ 1 → fully use past (continuation of ongoing pattern)

Candidate:    h̃_t = tanh(W · [r_t ⊗ h_{t-1}, x_t])
    Proposed new hidden state, modulated by the reset gate.

Output:       h_t = (1 - z_t) ⊗ h_{t-1} + z_t ⊗ h̃_t
    Linear interpolation between past and candidate.
```

For NILM: when the fridge compressor turns on (sudden power jump),
the update gate z_t opens wide, recording this event into the hidden
state. During the steady running period, z_t stays small, carrying
the "fridge is ON" information forward unchanged.

---

## Key Design Decisions

### 1. Conv1D before GRU
Following WGRU architecture. Conv1D extracts local patterns from
the DWT sub-bands (shapes, edges, oscillations) before GRU
integrates them temporally. Without Conv1D, the GRU would process
raw sub-band values — less informative and harder to learn from.

### 2. Unidirectional (forward only)
This baseline uses unidirectional GRU to represent the standard
GRU approach. The hidden state at position 240 (center) contains
information from positions 0-240 (the past) but NOT from positions
241-480 (the future). This is a known limitation addressed by the
BiLSTM baseline and the proposed BiGRU model.

### 3. Center extraction vs last timestep
Original WGRU uses the last timestep's output. We use the CENTER
timestep following the seq2point principle (Zhang et al. 2018).
This is more consistent across all baselines in our benchmark.

### 4. Hidden size 128
WGRU original uses 64 hidden units per GRU layer. We use 128 for
fair comparison with BiLSTM (which also uses 128). The SAED paper
shows WGRU with 270K parameters — our 128-hidden GRU is comparable.

---

## Parameters

| Component | Parameters |
|---|---|
| Conv1D (6→64, k=5) | 1,984 |
| GRU layer 1 (64→128) | 74,496 |
| GRU layer 2 (128→128) | 99,072 |
| Dense (128→64) | 8,256 |
| Power heads (64→1) × 5 | 325 |
| State heads (64→1) × 5 | 325 |
| **Total** | **~184K** |

INT8 size: ~0.184 MB — easily deployable on STM32MP2 NPU.

---

## Expected Performance (Reference)

From SAED paper (Virtsionis et al., 2021), WGRU results on UK-DALE:

| Appliance | Category | F1 | MAE (W) |
|---|---|---|---|
| Fridge | Cat 1 (same house) | 0.63 | 33.29 |
| Fridge | Cat 2 (diff house) | 0.82 | 28.46 |
| Kettle | Cat 1 | 0.65 | 7.35 |
| Kettle | Cat 2 | 0.90 | 14.04 |
| Washing M. | Cat 1 | 0.54 | 16.55 |
| Dishwasher | Cat 1 | 0.33 | 13.22 |
| Microwave | Cat 1 | 0.32 | 6.29 |

From NILMFormer (Petralia et al., KDD 2025), BiGRU baseline:

| Appliance | MAE (W) | MR |
|---|---|---|
| Dishwasher | 27.2 | 0.408 |
| Fridge | 36.7 | 0.364 |
| Kettle | 11.6 | 0.618 |
| Microwave | 12.4 | 0.040 |
| Washing machine | 21.1 | 0.086 |

Our GRU with DWT input should improve over plain WGRU numbers.

---

## Differences From Original WGRU

| Aspect | WGRU (Krystalakos 2018) | Our Implementation |
|---|---|---|
| Framework | TensorFlow | PyTorch |
| Input | (1, 50-100) raw power | (6, 480) DWT + temporal |
| Conv layers | 1× Conv1D | 1× Conv1D (same) |
| GRU layers | 2× Bidirectional GRU | 2× Unidirectional GRU |
| Output strategy | Last timestep | Center timestep (seq2point) |
| Output | Single appliance scalar | N×2 multi-task (power + state) |
| Gating | None | ŷ = p̂ × σ(ŝ) |
| Parameters | ~270K | ~184K |
| Window size | 50-100 | 480 |

---

## Usage

```python
from models.gru_model import GRUBaseline

model = GRUBaseline(
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
@inproceedings{krystalakos2018sliding,
  title={Sliding window approach for online energy disaggregation
         using artificial neural networks},
  author={Krystalakos, Odysseas and Nalmpantis, Christoforos
          and Vrakas, Dimitris},
  booktitle={10th Hellenic Conference on Artificial Intelligence},
  pages={1--6},
  year={2018}
}

@article{virtsionis2021saed,
  title={SAED: Self-attentive energy disaggregation},
  author={Virtsionis Gkalinikis, Nikolaos and Nalmpantis, Christoforos
          and Vrakas, Dimitris},
  journal={Machine Learning},
  pages={1--20},
  year={2021},
  publisher={Springer},
  doi={10.1007/s10994-021-06106-3}
}

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
```
