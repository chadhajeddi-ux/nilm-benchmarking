# CNN-LSTM Baseline  :  NILM Seq2Point

## References

### Primary
**Paper:** A Multi-Task Learning Model for NILM Based on
Discrete Wavelet Transform
**Authors:** Luo J., Liu S., Cai Z., Xiong C., Tu G.
**Journal:** The Journal of Supercomputing, 79:9021–9046, 2023
**DOI:** https://doi.org/10.1007/s11227-022-05000-6

### Supporting
**Paper:** Long-Term Recurrent Convolutional Networks for
Non-Intrusive Load Monitoring
**Authors:** Krystalakos et al.
**Conference:** ACM PETRA 2020

---

## What This Model Does

CNN-LSTM hybrid baseline for NILM using seq2point.
CNN blocks extract local multi-scale features from DWT sub-bands.
MaxPool(8) compresses temporal dimension (480→60) before LSTM.
LSTM models temporal dependencies across compressed features.

This is the most direct predecessor to the proposed model , 
adding bidirectionality, attention, and depthwise-separable
convolution to CNN-LSTM gives the full proposed architecture.

---

## Architecture

```
Input (B, 6, 480)
    │   6 channels = 4 DWT sub-bands + 2 temporal features
    │
    ▼
Conv1d(6→32, k=5) + ReLU + BN + Dropout(0.2)  → (B, 32, 480)
Conv1d(32→64, k=5) + ReLU + BN + Dropout(0.2) → (B, 64, 480)
Conv1d(64→64, k=5) + ReLU + BN + Dropout(0.2) → (B, 64, 480)
    │   3 blocks extract hierarchical local features
    │   BatchNorm stabilizes training across varying power levels
    │
    ▼
MaxPool1d(kernel=8) → (B, 64, 60)
    │   Temporal compression: 480 → 60 positions
    │   8× speedup for LSTM without significant loss
    │   NILM events span multiple timesteps so compression is safe
    │
    ▼
Permute → (B, 60, 64)
    │
    ▼
LSTM(64→128, layers=2, unidirectional) → (B, 60, 128)
    │   Models temporal dependencies in compressed space
    │   Past context only (positions 0..center_compressed)
    │
    ▼
Take center of compressed sequence [:, 30, :] → (B, 128)
    │   Position 30 corresponds to original position 240
    │   (480/2 / 8 = 30)
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

## Ablation Hierarchy

This model sits in a clear ablation chain:

```
CNN only
    → local features only, no temporal modeling
    → baseline: cnn_model.py

LSTM only
    → temporal modeling only, no local feature extraction
    → baseline: lstm_model.py

CNN + LSTM  (this model)
    → both local and temporal, but:
    → standard conv (not depthwise-separable)
    → unidirectional (not BiGRU)
    → no attention (no CBAM)
    → standard normalization (not DyT)

Proposed model
    → DepthwiseSep-Conv + BiGRU + Lite-CBAM(T) + DyT
    → all improvements combined
```

Each step from CNN-LSTM to Proposed adds a measurable contribution.
The ablation study on Day 19 quantifies each contribution separately.

---

## MaxPool Compression  :  Why It Works for NILM

At 6-second sampling, 480 timesteps = 48 minutes.
After MaxPool(8): 60 timesteps, each representing 48 seconds.

NILM appliance signatures:
- Kettle: ON for 2-3 minutes → spans 3-4 compressed timesteps ✓
- Fridge: ON for 10 minutes → spans 13 compressed timesteps ✓
- Washing machine: 45-minute cycle → spans all 60 timesteps ✓
- Microwave: 30 seconds → may be lost in compression ⚠️

The microwave risk is why the proposed model uses full-resolution
BiGRU (no MaxPool)  ,preserving all 480 timesteps for the LSTM.

--- 

## Parameters

| Component | Parameters |
|---|---|
| Conv block 1 (6→32, k=5) | 992 |
| Conv block 2 (32→64, k=5) | 10,304 |
| Conv block 3 (64→64, k=5) | 20,544 |
| LSTM layer 1 (64→128) | 98,816 |
| LSTM layer 2 (128→128) | 131,584 |
| Dense (128→64) | 8,256 |
| Power heads (64→1) × 5 | 325 |
| State heads (64→1) × 5 | 325 |
| **Total** | **~271K** |

INT8 size: ~0.271 MB  ,  deployable on STM32MP2 NPU.

---

## Usage

```python
from models.baselines.cnn_lstm_model import CNNLSTMBaseline

model = CNNLSTMBaseline(
    in_channels=6,
    window_size=480,
    n_appliances=5,
    conv_channels=[32, 64, 64],
    pool_size=8,
    lstm_hidden=128,
    dropout=0.2,
)

x = torch.randn(32, 6, 480)
powers, states, gated = model(x)
```

---

## Citations

```bibtex
@article{luo2023multi,
  title={A multi-task learning model for non-intrusive load
         monitoring based on discrete wavelet transform},
  author={Luo, Jie and Liu, Shubo and Cai, Zhaohui
          and Xiong, Chang and Tu, Guoqing},
  journal={The Journal of Supercomputing},
  volume={79},
  pages={9021--9046},
  year={2023},
  doi={10.1007/s11227-022-05000-6}
}
```
