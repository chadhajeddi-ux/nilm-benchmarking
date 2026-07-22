**CNN Baseline** ** : ** ** NILM Seq2Point**  
**Reference**  
**Paper:** Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring  
   
 **Authors:** Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, Charles Sutton  
   
 **Conference:** AAAI 2018 (32nd AAAI Conference on Artificial Intelligence)  
   
 **DOI:** [https://doi.org/10.1609/aaai.v32i1.11873  
   
 **arXiv:** ](https://doi.org/10.1609/aaai.v32i1.11873 "https://doi.org/10.1609/aaai.v32i1.11873")[https://arxiv.org/abs/1612.09106  
   
 **Official code:** ](https://arxiv.org/abs/1612.09106 "https://arxiv.org/abs/1612.09106")[https://github.com/MingjunZhong/seq2point-nilm](https://github.com/MingjunZhong/seq2point-nilm "https://github.com/MingjunZhong/seq2point-nilm")  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUeCE4yeIiT9CRVMWGAjJK2CbjNzVGcAAPzF2qu7Wl9PAAB47XoA/vcF8exqpY4AAAAASUVORK5CYII=)  
**What This Model Does**  
CNN baseline for NILM using the seq2point prediction strategy.  
   
 The model receives a window of 480 aggregate power timesteps and  
   
 predicts the power consumption AND ON/OFF state of each appliance  
   
 at the CENTER timestep only (position 240 out of 480).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsad4FCtY9ecwnkms4E2ELcGWmTmrKwAA/uLeqrU6vp4AAPDa/gDzUgM9+S8z3AAAAABJRU5ErkJggg==)  
**Architecture**  
Input (B, 6, 480)  
     │   6 channels = 4 DWT sub-bands (LP3, HP3, HP2, HP1)  
     │             + 2 temporal features (sin/cos hour-of-day)  
     │   480 timesteps = 48 minutes at 6-second sampling  
     │  
     ▼  
 Conv1D(in=6,  out=30, kernel=10, padding=same) + ReLU → (B, 30, 480)  
 Conv1D(in=30, out=30, kernel=8,  padding=same) + ReLU → (B, 30, 480)  
 Conv1D(in=30, out=40, kernel=6,  padding=same) + ReLU → (B, 40, 480)  
 Conv1D(in=40, out=50, kernel=5,  padding=same) + ReLU → (B, 50, 480)  
 Conv1D(in=50, out=50, kernel=5,  padding=same) + ReLU → (B, 50, 480)  
     │  
     │   Seq2point: take ONLY center timestep [index 240]  
     ▼  
 Center slice → (B, 50)  
     │  
     ▼  
 Linear(50 → 1024) + ReLU + Dropout(0.5) → (B, 1024)  
     │  
     ▼  
 For each appliance k (k = 1..5):  
     Power head k : Linear(1024 → 1) + ReLU → p̂_k ∈ [0, 1]  (normalized Watts)  
     State head k : Linear(1024 → 1)         → ŝ_logit,k     (raw logit)  
     Gated output : ŷ_k = p̂_k × σ(ŝ_logit,k)               (physically consistent)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSdYxZ4/mJjEsxE8W8GbCFuCLTOzVXsAAPzFuVZ3dXw9AQDgtesBxPEF3bv7x0IAAAAASUVORK5CYII=)  
**Key Design Decisions**  
**1. Conv1D instead of Conv2D**  
The original paper reshapes input to (1, 599, 1) and uses Conv2D  
   
 with kernel (k, 1). This is mathematically equivalent to Conv1D  
   
 with kernel k on a 1D sequence. Conv1D is the cleaner PyTorch  
   
 implementation for time series.  
**2. Center-slice instead of Flatten**  
The original code flattens ALL timesteps → Dense(29950 → 1024)  
   
 = 30.7 million parameters in one layer. This overfits on NILM data.  
Our implementation extracts ONLY the center timestep → Dense(50 → 1024)  
   
 = 51,200 parameters — 600× fewer. This is the correct interpretation  
   
 of seq2point: the model predicts FROM the center context, not by  
   
 memorizing all positions.  
**3. Multi-task output**  
The original paper trains one model per appliance with a single  
   
 scalar output. We use one shared model with N=5 parallel output  
   
 head pairs , more efficient, enables multi-task learning synergy.  
**4. Gated output**  
The original paper has no gated output ,  power prediction is  
   
 independent of state prediction. We add:  
   
 ŷ = p̂ × σ(ŝ_logit)  
   
 Source: Mamba-ECA-UNet (Fan et al., PSETC 2025).  
   
 Physical meaning: when the appliance is predicted OFF (σ → 0),  
   
 the power output is suppressed to zero automatically.  
**5. Hierarchical kernel sizes (10 → 8 → 6 → 5 → 5)**  
Each layer sees a progressively narrower local view:  
   
 kernel=10 → ~1 minute of context  (broad patterns)  
   
 kernel=8  → ~48 seconds  
   
 kernel=6  → ~36 seconds  
   
 kernel=5  → ~30 seconds (fine patterns, repeated × 2)  
   
 Stacking them creates a hierarchy from coarse to fine feature detection.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKY/jbnMIJ7FCt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85TgYRMv3/cwAAAABJRU5ErkJggg==)  
**Parameters**  
| | |  
|-|-|  
| **Component** | **Parameters** |   
| Conv block 1 (6→30, k=10) | 1,830 |   
| Conv block 2 (30→30, k=8) | 7,230 |   
| Conv block 3 (30→40, k=6) | 7,240 |   
| Conv block 4 (40→50, k=5) | 10,050 |   
| Conv block 5 (50→50, k=5) | 12,550 |   
| Dense (50→1024) | 52,224 |   
| Power heads (1024→1) × 5 | 5,125 |   
| State heads (1024→1) × 5 | 5,125 |   
| **Total** | **~101K** |   
   
INT8 size: ~0.101 MB ,  easily deployable on STM32MP2 NPU.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJGkPcrHpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaJkELjbMzy0AAAAASUVORK5CYII=)  
**Prediction Strategy** ** :** ** Seq2Point**  
Aggregate window:  [t-240, ..., t-1, t, t+1, ..., t+240]  
                                        ↑  
                                   center (index 240)  
   
 Model input:  all 480 timesteps (bilateral context)  
 Model output: prediction at t only  
   
 For streaming inference on STM32MP2:  
     Maintain rolling buffer of 480 points.  
     Every 6 seconds: add new measurement, drop oldest.  
     Run one inference → prediction for t-240 (24 minutes ago).  
     This 24-minute delay is acceptable for energy monitoring.  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSeYxZw/lVeDGMACBrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA6fOBdd+dKAKAAAAAElFTkSuQmCC)  
**Differences From Original Paper**  
| | | |  
|-|-|-|  
| **Aspect** | **Zhang et al. 2018** | **Our Implementation** |   
| Framework | TensorFlow/Keras | PyTorch |   
| Input shape | (1, 599, 1) | (6, 480) |   
| Input features | Raw power only | DWT sub-bands + temporal |   
| Conv type | Conv2D | Conv1D |   
| After last conv | Flatten all (29,950) | Center slice (50) |   
| Dense | 29,950 → 1024 | 50 → 1024 |   
| Output | 1 value (1 appliance) | N×2 (power + state, 5 appliances) |   
| Gating | None | ŷ = p̂ × σ(ŝ) |   
| Multi-task | No | Yes |   
| Parameters | ~30M (dense layer alone) | ~101K total |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKa/i8WMIR7ECt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85PAYartXEogAAAABJRU5ErkJggg==)  
**Training Configuration**  
optimizer    = Adam(lr=1e-3)  
 scheduler    = ReduceLROnPlateau(factor=0.5, patience=5)  
 batch_size   = 32  
 max_epochs   = 50  
 early_stop   = 10 epochs patience  
 dropout      = 0.5 (after dense layer)  
 loss         = L_state (Focal) + L_power (SmoothL1) + L_gated (SmoothL1)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNhYMEBIpD4ArCJDyywEZJWQZeZOaorAAD+4l6rrTq/ngAA8Nr+AEqmA1hl45m5AAAAAElFTkSuQmCC)  
**Expected Performance (Reference)**  
From NILMFormer (Petralia et al., KDD 2025) Table 2,  
   
 FCN baseline on UK-DALE (window=512, closest to our 480):  
| | | |  
|-|-|-|  
| **Appliance** | **MAE (W)** | **MR** |   
| Dishwasher | 45.3 | 0.171 |   
| Fridge | 28.0 | 0.478 |   
| Kettle | 29.2 | 0.353 |   
| Microwave | 17.0 | 0.027 |   
| Washing machine | 38.3 | 0.058 |   
   
Our CNN with DWT + multi-task + gating should improve on these numbers.  
   
 Results will be updated after training experiments.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/kSGMYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4qrBdGuSdJuAAAAAElFTkSuQmCC)  
**Usage**  
from models.cnn_model import CNNBaseline  
   
 model = CNNBaseline(  
     in_channels=6,       # 4 DWT + 2 temporal  
     window_size=480,  
     n_appliances=5,  
     dropout=0.5,  
 )  
   
 # Forward pass  
 x = torch.randn(32, 6, 480)        # batch of 32 windows  
 powers, states, gated = model(x)   # all shape (32, 5)  
   
 # Denormalize power to Watts  
 watts = dataset.denormalize_power(powers)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3KsQ0AIRAEsUW6Qij1KvnevhMSYmKQ7GiCGd09k3wBAOAVf+2o4wYAwE1qAdYuAy151mgcAAAAAElFTkSuQmCC)  
**Citation**  
@inproceedings{zhang2018sequence,  
   title={Sequence-to-point learning with neural networks  
          for non-intrusive load monitoring},  
   author={Zhang, Chaoyun and Zhong, Mingjun and Wang, Zongzuo  
           and Goddard, Nigel and Sutton, Charles},  
   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},  
   volume={32},  
   number={1},  
   year={2018},  
   doi={10.1609/aaai.v32i1.11873}  
 }  
   
