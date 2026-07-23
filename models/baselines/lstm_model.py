"""
lstm_model.py  :  LSTM Baseline for NILM (seq2point)
====================================================
Architecture based on:
    Hwang & Kang. "Nonintrusive Load Monitoring Using an LSTM
    with Feedback Structure." IEEE TIM, 2022.
    DOI: https://doi.org/10.1109/TIM.2022.3169536

    Mauch & Yang. "A New Approach for Supervised Power
    Disaggregation by Using a Deep Recurrent LSTM Network."
    IEEE GlobalSIP 2015.

Unidirectional LSTM baseline  ,  provides temporal context
from past only (positions 0→center). Compare against BiLSTM
to quantify the value of future context at the center point.

Architecture:
    Conv1d(6→16, k=4, same) → LSTM(16→64) → LSTM(64→128)
    → center slice → Dense(128→64) → MultiTaskHeads
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from config import WINDOW_SIZE, INPUT_CHANNELS, N_APPLIANCES
from cnn_model import MultiTaskHeads


class LSTMBaseline(nn.Module):
    """
    Unidirectional LSTM baseline for NILM.

    Same structure as BiLSTM but processes only forward direction.
    At center t=240: sees positions 0..240 only (past context).
    Compare with BiLSTM to quantify bidirectionality contribution.

    LSTM gates:
        forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        cell update: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        cell state:  c_t = f_t ⊗ c_{t-1} + i_t ⊗ c̃_t
        output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        hidden:      h_t = o_t ⊗ tanh(c_t)

    The separate cell state c_t is LSTM's key difference from GRU —
    it provides a dedicated long-term memory channel, independent of
    the hidden state h_t used for output. In practice for NILM, this
    distinction has limited impact (NILMFormer shows GRU~=LSTM).
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        window_size: int = WINDOW_SIZE,
        n_appliances: int = N_APPLIANCES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.center = window_size // 2
        self.drop = nn.Dropout(dropout)

        self.conv = nn.Conv1d(in_channels, 16, kernel_size=4,
                              stride=1, padding='same', bias=True)

        # Unidirectional LSTM (bidirectional=False)
        self.lstm1 = nn.LSTM(16, 64, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True, bidirectional=False)

        self.dense = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(in_features=64, n_appliances=n_appliances)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.drop(x).permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(self.drop(x))
        center = x[:, self.center, :]
        out = self.dense(center)
        return self.heads(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 55)
    print("LSTM Baseline — Self-Test")
    print("Reference: Hwang & Kang, IEEE TIM 2022")
    print("=" * 55)

    model = LSTMBaseline()
    B = 8
    x = torch.randn(B, INPUT_CHANNELS, WINDOW_SIZE)
    powers, states, gated = model(x)

    assert powers.shape == (B, N_APPLIANCES)
    assert (powers >= 0).all()
    assert (gated <= powers + 1e-6).all()

    with torch.no_grad():
        c = model.conv(x)
        p = model.drop(c).permute(0, 2, 1)
        l1, _ = model.lstm1(p)
        l2, _ = model.lstm2(model.drop(l1))

    print(f"[Input]  {tuple(x.shape)}")
    print(f"[Output] powers={tuple(powers.shape)}")
    print(f"[Shapes] conv={tuple(c.shape)}, "
          f"lstm1={tuple(l1.shape)}, lstm2={tuple(l2.shape)}")
    print(f"[Params] {model.count_parameters():,} | "
          f"{model.count_parameters()/1e6:.3f} MB INT8")
    print(f"\n[LSTM vs BiLSTM]")
    print(f"  LSTM  hidden at center: 128 (past only, positions 0→240)")
    print(f"  BiLSTM hidden at center: 256 (full bilateral, 0→480)")

    print(f"\n{'=' * 55}")
    print("All tests passed — LSTM baseline ready!")
    print(f"{'=' * 55}")
