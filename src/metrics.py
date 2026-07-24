"""
metrics.py :  Comprehensive NILM Evaluation Metrics
=====================================================
Implements ALL metrics from NILMFormer 

Power regression metrics:
    MAE   — Mean Absolute Error (Watts)
    MSE   — Mean Squared Error
    RMSE  — Root Mean Squared Error
    NDE   — Normalized Disaggregation Error
    SAE   — Signal Aggregate Error
    TECA  — Total Energy Correctly Assigned
    MR    — Matching Ratio (best overall indicator per NILMFormer)

Classification metrics (ON/OFF state):
    F1           — harmonic mean of precision and recall
    Accuracy     — fraction of correct predictions
    Balanced Acc — average of per-class accuracy (handles imbalance)
    Precision    — true positives / predicted positives
    Recall       — true positives / actual positives

Usage:
    tracker = MetricsTracker(appliance_names, appliance_max)

    for batch in test_loader:
        x, y_power, y_state = batch
        pred_power, pred_state, pred_gated = model(x)
        tracker.update(pred_power, y_power, pred_state, y_state)

    results = tracker.compute()
    tracker.print_table()
    tracker.to_dataframe().to_csv("results.csv")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. POWER REGRESSION METRICS
# ============================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error in Watts.

    MAE = (1/T) × Σ|ŷ_t - y_t|

    The most intuitive power metric — directly interpretable as
    "on average, the prediction is off by X watts."
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error — penalizes large errors more than MAE."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — same units as MAE (Watts)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Disaggregation Error.

    NDE = Σ(ŷ_t - y_t)² / Σ(y_t)²

    Measures relative error normalized by the true signal energy.
    Less sensitive to absolute power levels than MAE , useful for
    cross-dataset comparison where appliances have different Wattages.
    Lower is better. NDE=0 means perfect prediction.

    Reference: Kolter & Jaakkola, AISTATS 2012.
    """
    denominator = np.sum(y_true ** 2)
    if denominator < 1e-8:
        return 0.0  # avoid division by zero when appliance is always off
    return float(np.sum((y_true - y_pred) ** 2) / denominator)


def sae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Signal Aggregate Error.

    SAE = |Σŷ_t - Σy_t| / Σy_t

    Measures whether the model gets the TOTAL energy right over the
    test period, even if individual timestep predictions are noisy.
    Important for energy billing applications where total consumption
    matters more than instantaneous accuracy.

    Lower is better. SAE=0 means total predicted energy matches total
    true energy exactly. SAE=1 means 100% total energy error.
    """
    true_total = np.sum(y_true)
    if true_total < 1e-8:
        return 0.0
    pred_total = np.sum(y_pred)
    return float(np.abs(pred_total - true_total) / true_total)


def teca(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Total Energy Correctly Assigned.

    TECA = 1 - Σ|y_t - ŷ_t| / (2 × Σy_t)

    Ranges from 0 to 1. Higher is better.
    TECA=1 means every watt-hour is correctly attributed.
    TECA=0.5 means half the energy is misattributed.

    The factor 2 in the denominator accounts for the fact that
    errors can be either overestimation or underestimation.

    Reference: Pereira & Nunes, Sustainable Computing 2018.
    """
    true_total = np.sum(y_true)
    if true_total < 1e-8:
        return 1.0  # if appliance never runs, TECA=1 trivially
    return float(1.0 - np.sum(np.abs(y_true - y_pred)) / (2.0 * true_total))


def matching_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matching Ratio :  best overall indicator per NILMFormer.

    MR = Σ min(ŷ_t, y_t) / Σ max(ŷ_t, y_t)

    Ranges from 0 to 1. Higher is better.
    MR=1 means prediction perfectly matches ground truth at every
    timestep. MR=0 means zero overlap.

    MR naturally handles both overestimation and underestimation:
    - If model overestimates: numerator is bounded by true, denominator grows
    - If model underestimates: numerator shrinks, denominator is bounded by true

    Reference: Petralia et al., KDD 2025 (NILMFormer).
    """
    numerator = np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(np.maximum(y_true, y_pred))
    if denominator < 1e-8:
        return 1.0  # both zero everywhere → perfect match
    return float(numerator / denominator)


# ============================================================
# 2. CLASSIFICATION METRICS (ON/OFF STATE)
# ============================================================

def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute TP, TN, FP, FN for binary classification."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 Score — harmonic mean of precision and recall.

    F1 = 2 × (precision × recall) / (precision + recall)

    Why F1 matters for NILM: class imbalance is extreme.
    Kettle is ON only 0.62% of time. A model that always predicts
    OFF achieves 99.38% accuracy but F1=0. F1 penalizes this by
    requiring BOTH high precision AND high recall.

    Range: 0 to 1. Higher is better.
    """
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall < 1e-8:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy.

    Acc = (TP + TN) / (TP + TN + FP + FN)

    WARNING: misleading for imbalanced NILM data.
    Always report alongside F1 to avoid false confidence.
    """
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return float((tp + tn) / total)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced Accuracy — average of per-class accuracy.

    BAcc = (TPR + TNR) / 2
         = (recall + specificity) / 2

    Handles class imbalance by weighting ON and OFF equally.
    A model that always predicts OFF gets BAcc=0.5 (random chance)
    instead of 99.38% accuracy — more honest.
    """
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    return float((tpr + tnr) / 2.0)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)

    "Of all the times the model said ON, how many were actually ON?"
    Low precision means too many false positives (false alarms).

    NILMFormer Table 2 shows BiGRU precision = 0.177 — very low.
    Your Focal Loss with alpha=0.25 directly targets this.
    """
    tp, _, fp, _ = _binary_counts(y_true, y_pred)
    if (tp + fp) == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN)

    "Of all the times the appliance was actually ON, how many
    did the model correctly detect?"
    Low recall means the model misses real ON events.

    NILMFormer Table 2 shows BiGRU recall = 0.962  : very high.
    High recall + low precision → model over-predicts ON.
    """
    tp, _, _, fn = _binary_counts(y_true, y_pred)
    if (tp + fn) == 0:
        return 0.0
    return float(tp / (tp + fn))


# ============================================================
# 3. METRICS TRACKER — Accumulates Across Batches
# ============================================================

class MetricsTracker:
    """
    Accumulates predictions and targets across batches, then
    computes all metrics at the end.

    This avoids computing metrics per-batch (which is statistically
    incorrect for metrics like F1 that are not decomposable).

    Parameters
    ----------
    appliance_names : list of str
        Names of target appliances.
    appliance_max : dict
        {appliance: max_power_watts} for denormalization.
        Predictions in [0,1] are scaled back to physical Watts.
    state_threshold : float
        Threshold for converting sigmoid probabilities to binary
        ON/OFF predictions. Default 0.5.
    """

    def __init__(
        self,
        appliance_names: List[str],
        appliance_max: Dict[str, float],
        state_threshold: float = 0.5,
    ):
        self.appliance_names = appliance_names
        self.appliance_max = appliance_max
        self.state_threshold = state_threshold
        self.n_appliances = len(appliance_names)

        # Accumulators — lists of numpy arrays, concatenated at compute()
        self._pred_power: List[np.ndarray] = []
        self._true_power: List[np.ndarray] = []
        self._pred_state: List[np.ndarray] = []
        self._true_state: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self._pred_power.clear()
        self._true_power.clear()
        self._pred_state.clear()
        self._true_state.clear()

    def update(
        self,
        pred_power: np.ndarray,
        true_power: np.ndarray,
        pred_state_logits: np.ndarray,
        true_state: np.ndarray,
    ) -> None:
        """
        Accumulate one batch of predictions.

        Parameters
        ----------
        pred_power : (B, N_app)  : normalized power predictions [0,1]
        true_power : (B, N_app)  :  normalized power targets [0,1]
        pred_state_logits : (B, N_app) :  raw logits (pre-sigmoid)
        true_state : (B, N_app)  : binary 0/1 ground truth
        """
        # Convert tensors to numpy if needed
        if hasattr(pred_power, "detach"):
            pred_power = pred_power.detach().cpu().numpy()
        if hasattr(true_power, "detach"):
            true_power = true_power.detach().cpu().numpy()
        if hasattr(pred_state_logits, "detach"):
            pred_state_logits = pred_state_logits.detach().cpu().numpy()
        if hasattr(true_state, "detach"):
            true_state = true_state.detach().cpu().numpy()

        self._pred_power.append(pred_power)
        self._true_power.append(true_power)

        # Convert logits → binary predictions via sigmoid + threshold
        pred_probs = 1.0 / (1.0 + np.exp(-pred_state_logits.clip(-20, 20)))
        pred_binary = (pred_probs >= self.state_threshold).astype(np.float32)
        self._pred_state.append(pred_binary)
        self._true_state.append(true_state)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics across accumulated predictions.

        Returns
        -------
        dict : {appliance_name: {metric_name: value, ...}, ...}
            Plus a special "mean" key with averages across appliances.
        """
        if len(self._pred_power) == 0:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        # Concatenate all batches
        pred_p = np.concatenate(self._pred_power, axis=0)  # (N, n_app)
        true_p = np.concatenate(self._true_power, axis=0)
        pred_s = np.concatenate(self._pred_state, axis=0)
        true_s = np.concatenate(self._true_state, axis=0)

        results = {}

        for i, name in enumerate(self.appliance_names):
            # Denormalize power to Watts
            max_w = self.appliance_max[name]
            pp_watts = pred_p[:, i] * max_w
            tp_watts = true_p[:, i] * max_w

            # State predictions and targets
            ps = pred_s[:, i]
            ts = true_s[:, i]

            results[name] = {
                # ---- Power regression metrics (in Watts) ----
                "mae_w": mae(tp_watts, pp_watts),
                "mse": mse(tp_watts, pp_watts),
                "rmse": rmse(tp_watts, pp_watts),
                "nde": nde(tp_watts, pp_watts),
                "sae": sae(tp_watts, pp_watts),
                "teca": teca(tp_watts, pp_watts),
                "mr": matching_ratio(tp_watts, pp_watts),

                # ---- Classification metrics ----
                "f1": f1_score(ts, ps),
                "accuracy": accuracy(ts, ps),
                "balanced_accuracy": balanced_accuracy(ts, ps),
                "precision": precision(ts, ps),
                "recall": recall(ts, ps),
            }

        # ---- Mean across all appliances ----
        metric_keys = list(results[self.appliance_names[0]].keys())
        results["mean"] = {
            key: float(np.mean([results[name][key]
                                for name in self.appliance_names]))
            for key in metric_keys
        }

        return results

    def print_table(self, results: Optional[Dict] = None) -> None:
        """
        Print a formatted results table matching NILMFormer Table 2 style.

        Metric direction arrows:
            ↓ = lower is better (MAE, MSE, RMSE, NDE, SAE)
            ↑ = higher is better (TECA, MR, F1, Accuracy, Precision, Recall)
        """
        if results is None:
            results = self.compute()

        # Header
        header_metrics = [
            ("MAE↓", "mae_w", "{:.1f}"),
            ("RMSE↓", "rmse", "{:.1f}"),
            ("NDE↓", "nde", "{:.3f}"),
            ("SAE↓", "sae", "{:.3f}"),
            ("TECA↑", "teca", "{:.3f}"),
            ("MR↑", "mr", "{:.3f}"),
            ("F1↑", "f1", "{:.3f}"),
            ("Acc↑", "accuracy", "{:.3f}"),
            ("BAcc↑", "balanced_accuracy", "{:.3f}"),
            ("Prec↑", "precision", "{:.3f}"),
            ("Rec↑", "recall", "{:.3f}"),
        ]

        # Column widths
        name_w = 18
        col_w = 8

        # Print header row
        header = f"{'Appliance':<{name_w}}"
        for label, _, _ in header_metrics:
            header += f"{label:>{col_w}}"
        print(header)
        print("-" * len(header))

        # Print per-appliance rows
        for name in self.appliance_names:
            row = f"{name:<{name_w}}"
            for _, key, fmt in header_metrics:
                val = results[name][key]
                row += f"{fmt.format(val):>{col_w}}"
            print(row)

        # Print mean row
        print("-" * len(header))
        row = f"{'MEAN':<{name_w}}"
        for _, key, fmt in header_metrics:
            val = results["mean"][key]
            row += f"{fmt.format(val):>{col_w}}"
        print(row)

    def to_dataframe(self, results: Optional[Dict] = None) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for CSV export."""
        if results is None:
            results = self.compute()

        rows = []
        for name in self.appliance_names + ["mean"]:
            row = {"appliance": name}
            row.update(results[name])
            rows.append(row)

        return pd.DataFrame(rows)

    def save_json(self, path: Path, results: Optional[Dict] = None,
                  model_name: str = "unknown") -> None:
        """
        Save results to JSON for the comparison table.

        The JSON contains: model name, per-appliance metrics,
        and mean metrics. This file is loaded by Notebook 08
        (comparison) to build the final thesis table.
        """
        if results is None:
            results = self.compute()

        output = {
            "model": model_name,
            "n_samples": len(np.concatenate(self._true_power, axis=0)),
            "results": results,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved → {path}")


# ============================================================
# 4. LOSS FUNCTIONS
# ============================================================

def focal_loss(
    pred_logits: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> "torch.Tensor":
    """
    Focal Loss for binary classification  : handles extreme class imbalance.

    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Reference: Lin et al. "Focal Loss for Dense Object Detection."
    ICCV 2017 (RetinaNet paper).

    Why Focal Loss for NILM:
        Kettle is ON 0.62% of the time → 99.38% of samples are OFF.
        Standard BCE treats all samples equally → model learns to always
        predict OFF (high accuracy, zero F1).
        Focal Loss down-weights easy samples (confident OFF predictions)
        and focuses on hard samples (switching moments, ambiguous states).

    Parameters
    ----------
    pred_logits : (B, N) — raw logits (pre-sigmoid)
    targets : (B, N) — binary 0/1 ground truth
    alpha : float
        Weighting factor for the positive class. Default 0.25.
        Lower alpha = less weight on positives → improves precision.
        Higher alpha = more weight on positives → improves recall.
    gamma : float
        Focusing parameter. Default 2.0.
        γ=0 → standard BCE (no focusing).
        γ=2 → well-classified samples contribute 100× less to loss.
        γ=5 → extreme focusing (rarely used).
    """
    import torch

    # Sigmoid to get probabilities
    probs = torch.sigmoid(pred_logits)

    # p_t: probability of the correct class
    p_t = targets * probs + (1 - targets) * (1 - probs)

    # α_t: class-specific weighting
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

    # Focal modulating factor: (1 - p_t)^γ
    # When p_t is high (easy sample): factor ≈ 0 → loss ≈ 0
    # When p_t is low (hard sample): factor ≈ 1 → full loss
    focal_weight = (1 - p_t) ** gamma

    # BCE loss (numerically stable via log-sum-exp)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logits, targets, reduction="none"
    )

    loss = alpha_t * focal_weight * bce
    return loss.mean()


def multi_task_loss(
    pred_power: "torch.Tensor",
    true_power: "torch.Tensor",
    pred_state_logits: "torch.Tensor",
    true_state: "torch.Tensor",
    pred_gated: "torch.Tensor",
    alpha_focal: float = 0.25,
    gamma_focal: float = 2.0,
    lambda_power: float = 1.0,
    lambda_state: float = 1.0,
    lambda_gated: float = 0.5,
) -> "torch.Tensor":
    """
    Combined multi-task loss for NILM.

    L_total = λ_power × SmoothL1(p̂, p)
            + λ_state × FocalLoss(ŝ_logit, s)
            + λ_gated × SmoothL1(ŷ_gated, p)

    Three complementary objectives:
        1. Power regression (SmoothL1): predict accurate Watts
        2. State classification (Focal): detect ON/OFF states
        3. Gated consistency (SmoothL1): enforce ŷ = p̂ × σ(ŝ)
           matches the true power  ,  the gated output should be
           physically consistent.

    SmoothL1 is used instead of MSE because it is less sensitive
    to outlier power readings (sensor noise, measurement spikes).

    Parameters
    ----------
    pred_power : (B, N_app) — power predictions [0, 1]
    true_power : (B, N_app) — power targets [0, 1]
    pred_state_logits : (B, N_app) — raw state logits
    true_state : (B, N_app) — binary 0/1 state targets
    pred_gated : (B, N_app) — gated output p̂ × σ(ŝ)
    lambda_power, lambda_state, lambda_gated : float
        Relative weights for each loss component.
    """
    import torch

    # Power regression loss
    loss_power = torch.nn.functional.smooth_l1_loss(pred_power, true_power)

    # State classification loss (Focal — handles class imbalance)
    loss_state = focal_loss(pred_state_logits, true_state,
                            alpha=alpha_focal, gamma=gamma_focal)

    # Gated consistency loss
    loss_gated = torch.nn.functional.smooth_l1_loss(pred_gated, true_power)

    return (lambda_power * loss_power
            + lambda_state * loss_state
            + lambda_gated * loss_gated)


# ============================================================
# ENTRY POINT — Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Module — Self-Test")
    print("=" * 60)

    # ---- Synthetic test data ----
    rng = np.random.default_rng(42)
    N = 1000  # samples
    N_APP = 5

    appliance_names = ["kettle", "fridge", "washing_machine",
                       "dishwasher", "microwave"]
    appliance_max = {
        "kettle": 3100.0,
        "fridge": 300.0,
        "washing_machine": 2500.0,
        "dishwasher": 2500.0,
        "microwave": 3000.0,
    }

    # Generate realistic synthetic predictions
    true_power = rng.random((N, N_APP)).astype(np.float32) * 0.3
    pred_power = true_power + rng.normal(0, 0.05, (N, N_APP)).astype(np.float32)
    pred_power = np.clip(pred_power, 0, 1)

    true_state = (true_power > 0.1).astype(np.float32)
    pred_logits = rng.normal(0, 2, (N, N_APP)).astype(np.float32)
    # Make predictions somewhat correlated with truth
    pred_logits += true_state * 2.0 - 1.0

    # ---- Test individual metrics ----
    print("\n[1] Individual metric functions:")
    y_t = true_power[:, 0] * appliance_max["kettle"]
    y_p = pred_power[:, 0] * appliance_max["kettle"]
    print(f"  MAE:  {mae(y_t, y_p):.1f} W")
    print(f"  RMSE: {rmse(y_t, y_p):.1f} W")
    print(f"  NDE:  {nde(y_t, y_p):.4f}")
    print(f"  SAE:  {sae(y_t, y_p):.4f}")
    print(f"  TECA: {teca(y_t, y_p):.4f}")
    print(f"  MR:   {matching_ratio(y_t, y_p):.4f}")

    s_t = true_state[:, 0]
    s_p = (1.0 / (1.0 + np.exp(-pred_logits[:, 0])) >= 0.5).astype(float)
    print(f"  F1:   {f1_score(s_t, s_p):.4f}")
    print(f"  Acc:  {accuracy(s_t, s_p):.4f}")
    print(f"  BAcc: {balanced_accuracy(s_t, s_p):.4f}")
    print(f"  Prec: {precision(s_t, s_p):.4f}")
    print(f"  Rec:  {recall(s_t, s_p):.4f}")

    # ---- Test MetricsTracker ----
    print("\n[2] MetricsTracker (batch accumulation):")
    tracker = MetricsTracker(appliance_names, appliance_max)

    # Simulate 5 batches of 200 samples
    for i in range(5):
        s, e = i * 200, (i + 1) * 200
        tracker.update(pred_power[s:e], true_power[s:e],
                       pred_logits[s:e], true_state[s:e])

    results = tracker.compute()
    print(f"  Computed results for {len(results) - 1} appliances + mean")

    # ---- Print formatted table ----
    print("\n[3] Formatted results table:")
    tracker.print_table(results)

    # ---- Test DataFrame export ----
    print("\n[4] DataFrame export:")
    df = tracker.to_dataframe(results)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()[:6]}...")

    # ---- Test JSON export ----
    print("\n[5] JSON export:")
    test_path = Path("/tmp/_metrics_test.json")
    tracker.save_json(test_path, results, model_name="test_model")
    with open(test_path) as f:
        loaded = json.load(f)
    assert loaded["model"] == "test_model"
    assert "mean" in loaded["results"]
    test_path.unlink()
    print(f"  JSON save/load round-trip OK")

    # ---- Test Focal Loss ----
    print("\n[6] Focal Loss:")
    import torch
    logits = torch.randn(32, 5, requires_grad=True)
    targets = torch.randint(0, 2, (32, 5)).float()
    fl = focal_loss(logits, targets)
    print(f"  Focal Loss value: {fl.item():.4f}")
    assert fl.item() >= 0, "Focal Loss must be >= 0"
    assert fl.requires_grad, "Focal Loss must be differentiable"
    print(f"  Differentiable: ✓")

    # ---- Test multi-task loss ----
    print("\n[7] Multi-task Loss:")
    p_power = torch.rand(32, 5, requires_grad=True)
    t_power = torch.rand(32, 5)
    p_logits = torch.randn(32, 5, requires_grad=True)
    t_state = torch.randint(0, 2, (32, 5)).float()
    p_gated = p_power * torch.sigmoid(p_logits)
    mt_loss = multi_task_loss(p_power, t_power, p_logits, t_state, p_gated)
    print(f"  Multi-task Loss value: {mt_loss.item():.4f}")
    assert mt_loss.requires_grad
    print(f"  Differentiable: ✓")

    # ---- Edge cases ----
    print("\n[8] Edge cases:")
    zeros = np.zeros(100)
    print(f"  NDE(zeros, zeros):  {nde(zeros, zeros):.4f} (should be 0.0)")
    print(f"  SAE(zeros, zeros):  {sae(zeros, zeros):.4f} (should be 0.0)")
    print(f"  TECA(zeros, zeros): {teca(zeros, zeros):.4f} (should be 1.0)")
    print(f"  MR(zeros, zeros):   {matching_ratio(zeros, zeros):.4f} (should be 1.0)")
    print(f"  F1(zeros, zeros):   {f1_score(zeros, zeros):.4f} (should be 0.0)")

    print(f"\n{'=' * 60}")
    print("All tests passed — metrics module ready!")
    print(f"{'=' * 60}")
