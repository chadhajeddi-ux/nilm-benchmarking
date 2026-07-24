"""
train.py  : Unified Training Loop for NILM Models
====================================================
Trains any of the 8 models (7 baselines + BiWave-NILM) with the
same code. All models share the same forward signature:
    powers, states, gated = model(x)

Training modes:
    1. Normal: multi-task loss (SmoothL1 + Focal + Gated)
    2. Distillation: + KL divergence from teacher soft labels
    3. Domain adaptation: + GRL adversarial loss (optional)

Best checkpoint selected by Matching Ratio (MR)  :  the best
overall NILM indicator per NILMFormer (Petralia et al., KDD 2025).

Usage:
    # Single model
    python src/train.py --model biwave --dataset ukdale --epochs 50

    # All baselines + proposed
    python src/train.py --all --dataset ukdale --epochs 50

    # With distillation from teacher
    python src/train.py --model biwave --dataset ukdale \
        --teacher checkpoints/mamba_teacher.pth --T 4

    # Quick test (small subset, 3 epochs)
    python src/train.py --model biwave --dataset ukdale --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "proposed"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

from config import (
    WINDOW_SIZE,
    INPUT_CHANNELS,
    N_APPLIANCES,
    APPLIANCE_NAMES,
    APPLIANCES,
    BATCH_SIZE,
    TRAIN_STRIDE,
    TEST_STRIDE,
    SEED,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
)
from metrics import MetricsTracker, multi_task_loss
from dataset import (
    NILMDataset,
    NormStats,
    split_train_val,
    build_dataloaders,
)


# ============================================================
# 1. MODEL REGISTRY  : maps name → class
# ============================================================

def get_model_registry() -> Dict[str, type]:
    """
    Lazy import of all model classes.
    Returns a dict mapping model name strings to model classes.
    """
    from cnn_model import CNNBaseline
    from gru_model import GRUBaseline
    from bigru_model import BiGRUBaseline
    from lstm_model import LSTMBaseline
    from bilstm_model import BiLSTMBaseline
    from cnn_lstm_model import CNNLSTMBaseline
    from nilmformer_model import NILMFormerModel
    try:
        from proposed_model import BiWaveNILM
    except ImportError:
        from proposed_model import ProposedModel as BiWaveNILM

    return {
        "cnn": CNNBaseline,
        "gru": GRUBaseline,
        "bigru": BiGRUBaseline,
        "lstm": LSTMBaseline,
        "bilstm": BiLSTMBaseline,
        "cnn_lstm": CNNLSTMBaseline,
        "nilmformer": NILMFormerModel,
        "biwave": BiWaveNILM,
    }


# ============================================================
# 2. TRAINING  : One Epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    distil_alpha: float = 0.5,
    distil_T: float = 4.0,
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The student model to train.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer (Adam/AdamW).
    device : torch.device
        Training device (cuda/cpu).
    teacher_model : nn.Module, optional
        Frozen teacher for knowledge distillation. Default None.
    distil_alpha : float
        Weight for task loss vs distillation loss.
        L = alpha * L_task + (1-alpha) * L_distil. Default 0.5.
    distil_T : float
        Temperature for soft label distillation. Default 4.0.

    Returns
    -------
    float : average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y_power, y_state in tqdm(loader, desc="  Train", leave=False):
        x = x.to(device)
        y_power = y_power.to(device)
        y_state = y_state.to(device)

        # Forward pass
        pred_power, pred_state, pred_gated = model(x)

        # Multi-task loss
        loss_task = multi_task_loss(
            pred_power, y_power,
            pred_state, y_state,
            pred_gated,
        )

        # Optional: knowledge distillation
        if teacher_model is not None:
            with torch.no_grad():
                t_power, t_state, _ = teacher_model(x)

            # Temperature-scaled soft label distillation
            # Reference: Hur et al. Sensors 2022; Hinton et al. 2015
            teacher_soft = torch.sigmoid(t_state / distil_T)
            student_soft = torch.sigmoid(pred_state / distil_T)

            loss_distil = (distil_T ** 2) * nn.functional.binary_cross_entropy(
                student_soft, teacher_soft.detach(),
            )

            loss = distil_alpha * loss_task + (1 - distil_alpha) * loss_distil
        else:
            loss = loss_task

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients in GRU/LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# 3. VALIDATION  : One Epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tracker: MetricsTracker,
) -> Tuple[float, Dict]:
    """
    Validate for one epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    tracker : MetricsTracker
        Accumulates predictions for metric computation.

    Returns
    -------
    (val_loss, metrics_dict)
        val_loss : float — average validation loss
        metrics_dict : dict — full metrics from MetricsTracker
    """
    model.eval()
    tracker.reset()
    total_loss = 0.0
    n_batches = 0

    for x, y_power, y_state in tqdm(loader, desc="  Valid", leave=False):
        x = x.to(device)
        y_power = y_power.to(device)
        y_state = y_state.to(device)

        pred_power, pred_state, pred_gated = model(x)

        loss = multi_task_loss(
            pred_power, y_power,
            pred_state, y_state,
            pred_gated,
        )

        total_loss += loss.item()
        n_batches += 1

        # Accumulate for metrics (on CPU)
        tracker.update(pred_power, y_power, pred_state, y_state)

    val_loss = total_loss / max(n_batches, 1)
    metrics = tracker.compute()
    return val_loss, metrics


# ============================================================
# 4. EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    Monitors MR (Matching Ratio) — higher is better.
    If MR does not improve for `patience` consecutive epochs,
    training stops to save GPU time and prevent overfitting.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Update with current epoch score.

        Returns True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ============================================================
# 5. TRAINING ORCHESTRATOR  : Full Pipeline
# ============================================================

def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stats: NormStats,
    device: torch.device,
    max_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    teacher_model: Optional[nn.Module] = None,
    distil_alpha: float = 0.5,
    distil_T: float = 4.0,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Complete training pipeline for one model.

    Handles: optimizer, scheduler, early stopping, checkpointing,
    history logging, and final evaluation.

    Parameters
    ----------
    model_name : str
        Identifier for saving checkpoints and results.
    model : nn.Module
        The model to train.
    train_loader, val_loader : DataLoader
        Data loaders from dataset.py.
    stats : NormStats
        Normalization statistics (saved with checkpoint).
    device : torch.device
    max_epochs : int
        Maximum training epochs. Default 50.
    lr : float
        Initial learning rate. Default 1e-3.
    patience : int
        Early stopping patience. Default 10.
    teacher_model : nn.Module, optional
        Frozen teacher for distillation. Default None.
    distil_alpha, distil_T : float
        Distillation hyperparameters.
    save_dir : Path, optional
        Directory for checkpoints and results.
        Default: experiments/

    Returns
    -------
    dict : training history + final metrics
    """
    if save_dir is None:
        save_dir = RESULTS_DIR.parent  # experiments/

    results_dir = save_dir / "results"
    ckpt_dir = save_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Move model to device
    model = model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    # Optimizer: AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler: reduce LR by 0.5 when MR plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, 
    )

    # Early stopping on MR (higher is better)
    early_stop = EarlyStopping(patience=patience)

    # Metrics tracker
    appliance_max = {a: float(APPLIANCES[a]["max_power"]) for a in APPLIANCE_NAMES}
    tracker = MetricsTracker(APPLIANCE_NAMES, appliance_max)

    # Training history
    history = {
        "model": model_name,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_mr": [],
        "val_f1": [],
        "val_mae": [],
        "lr": [],
    }

    best_mr = -float("inf")
    best_epoch = 0
    best_state = None
    start_time = time.time()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs} | LR: {lr} | Patience: {patience}")
    if teacher_model is not None:
        print(f"Distillation: alpha={distil_alpha}, T={distil_T}")
    print(f"{'='*60}\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            teacher_model, distil_alpha, distil_T,
        )

        # ---- Validate ----
        val_loss, metrics = validate_one_epoch(
            model, val_loader, device, tracker,
        )

        # Extract key metrics
        val_mr = metrics["mean"]["mr"]
        val_f1 = metrics["mean"]["f1"]
        val_mae = metrics["mean"]["mae_w"]
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # ---- Log ----
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mr"].append(val_mr)
        history["val_f1"].append(val_f1)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        # ---- Print epoch summary ----
        print(f"Epoch {epoch:3d}/{max_epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"MR: {val_mr:.3f} | F1: {val_f1:.3f} | "
              f"MAE: {val_mae:.1f}W | "
              f"LR: {current_lr:.1e} | "
              f"{epoch_time:.0f}s", end="")

        # ---- Check for improvement ----
        if val_mr > best_mr:
            best_mr = val_mr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(" ★ best", end="")

        print()  # newline

        # ---- Scheduler step (based on MR) ----
        scheduler.step(val_mr)

        # ---- Early stopping ----
        if early_stop.step(val_mr):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time

    # ---- Save best checkpoint ----
    if best_state is not None:
        checkpoint = {
            "model_name": model_name,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_mr": best_mr,
            "n_params": n_params,
            "norm_stats": {
                "agg_mean": stats.agg_mean,
                "agg_std": stats.agg_std,
                "appliance_max": stats.appliance_max,
            },
        }
        ckpt_path = ckpt_dir / f"{model_name}_best.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"\nBest checkpoint saved → {ckpt_path}")
        print(f"  Best epoch: {best_epoch} | Best MR: {best_mr:.4f}")

    # ---- Load best model and compute final metrics ----
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    _, final_metrics = validate_one_epoch(model, val_loader, device, tracker)

    # ---- Print final results table ----
    print(f"\nFinal results ({model_name}, epoch {best_epoch}):")
    tracker.print_table(final_metrics)

    # ---- Save history ----
    history["training_time_seconds"] = total_time
    history["best_epoch"] = best_epoch
    history["best_val_mr"] = best_mr

    history_path = results_dir / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {history_path}")

    # ---- Save final metrics ----
    tracker.save_json(
        results_dir / f"{model_name}_metrics.json",
        final_metrics,
        model_name=model_name,
    )

    # ---- Save NormStats ----
    stats.save(results_dir / f"{model_name}_norm_stats.json")

    return {
        "history": history,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_mr": best_mr,
        "training_time": total_time,
    }


# ============================================================
# 6. DATA LOADING HELPER
# ============================================================

def load_data(
    dataset_name: str = "UK-DALE",
    house: int = 1,
    val_fraction: float = 0.15,
    batch_size: int = BATCH_SIZE,
    train_stride: int = TRAIN_STRIDE,
    val_stride: int = TEST_STRIDE,
    num_workers: int = 4,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, NormStats]:
    """
    Load and preprocess data, build DataLoaders.

    Parameters
    ----------
    dataset_name : str
        "UK-DALE", "REDD", or "AMPds".
    house : int
        House number to load.
    val_fraction : float
        Fraction of data for validation (time-based split).
    batch_size : int
        Batch size for training. Default from config.
    quick : bool
        If True, use only first 50,000 rows for quick testing.

    Returns
    -------
    (train_loader, val_loader, stats)
    """
    from preprocessing import load_ukdale_house, preprocess_house
    from dataset import cache_path, save_clean_df, load_clean_df

    # Try loading cached clean data first
    cached = load_clean_df(dataset_name, house)
    if cached is not None:
        clean_df = cached
    else:
        print(f"\nLoading {dataset_name} House {house}...")
        if dataset_name.upper() in ["UK-DALE", "UKDALE"]:
            raw_df = load_ukdale_house(house=house)
        else:
            raise NotImplementedError(
                f"Dataset '{dataset_name}' loader not yet implemented. "
                f"Available: UK-DALE"
            )
        clean_df = preprocess_house(raw_df)
        save_clean_df(clean_df, dataset_name, house)

    # Quick mode: use only first 50K rows for fast testing
    if quick:
        clean_df = clean_df.iloc[:50_000]
        print(f"Quick mode: using {len(clean_df):,} rows")

    # Time-based train/val split
    train_df, val_df = split_train_val(clean_df, val_fraction)
    print(f"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    # Build DataLoaders
    train_loader, val_loader, stats = build_dataloaders(
        train_df, val_df,
        batch_size=batch_size,
        train_stride=train_stride,
        val_stride=val_stride,
        num_workers=num_workers,
        add_temporal_features=True,
       )   
    return train_loader, val_loader, stats


# ============================================================
# 7. CLI INTERFACE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NILM models — unified training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train BiWave-NILM on UK-DALE
    python src/train.py --model biwave --dataset ukdale --epochs 50

    # Quick test (3 epochs, small data subset)
    python src/train.py --model biwave --quick

    # Train all models sequentially
    python src/train.py --all --dataset ukdale --epochs 50

    # Train with knowledge distillation
    python src/train.py --model biwave --teacher checkpoints/teacher.pth
        """,
    )

    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", type=str,
        choices=["cnn", "gru", "bigru", "lstm", "bilstm",
                 "cnn_lstm", "nilmformer", "biwave"],
        help="Model to train",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Train all models sequentially",
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="ukdale",
        choices=["ukdale", "redd", "ampds"],
        help="Dataset to use (default: ukdale)",
    )
    parser.add_argument(
        "--house", type=int, default=1,
        help="House number (default: 1)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    # Distillation
    parser.add_argument("--teacher", type=str, default=None,
                        help="Path to teacher checkpoint for distillation")
    parser.add_argument("--distil-alpha", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=4.0,
                        help="Distillation temperature")

    # Quick test mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 epochs, 50K rows")

    return parser.parse_args()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    """Entry point — parse args and launch training."""
    args = parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Quick mode overrides ----
    if args.quick:
        args.epochs = 3
        args.num_workers = 0
        print("Quick test mode: 3 epochs, 50K rows, 0 workers")

    # ---- Dataset mapping ----
    dataset_map = {
        "ukdale": "UK-DALE",
        "redd": "REDD",
        "ampds": "AMPds",
    }
    dataset_name = dataset_map[args.dataset]

    # ---- Load data (shared across all models) ----
    train_loader, val_loader, stats = load_data(
        dataset_name=dataset_name,
        house=args.house,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quick=args.quick,
    )

    # ---- Model registry ----
    registry = get_model_registry()

    # ---- Determine which models to train ----
    if args.all:
        model_names = list(registry.keys())
        print(f"\nTraining ALL {len(model_names)} models sequentially")
    else:
        model_names = [args.model]

    # ---- Optional: load teacher for distillation ----
    teacher_model = None
    if args.teacher is not None:
        print(f"\nLoading teacher model from {args.teacher}")
        teacher_ckpt = torch.load(args.teacher, map_location=device)
        # Determine teacher class from checkpoint
        teacher_name = teacher_ckpt.get("model_name", "unknown")
        if teacher_name in registry:
            teacher_model = registry[teacher_name]()
            teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
            print(f"Teacher loaded: {teacher_name}")
        else:
            print(f"WARNING: Unknown teacher model '{teacher_name}'")

    # ---- Train each model ----
    all_results = {}

    for model_name in model_names:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_name}")
        print(f"{'#' * 60}")

        # Instantiate model
        model_class = registry[model_name]
        model = model_class(
            in_channels=INPUT_CHANNELS,
            window_size=WINDOW_SIZE,
            n_appliances=N_APPLIANCES,
        )

        # Train
        result = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            stats=stats,
            device=device,
            max_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            teacher_model=teacher_model,
            distil_alpha=args.distil_alpha,
            distil_T=args.T,
        )

        all_results[model_name] = result

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Summary ----
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Training Summary — All Models")
        print(f"{'='*60}")
        print(f"{'Model':<16} {'Best MR':>8} {'Best F1':>8} "
              f"{'MAE(W)':>8} {'Epoch':>6} {'Time':>8}")
        print("-" * 56)
        for name, res in all_results.items():
            h = res["history"]
            best_idx = h["best_epoch"] - 1
            print(f"{name:<16} "
                  f"{h['val_mr'][best_idx]:>8.3f} "
                  f"{h['val_f1'][best_idx]:>8.3f} "
                  f"{h['val_mae'][best_idx]:>8.1f} "
                  f"{h['best_epoch']:>6d} "
                  f"{h['training_time_seconds']:>7.0f}s")

    print(f"\nAll training complete!")


if __name__ == "__main__":
    main()