"""
evaluate.py — Evaluation Pipeline for NILM Models
====================================================
Handles three evaluation scenarios:

1. WITHIN-DATASET evaluation
   Train on houses 1,3,4,5 → Test on house 2 (UK-DALE protocol)
   Matches NILMFormer (Petralia et al., KDD 2025) experimental setup.

2. CROSS-DATASET evaluation
   Train on UK-DALE → Test on REDD (and vice versa)
   Measures domain generalization without retraining.

3. PSEUDO-LABEL fine-tuning (optional)
   After cross-dataset evaluation, generate high-confidence pseudo-labels
   on target domain and fine-tune for 5 epochs.
   Inspired by Hur et al. (Sensors 2022).

Usage:
    # Within-dataset evaluation
    python src/evaluate.py \
        --checkpoint experiments/checkpoints/biwave_best.pth \
        --model biwave \
        --dataset ukdale \
        --house 2

    # Cross-dataset evaluation
    python src/evaluate.py \
        --checkpoint experiments/checkpoints/biwave_best.pth \
        --model biwave \
        --source ukdale \
        --target redd \
        --cross-dataset

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
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
    TEST_STRIDE,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
)
from dataset import (
    NILMDataset,
    NormStats,
    build_dataloaders,
    load_clean_df,
    save_clean_df,
)
from metrics import MetricsTracker


# ============================================================
# 1. LOAD MODEL FROM CHECKPOINT
# ============================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, NormStats]:
    """
    Load a trained model and its NormStats from a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the .pth checkpoint file saved by train.py.
    model_name : str
        Model identifier for the registry.
    device : torch.device

    Returns
    -------
    (model, stats) — model loaded with best weights, NormStats for
    denormalization and input normalization.
    """
    from train import get_model_registry

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Instantiate model
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {list(registry.keys())}")

    model = registry[model_name](
        in_channels=INPUT_CHANNELS,
        window_size=WINDOW_SIZE,
        n_appliances=N_APPLIANCES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Reconstruct NormStats from checkpoint
    ns = checkpoint["norm_stats"]
    stats = NormStats(
        agg_mean=ns["agg_mean"],
        agg_std=ns["agg_std"],
        appliance_max=ns["appliance_max"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {model_name} | "
          f"Params: {n_params:,} | "
          f"Best epoch: {checkpoint.get('best_epoch', '?')} | "
          f"Best MR: {checkpoint.get('best_val_mr', 0):.4f}")

    return model, stats


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

def load_test_data(
    dataset_name: str,
    house: int,
    stats: NormStats,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    stride: int = TEST_STRIDE,
) -> DataLoader:
    """
    Load test data for a specific house.

    Uses EVAL_STRIDE (non-overlapping windows) by default for
    fair comparison with NILMFormer evaluation protocol.

    Parameters
    ----------
    dataset_name : str
        "UK-DALE", "REDD", or "AMPds".
    house : int
        House number (test house).
    stats : NormStats
        Normalization statistics from training — NOT recomputed
        on test data to prevent leakage.
    stride : int
        Window stride. Use WINDOW_SIZE for non-overlapping (NILMFormer
        protocol) or 1 for dense evaluation.

    Returns
    -------
    DataLoader for the test house.
    """
    from preprocessing import load_ukdale_house, preprocess_house

    # Try cache first
    cached = load_clean_df(dataset_name, house)
    if cached is not None:
        clean_df = cached
    else:
        print(f"Loading {dataset_name} House {house}...")
        if dataset_name.upper() in ["UK-DALE", "UKDALE"]:
            raw_df = load_ukdale_house(house=house)
        elif dataset_name.upper() == "REDD":
            from preprocessing import load_redd_house
            raw_df = load_redd_house(house=house)
        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' not supported yet.")

        clean_df = preprocess_house(raw_df)
        save_clean_df(clean_df, dataset_name, house)

    # Build test dataset using TRAIN stats (no leakage)
    test_ds = NILMDataset(
        clean_df, stats,
        stride=stride,
        add_temporal_features=True,
    )

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# 3. EVALUATE ONE MODEL ON ONE TEST SET
# ============================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str = "model",
    dataset_name: str = "unknown",
    house: int = 0,
    save_path: Optional[Path] = None,
) -> Dict:
    """
    Run full evaluation on a test set.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    test_loader : DataLoader
        Test data loader.
    device : torch.device
    model_name : str
        For labeling results.
    dataset_name : str
        For labeling results.
    house : int
        Test house number.
    save_path : Path, optional
        Where to save JSON results.

    Returns
    -------
    dict : full metrics per appliance + mean.
    """
    model.eval()

    appliance_max = {a: float(APPLIANCES[a]["max_power"]) for a in APPLIANCE_NAMES}
    tracker = MetricsTracker(APPLIANCE_NAMES, appliance_max)

    for x, y_power, y_state in tqdm(
        test_loader,
        desc=f"  Evaluating {model_name} on {dataset_name} house {house}"
    ):
        x = x.to(device)
        pred_power, pred_state, pred_gated = model(x)
        tracker.update(pred_power, y_power, pred_state, y_state)

    results = tracker.compute()

    # Print formatted table
    print(f"\nResults: {model_name} on {dataset_name} House {house}")
    tracker.print_table(results)

    # Save to JSON
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": model_name,
            "dataset": dataset_name,
            "house": house,
            "evaluation_type": "within_dataset",
            "results": results,
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved → {save_path}")

    return results


# ============================================================
# 4. CROSS-DATASET EVALUATION
# ============================================================

def evaluate_cross_dataset(
    model: torch.nn.Module,
    stats: NormStats,
    source_dataset: str,
    target_dataset: str,
    target_house: int,
    device: torch.device,
    model_name: str = "model",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Cross-dataset evaluation — test on unseen domain without retraining.

    The model was trained on source_dataset. We evaluate it directly
    on target_dataset without any fine-tuning. This measures domain
    generalization — a key contribution of your thesis.

    Following the 6 transfer directions:
        UK-DALE → REDD
        REDD → UK-DALE
        UK-DALE → AMPds
        AMPds → UK-DALE
        REDD → AMPds
        AMPds → REDD

    Parameters
    ----------
    model : nn.Module
        Model trained on source_dataset.
    stats : NormStats
        Normalization stats from source_dataset training.
    source_dataset : str
        Dataset the model was trained on.
    target_dataset : str
        Dataset to evaluate on (unseen domain).
    target_house : int
        House number in target dataset.
    """
    print(f"\n{'='*60}")
    print(f"Cross-dataset: {source_dataset} → {target_dataset}")
    print(f"{'='*60}")

    # Load target test data using SOURCE stats (no recomputation)
    test_loader = load_test_data(
        target_dataset, target_house, stats,
        batch_size=batch_size, num_workers=num_workers,
    )

    # Evaluate
    save_path = None
    if save_dir is not None:
        fname = f"{model_name}_{source_dataset}_to_{target_dataset}_h{target_house}.json"
        save_path = Path(save_dir) / fname

    results = evaluate(
        model, test_loader, device,
        model_name=model_name,
        dataset_name=f"{source_dataset}→{target_dataset}",
        house=target_house,
        save_path=save_path,
    )

    return results


# ============================================================
# 5. PSEUDO-LABEL FINE-TUNING (optional Stage 3)
# ============================================================

def pseudo_label_finetune(
    model: torch.nn.Module,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    confidence_threshold: float = 0.90,
    n_epochs: int = 5,
) -> torch.nn.Module:
    """
    Fine-tune model on high-confidence pseudo-labels from target domain.

    Inspired by Hur et al. "Semi-Supervised Domain Adaptation for
    Multi-Label Classification on NILM." Sensors 2022.

    Stage 3 of their framework:
        1. Generate pseudo-labels for target domain using current model
        2. Keep only high-confidence predictions (> threshold)
        3. Fine-tune model on pseudo-labeled target data

    Parameters
    ----------
    model : nn.Module
        Model after cross-dataset evaluation.
    target_loader : DataLoader
        Unlabeled target domain data.
    optimizer : torch.optim.Optimizer
    device : torch.device
    confidence_threshold : float
        Minimum sigmoid probability to accept as pseudo-label.
        Default 0.90 — only very confident predictions.
    n_epochs : int
        Number of fine-tuning epochs. Default 5.

    Returns
    -------
    Fine-tuned model.
    """
    from metrics import multi_task_loss

    print(f"\nPseudo-label fine-tuning ({n_epochs} epochs, "
          f"threshold={confidence_threshold})")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_pseudo = 0

        for x, y_power, y_state in target_loader:
            x = x.to(device)

            # Generate pseudo-labels with current model
            with torch.no_grad():
                pred_power, pred_state, pred_gated = model(x)
                pred_probs = torch.sigmoid(pred_state)

            # Keep only high-confidence pseudo-labels
            # A sample is confident if ALL appliances exceed threshold
            # OR if ANY appliance clearly OFF (prob < 1-threshold)
            confident_mask = (
                (pred_probs > confidence_threshold) |
                (pred_probs < (1 - confidence_threshold))
            ).all(dim=1)  # (B,) — True if all appliances are confident

            if confident_mask.sum() == 0:
                continue  # skip batch if no confident predictions

            # Use pseudo-labels for confident samples
            x_conf = x[confident_mask]
            pseudo_power = pred_power[confident_mask].detach()
            pseudo_state = (pred_probs[confident_mask] > 0.5).float()

            # Forward + loss on pseudo-labeled data
            pred_p, pred_s, pred_g = model(x_conf)
            loss = multi_task_loss(pred_p, pseudo_power, pred_s, pseudo_state, pred_g)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            n_pseudo += confident_mask.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Pseudo-labeled samples: {n_pseudo:,}")

    return model


# ============================================================
# 6. FULL EVALUATION PIPELINE — ALL MODELS
# ============================================================

def evaluate_all_models(
    dataset_name: str = "UK-DALE",
    test_house: int = 2,
    checkpoints_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    num_workers: int = 4,
) -> Dict:
    """
    Evaluate ALL trained models on the same test set.

    Loads each checkpoint, runs evaluation, saves results.
    Builds the comparison table for your thesis.

    Parameters
    ----------
    dataset_name : str
        Test dataset. Default "UK-DALE".
    test_house : int
        Test house. Default 2 (UK-DALE house 2 = NILMFormer protocol).
    checkpoints_dir : Path, optional
        Directory containing model checkpoints.
    results_dir : Path, optional
        Directory to save evaluation results.

    Returns
    -------
    dict : {model_name: metrics_dict} for all evaluated models.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoints_dir is None:
        checkpoints_dir = CHECKPOINTS_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_names = [
        "cnn", "gru", "bigru", "lstm",
        "bilstm", "cnn_lstm", "biwave",
    ]

    all_results = {}

    for model_name in model_names:
        ckpt_path = Path(checkpoints_dir) / f"{model_name}_best.pth"

        if not ckpt_path.exists():
            print(f"\nSkipping {model_name} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # Load model + stats
        model, stats = load_model_from_checkpoint(ckpt_path, model_name, device)

        # Load test data
        test_loader = load_test_data(
            dataset_name, test_house, stats,
            num_workers=num_workers,
        )

        # Evaluate
        save_path = Path(results_dir) / f"{model_name}_test_metrics.json"
        results = evaluate(
            model, test_loader, device,
            model_name=model_name,
            dataset_name=dataset_name,
            house=test_house,
            save_path=save_path,
        )

        all_results[model_name] = results

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save combined results
    combined_path = Path(results_dir) / "all_models_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved → {combined_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY — All Models")
    print(f"{'='*60}")
    print(f"{'Model':<16} {'MAE(W)':>8} {'MR':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 48)
    for name, res in all_results.items():
        m = res["mean"]
        print(f"{name:<16} {m['mae_w']:>8.1f} {m['mr']:>8.3f} "
              f"{m['f1']:>8.3f} {m['accuracy']:>8.3f}")

    return all_results


# ============================================================
# 7. CLI INTERFACE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained NILM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Within-dataset evaluation
    python src/evaluate.py \\
        --checkpoint experiments/checkpoints/biwave_best.pth \\
        --model biwave --dataset ukdale --house 2

    # Cross-dataset evaluation
    python src/evaluate.py \\
        --checkpoint experiments/checkpoints/biwave_best.pth \\
        --model biwave --source ukdale --target redd --cross-dataset

    # Evaluate all trained models
    python src/evaluate.py --all --dataset ukdale --house 2
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint")
    group.add_argument("--all", action="store_true",
                       help="Evaluate all trained models")

    parser.add_argument("--model", type=str,
                        choices=["cnn", "gru", "bigru", "lstm",
                                 "bilstm", "cnn_lstm", "nilmformer", "biwave"],
                        help="Model name (required with --checkpoint)")
    parser.add_argument("--dataset", type=str, default="ukdale",
                        choices=["ukdale", "redd", "ampds"])
    parser.add_argument("--house", type=int, default=2,
                        help="Test house number (default: 2)")

    # Cross-dataset options
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Run cross-dataset evaluation")
    parser.add_argument("--source", type=str, default="ukdale",
                        help="Source domain (model was trained on this)")
    parser.add_argument("--target", type=str, default="redd",
                        help="Target domain (unseen during training)")

    # Pseudo-label fine-tuning
    parser.add_argument("--pseudo-label", action="store_true",
                        help="Apply pseudo-label fine-tuning after cross-dataset eval")
    parser.add_argument("--pl-threshold", type=float, default=0.90,
                        help="Confidence threshold for pseudo-labels (default 0.90)")
    parser.add_argument("--pl-epochs", type=int, default=5,
                        help="Fine-tuning epochs for pseudo-labeling (default 5)")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_map = {
        "ukdale": "UK-DALE",
        "redd": "REDD",
        "ampds": "AMPds",
    }

    # ---- Evaluate ALL models ----
    if args.all:
        dataset_name = dataset_map[args.dataset]
        evaluate_all_models(
            dataset_name=dataset_name,
            test_house=args.house,
            device=device,
            num_workers=args.num_workers,
        )
        return

    # ---- Single model evaluation ----
    if args.model is None:
        print("ERROR: --model is required with --checkpoint")
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    model, stats = load_model_from_checkpoint(
        checkpoint_path, args.model, device
    )

    # ---- Cross-dataset ----
    if args.cross_dataset:
        source = dataset_map[args.source]
        target = dataset_map[args.target]

        results = evaluate_cross_dataset(
            model, stats,
            source_dataset=source,
            target_dataset=target,
            target_house=args.house,
            device=device,
            model_name=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_dir=RESULTS_DIR,
        )

        # Optional pseudo-label fine-tuning
        if args.pseudo_label:
            target_loader = load_test_data(
                target, args.house, stats,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-4
            )
            model = pseudo_label_finetune(
                model, target_loader, optimizer, device,
                confidence_threshold=args.pl_threshold,
                n_epochs=args.pl_epochs,
            )

            # Re-evaluate after fine-tuning
            print("\nRe-evaluating after pseudo-label fine-tuning...")
            evaluate_cross_dataset(
                model, stats,
                source_dataset=source,
                target_dataset=target,
                target_house=args.house,
                device=device,
                model_name=f"{args.model}_finetuned",
                save_dir=RESULTS_DIR,
            )

    # ---- Within-dataset ----
    else:
        dataset_name = dataset_map[args.dataset]
        test_loader = load_test_data(
            dataset_name, args.house, stats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        save_path = RESULTS_DIR / f"{args.model}_test_metrics.json"
        evaluate(
            model, test_loader, device,
            model_name=args.model,
            dataset_name=dataset_name,
            house=args.house,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
