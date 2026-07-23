"""
dataset.py — PyTorch Dataset for NILM Seq2Point Learning
=========================================================
Bridges the preprocessing pipeline and the neural network training loop.

Pipeline position:
    preprocessing.py → [clean DataFrame] → dataset.py → [batches] → models.py

Key design decisions (engineering rationale):

1. ON-THE-FLY windowing + DWT (not full offline precomputation).
   Storage math with stride=1 on UK-DALE House 1:
       ~10.17M windows × (4 × 480 float32) ≈ 76 GB  → impossible to store.
   Instead we cache only the CLEAN SERIES (~450 MB as .npz) and compute
   each window's DWT in __getitem__ (~0.3 ms). With DataLoader
   num_workers > 0, DWT runs in parallel on CPU while the GPU trains —
   zero bottleneck for models of this size.

2. GAP-AWARE window generation.
   preprocess_house() drops rows with unfillable gaps (~10% of UK-DALE).
   The remaining index is therefore NOT contiguous. A naive sliding
   window would silently stitch two distant moments into one "window",
   creating physically false training examples. We detect contiguous
   segments first and only generate windows fully inside one segment.

3. NORMALIZATION with train-set statistics only.
   The aggregate is z-scored BEFORE DWT (so all sub-bands inherit a
   consistent scale). Appliance power targets are scaled to [0, 1] by
   their configured max_power (simple, bounded, easily invertible on
   the embedded target). Statistics are computed on TRAIN data and
   passed to val/test datasets — never recomputed on test (no leakage).

4. SEQ2POINT labels.
   Each window of 480 aggregate points predicts power + ON/OFF state
   of every appliance at the CENTER timestep only (Luo et al., 2023).

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    APPLIANCES,
    APPLIANCE_NAMES,
    WINDOW_SIZE,
    TRAIN_STRIDE,
    TEST_STRIDE,
    BATCH_SIZE,
    DATA_PROCESSED_DIR,
    SEED,
)
from dwt import dwt_transform


# ============================================================
# 1. NORMALIZATION STATISTICS (train-only, leakage-safe)
# ============================================================

@dataclass
class NormStats:
    """
    Container for normalization statistics.

    Computed ONCE on the training split, then reused unchanged for
    validation and test — this prevents data leakage (test statistics
    must never influence the model or its inputs).

    Attributes
    ----------
    agg_mean, agg_std : float
        Mean / std of the aggregate power on the TRAIN split.
        Used to z-score the aggregate window BEFORE DWT.
    appliance_max : dict
        {appliance: max_power} from config — scales power targets
        to [0, 1]. Using the configured physical maximum (not the
        empirical max) keeps the scale identical across datasets,
        which matters for cross-dataset generalization experiments.
    """
    agg_mean: float
    agg_std: float
    appliance_max: Dict[str, float] = field(default_factory=dict)

    # ---- persistence (reproducibility + embedded deployment) ----
    def save(self, path: Path) -> None:
        """Save stats as JSON — the SAME values must be flashed to the
        STM32MP2 firmware so embedded inference normalizes identically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_train_series(cls, aggregate: np.ndarray) -> "NormStats":
        """Compute statistics from the TRAIN aggregate series only."""
        return cls(
            agg_mean=float(np.mean(aggregate)),
            agg_std=float(np.std(aggregate) + 1e-8),  # epsilon: never /0
            appliance_max={a: float(APPLIANCES[a]["max_power"])
                           for a in APPLIANCE_NAMES},
        )


# ============================================================
# 2. GAP-AWARE SEGMENT DETECTION
# ============================================================

def find_contiguous_segments(
    index: pd.DatetimeIndex,
    sampling_seconds: int = 6,
) -> List[Tuple[int, int]]:
    """
    Split a (possibly gapped) DatetimeIndex into contiguous segments.

    After preprocess_house() drops unfillable gaps, consecutive rows
    are no longer guaranteed to be `sampling_seconds` apart. A window
    must never span a discontinuity — it would merge e.g. Tuesday 23:59
    with Friday 08:00 into one "48-minute" window.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamp index of the clean DataFrame.
    sampling_seconds : int
        Expected spacing between consecutive rows.

    Returns
    -------
    List of (start_idx, end_idx) integer positions, end exclusive.
        Each segment satisfies: diff(index) == sampling_seconds inside.

    Example
    -------
    >>> # rows 0..99 contiguous, gap, rows 100..250 contiguous
    >>> segments
    [(0, 100), (100, 251)]
    """
    if len(index) == 0:
        return []

    diffs = index.to_series().diff().dt.total_seconds().to_numpy()
    # First diff is NaN; a "break" is any step != expected sampling
    breaks = np.where(diffs != sampling_seconds)[0]  # includes idx 0 (NaN)

    starts = np.concatenate(([0], breaks[breaks > 0]))
    ends = np.concatenate((breaks[breaks > 0], [len(index)]))

    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def build_window_starts(
    segments: List[Tuple[int, int]],
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Generate all valid window start positions, restricted to windows
    that fit ENTIRELY inside one contiguous segment.

    Returns
    -------
    np.ndarray of int32 start indices into the clean arrays.
    """
    starts: List[np.ndarray] = []
    for seg_start, seg_end in segments:
        last_valid_start = seg_end - window_size  # inclusive
        if last_valid_start >= seg_start:
            starts.append(
                np.arange(seg_start, last_valid_start + 1, stride, dtype=np.int32)
            )
    if not starts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(starts)


# ============================================================
# 3. THE PYTORCH DATASET
# ============================================================

class NILMDataset(Dataset):
    """
    Seq2point NILM dataset: aggregate window → (DWT features, targets).

    Each item:
        x            : float32 tensor (4, 480)   — DWT sub-bands of the
                                                   z-scored aggregate window
        y_power      : float32 tensor (N_app,)   — power at center, in [0,1]
        y_state      : float32 tensor (N_app,)   — 0/1 state at center

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocess_house(): columns
        ['aggregate', *APPLIANCE_NAMES, *'{a}_state'] with DatetimeIndex.
    stats : NormStats
        Normalization statistics (compute on TRAIN split only).
    window_size : int
        Timesteps per window (default 480 = 48 min at 6 s).
    stride : int
        Step between consecutive window starts (1 for training).
    sampling_seconds : int
        Expected index spacing — used for gap detection.
    apply_dwt : bool
        If False, x is the raw normalized window with shape (1, 480).
        Used for the "no-DWT" ablation variant (Day 19 of the plan).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stats: NormStats,
        window_size: int = WINDOW_SIZE,
        stride: int = TRAIN_STRIDE,
        sampling_seconds: int = 6,
        apply_dwt: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.stats = stats
        self.apply_dwt = apply_dwt

        # ---- extract numpy arrays once (fast __getitem__ later) ----
        self.aggregate = df["aggregate"].to_numpy(dtype=np.float32)

        self.power = np.stack(
            [df[a].to_numpy(dtype=np.float32) for a in APPLIANCE_NAMES],
            axis=1,
        )  # shape (T, N_app)

        self.state = np.stack(
            [df[f"{a}_state"].to_numpy(dtype=np.float32) for a in APPLIANCE_NAMES],
            axis=1,
        )  # shape (T, N_app)

        # ---- per-appliance max for target scaling to [0, 1] ----
        self.app_max = np.array(
            [stats.appliance_max[a] for a in APPLIANCE_NAMES],
            dtype=np.float32,
        )  # shape (N_app,)

        # ---- gap-aware valid window starts ----
        segments = find_contiguous_segments(df.index, sampling_seconds)
        self.window_starts = build_window_starts(segments, window_size, stride)

        n_naive = max(0, (len(df) - window_size) // stride + 1)
        n_valid = len(self.window_starts)
        print(f"NILMDataset: {n_valid:,} valid windows "
              f"({len(segments)} contiguous segments; "
              f"{n_naive - n_valid:,} gap-crossing windows excluded)")

    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.window_starts)

    # --------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[idx])
        end = start + self.window_size
        center = start + self.window_size // 2

        # ---- input: z-score aggregate window, then DWT ----
        window = self.aggregate[start:end]
        window = (window - self.stats.agg_mean) / self.stats.agg_std

        if self.apply_dwt:
            x = dwt_transform(window)              # (4, 480) float32
        else:
            x = window[np.newaxis, :].astype(np.float32)  # (1, 480)

        # ---- targets at the CENTER point (seq2point) ----
        y_power = self.power[center] / self.app_max     # scaled to [0,1]
        y_state = self.state[center]                    # already 0/1

        return (
            torch.from_numpy(x),
            torch.from_numpy(y_power.astype(np.float32)),
            torch.from_numpy(y_state.astype(np.float32)),
        )

    # --------------------------------------------------------
    def denormalize_power(self, y_power: torch.Tensor) -> torch.Tensor:
        """
        Convert model outputs in [0, 1] back to Watts.
        Used by metrics.py so that MAE is reported in physical units.

        y_power : tensor (..., N_app) — normalized predictions.
        """
        scale = torch.as_tensor(self.app_max, device=y_power.device)
        return y_power * scale


# ============================================================
# 4. TIME-BASED TRAIN/VAL SPLIT (no shuffling across time)
# ============================================================

def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one house's clean DataFrame into train / validation by TIME:
    the last `val_fraction` of the recording becomes validation.

    Why time-based (never random-row) splitting:
        Windows overlap heavily at stride=1 — two windows one step apart
        share 479 of 480 points. A random split would put near-identical
        windows in both train and val, making validation loss a lie.
        A temporal split guarantees val windows are genuinely unseen.
    """
    split_idx = int(len(df) * (1.0 - val_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ============================================================
# 5. CLEAN-SERIES CACHE (skip pandas preprocessing on re-runs)
# ============================================================

def cache_path(dataset_name: str, house: int) -> Path:
    return DATA_PROCESSED_DIR / f"{dataset_name.lower()}_house{house}.parquet"


def save_clean_df(df: pd.DataFrame, dataset_name: str, house: int) -> Path:
    """Cache the preprocessed DataFrame (~450 MB → parquet ~150 MB)."""
    path = cache_path(dataset_name, house)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Cached clean data → {path}")
    return path


def load_clean_df(dataset_name: str, house: int) -> Optional[pd.DataFrame]:
    """Load cached clean DataFrame if it exists, else None."""
    path = cache_path(dataset_name, house)
    if path.exists():
        print(f"Loading cached clean data ← {path}")
        return pd.read_parquet(path)
    return None


# ============================================================
# 6. ONE-CALL DATALOADER FACTORY
# ============================================================

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    train_stride: int = TRAIN_STRIDE,
    val_stride: int = TEST_STRIDE,
    num_workers: int = 4,
    apply_dwt: bool = True,
) -> Tuple[DataLoader, DataLoader, NormStats]:
    """
    Build train + validation DataLoaders with leakage-safe normalization.

    Returns (train_loader, val_loader, stats). `stats` must be saved
    with the experiment: evaluation and embedded deployment need the
    exact same normalization constants.
    """
    # Statistics from TRAIN aggregate only — never from validation.
    stats = NormStats.from_train_series(train_df["aggregate"].to_numpy())

    train_ds = NILMDataset(train_df, stats, stride=train_stride,
                           apply_dwt=apply_dwt)
    val_ds = NILMDataset(val_df, stats, stride=val_stride,
                         apply_dwt=apply_dwt)

    g = torch.Generator()
    g.manual_seed(SEED)  # reproducible shuffling

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # windows are i.i.d. samples for seq2point
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,        # stable batch statistics during training
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # deterministic evaluation order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, stats


# ============================================================
# ENTRY POINT — Self-test on synthetic data (no big files needed)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Module — Self-Test (synthetic data)")
    print("=" * 60)

    # ---- build a small synthetic clean DataFrame with one gap ----
    rng = np.random.default_rng(SEED)
    n1, gap, n2 = 3000, 500, 2000          # two segments, one hole
    t1 = pd.date_range("2024-01-01", periods=n1, freq="6s")
    t2 = pd.date_range(t1[-1] + pd.Timedelta(seconds=6 * (gap + 1)),
                       periods=n2, freq="6s")
    index = t1.append(t2)

    agg = rng.normal(300, 50, n1 + n2).clip(0)
    data = {"aggregate": agg}
    for a in APPLIANCE_NAMES:
        p = rng.random(n1 + n2) * 10
        on = rng.random(n1 + n2) < 0.05
        p[on] += APPLIANCES[a]["power_threshold"] * 1.5
        data[a] = p.clip(0, APPLIANCES[a]["max_power"])
        data[f"{a}_state"] = (data[a] > APPLIANCES[a]["power_threshold"]).astype(int)

    df = pd.DataFrame(data, index=index)
    print(f"\n[1] Synthetic clean df: {df.shape}, one gap of {gap} steps")

    # ---- split, stats, datasets ----
    train_df, val_df = split_train_val(df, val_fraction=0.2)
    print(f"[2] Split: train={len(train_df)}, val={len(val_df)} (time-based)")

    train_loader, val_loader, stats = build_dataloaders(
        train_df, val_df, batch_size=16, num_workers=0
    )
    print(f"[3] NormStats: mean={stats.agg_mean:.1f}, std={stats.agg_std:.1f}")

    # ---- one batch through ----
    x, y_power, y_state = next(iter(train_loader))
    print(f"[4] Batch shapes: x={tuple(x.shape)}, "
          f"y_power={tuple(y_power.shape)}, y_state={tuple(y_state.shape)}")
    assert x.shape[1:] == (4, WINDOW_SIZE)
    assert y_power.shape[1] == len(APPLIANCE_NAMES)
    assert float(y_power.max()) <= 1.0 + 1e-6, "power targets must be in [0,1]"

    # ---- gap-awareness check ----
    ds = train_loader.dataset
    seg_ok = all(
        (s + WINDOW_SIZE <= n1) or (s >= n1)
        for s in ds.window_starts
    )
    print(f"[5] Gap-crossing windows excluded: {seg_ok}")
    assert seg_ok

    # ---- no-DWT ablation variant ----
    ds_raw = NILMDataset(train_df, stats, stride=10, apply_dwt=False)
    x_raw, _, _ = ds_raw[0]
    print(f"[6] Ablation (apply_dwt=False) input shape: {tuple(x_raw.shape)}")
    assert x_raw.shape == (1, WINDOW_SIZE)

    # ---- stats persistence ----
    p = DATA_PROCESSED_DIR / "_selftest_stats.json"
    stats.save(p)
    stats2 = NormStats.load(p)
    assert stats2.agg_mean == stats.agg_mean
    p.unlink()
    print(f"[7] NormStats save/load round-trip OK")

    print(f"\n{'=' * 60}")
    print("All tests passed — dataset module ready!")
    print(f"{'=' * 60}")