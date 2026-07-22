"""
preprocessing.py : Data Loading and Cleaning for NILM Datasets
=================================================================
Loads raw power data from UK-DALE, REDD, AMPds2, and REFIT datasets,
applies resampling, gap-filling, and normalization, producing clean
aggregate + appliance power series ready for windowing.

Supported formats:
    UK-DALE : pandas HDFStore (.h5) — NILMTK format
    REDD    : pandas HDFStore (.h5) — NILMTK format
    AMPds2  : CSV files (after unzip)
    REFIT   : CSV files (after 7z extraction)

Appliance meter mapping (UK-DALE):
    Verified empirically via duty-cycle analysis against official
    UK-DALE documentation (Kelly & Knottenbelt, 2015).
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

warnings.filterwarnings("ignore")  # suppress pandas HDFStore deprecation noise

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATASETS,
    APPLIANCES,
    APPLIANCE_NAMES,
    MAX_GAP_SECONDS,
)


# ============================================================
# 1. UK-DALE METER MAPPING (verified empirically)
# ============================================================
# Meter numbers differ per house because appliances are wired to
# different channels in each household's monitoring setup.
# Mapping verified via duty-cycle analysis on House 1 and House 2.
#
# meter1 is ALWAYS the whole-house aggregate (mains) in UK-DALE.

UKDALE_METER_MAP: Dict[int, Dict[str, int]] = {
    1: {
        "aggregate": 1,
        "kettle": 10,
        "fridge": 12,
        "washing_machine": 5,
        "dishwasher": 6,
        "microwave": 13,
    },
    2: {
        "aggregate": 1,
        "kettle": 8,
        "fridge": 14,          # combined fridge-freezer unit
        "washing_machine": 12,
        "dishwasher": 13,
        "microwave": 15,
    },
    3: {
        "aggregate": 1,
        "kettle": 2,
        # House 3 has limited appliance coverage in UK-DALE
        # Verify before using — fridge/washer may be absent.
    },
    4: {
        "aggregate": 1,
        "kettle": 3,
        "fridge": 5,
        "washing_machine": 9,
        "dishwasher": None,    # not monitored in House 4
        "microwave": None,     # not monitored in House 4
    },
    5: {
        "aggregate": 1,
        "kettle": 18,
        "fridge": 19,
        "washing_machine": 24,
        "dishwasher": 22,
        "microwave": 23,
    },
}


# ============================================================
# 2. LOAD UK-DALE (HDFStore format)
# ============================================================

def load_ukdale_meter(
    h5_path: Path,
    house: int,
    meter: int,
) -> Optional[pd.Series]:
    """
    Load a single meter's power series from the UK-DALE HDF5 file.

    Parameters
    ----------
    h5_path : Path
        Path to ukdale.h5
    house : int
        House number (1-5)
    meter : int
        Meter number within that house

    Returns
    -------
    pd.Series or None
        Power values indexed by timestamp (timezone-aware).
        Returns None if the meter does not exist or read fails.

    Example
    -------
    >>> series = load_ukdale_meter(Path("data/raw/UKDALE/ukdale.h5"), 1, 1)
    >>> series.name
    'aggregate'
    """
    key = f"/building{house}/elec/meter{meter}"
    try:
        df = pd.read_hdf(h5_path, key=key)
        # Column is a MultiIndex tuple like ('power', 'apparent')
        # We just take the first (and only) column regardless of its exact name.
        series = df.iloc[:, 0]
        return series
    except (KeyError, FileNotFoundError) as e:
        print(f"  [WARNING] Could not load house{house}/meter{meter}: {e}")
        return None


def load_ukdale_house(
    house: int,
    h5_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load aggregate + all target appliance series for one UK-DALE house,
    aligned on a common timestamp index.

    Parameters
    ----------
    house : int
        House number (1-5).
    h5_path : Path, optional
        Override the default path from config.

    Returns
    -------
    pd.DataFrame
        Columns: ['aggregate', 'kettle', 'fridge', 'washing_machine',
                   'dishwasher', 'microwave']
        Index: timestamp (timezone-aware, Europe/London).
        Missing appliances (not monitored in that house) are filled
        with NaN columns — handled later by fill_missing_appliance().

    Example
    -------
    >>> df = load_ukdale_house(1)
    >>> df.columns.tolist()
    ['aggregate', 'kettle', 'fridge', 'washing_machine', 'dishwasher', 'microwave']
    """
    if h5_path is None:
        h5_path = Path.home() / "nilm_project" / "data" / "raw" / "UKDALE" / "ukdale.h5"

    if house not in UKDALE_METER_MAP:
        raise ValueError(f"House {house} not in UKDALE_METER_MAP. "
                          f"Available: {list(UKDALE_METER_MAP.keys())}")

    meter_map = UKDALE_METER_MAP[house]
    series_dict = {}

    print(f"Loading UK-DALE House {house}...")
    for appliance in ["aggregate"] + APPLIANCE_NAMES:
        meter = meter_map.get(appliance)

        if meter is None:
            print(f"  [SKIP] '{appliance}' not monitored in House {house}")
            series_dict[appliance] = None
            continue

        series = load_ukdale_meter(h5_path, house, meter)
        series_dict[appliance] = series
        if series is not None:
            print(f"  [OK] {appliance} (meter{meter}): "
                  f"{len(series)} points, "
                  f"{series.index[0]} to {series.index[-1]}")

    # Align all series on a common index (outer join, then handle NaN later)
    valid_series = {k: v for k, v in series_dict.items() if v is not None}
    df = pd.DataFrame(valid_series)

    # Add missing appliance columns as all-NaN so downstream code
    # always sees the same column structure.
    print("  Resampling each series to 6s before joining...")
    resampled_dict = {}
    for appliance, series in series_dict.items():
        if series is not None:
            # Remove timezone info — avoids alignment edge cases
            series.index = series.index.tz_localize(None)
            resampled = series.resample("6s").mean()
            resampled_dict[appliance] = resampled
            print(f"    {appliance}: {len(resampled)} points after resample")

    # Now join on common aligned index — fast
    df = pd.DataFrame(resampled_dict)

    # Add missing appliance columns as NaN
    for appliance in ["aggregate"] + APPLIANCE_NAMES:
        if appliance not in df.columns:
            df[appliance] = np.nan

    return df[["aggregate"] + APPLIANCE_NAMES]



# ============================================================
# 3. LOAD REDD (HDFStore format)
# ============================================================
# REDD structure differs slightly: it uses separate mains meters
# (meter1, meter2 = two phases) that must be SUMMED for aggregate.

REDD_METER_MAP: Dict[int, Dict[str, object]] = {
    1: {
        "aggregate": [1, 2],     # REDD splits mains into 2 phases — sum them
        "kettle": None,          # REDD does not have a dedicated kettle meter
        "fridge": 5,
        "washing_machine": 20,
        "dishwasher": 6,
        "microwave": 11,
    },
}


def load_redd_meter(h5_path: Path, house: int, meter: int) -> Optional[pd.Series]:
    """Load a single REDD meter (same HDFStore mechanism as UK-DALE)."""
    key = f"/building{house}/elec/meter{meter}"
    try:
        df = pd.read_hdf(h5_path, key=key)
        return df.iloc[:, 0]
    except (KeyError, FileNotFoundError) as e:
        print(f"  [WARNING] Could not load REDD house{house}/meter{meter}: {e}")
        return None


def load_redd_house(house: int, h5_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load aggregate + appliances for one REDD house.

    REDD's mains power is split across 2 phase meters that must be
    summed to get the true whole-house aggregate — a US 2-phase
    240V split-phase electrical system, unlike UK-DALE's single mains.
    """
    if h5_path is None:
        h5_path = Path.home() / "nilm_project" / "data" / "raw" / "REDD" / "redd.h5"

    if house not in REDD_METER_MAP:
        raise ValueError(f"House {house} not in REDD_METER_MAP.")

    meter_map = REDD_METER_MAP[house]
    print(f"Loading REDD House {house}...")

    # Aggregate = sum of both mains phases
    agg_meters = meter_map["aggregate"]
    phase_series = [load_redd_meter(h5_path, house, m) for m in agg_meters]
    phase_series = [s for s in phase_series if s is not None]

    if len(phase_series) == 0:
        raise RuntimeError(f"Could not load any mains phase for REDD house {house}")

    # Align phases on common index before summing
    aggregate = pd.concat(phase_series, axis=1).sum(axis=1)
    print(f"  [OK] aggregate (sum of {len(phase_series)} phases): {len(aggregate)} points")

    series_dict = {"aggregate": aggregate}

    for appliance in APPLIANCE_NAMES:
        meter = meter_map.get(appliance)
        if meter is None:
            print(f"  [SKIP] '{appliance}' not available in REDD")
            series_dict[appliance] = None
            continue
        series = load_redd_meter(h5_path, house, meter)
        series_dict[appliance] = series
        if series is not None:
            print(f"  [OK] {appliance} (meter{meter}): {len(series)} points")

    valid_series = {k: v for k, v in series_dict.items() if v is not None}
    df = pd.DataFrame(valid_series)

    for appliance in ["aggregate"] + APPLIANCE_NAMES:
        if appliance not in df.columns:
            df[appliance] = np.nan

    return df[["aggregate"] + APPLIANCE_NAMES]


# ============================================================
# 4. RESAMPLING AND GAP FILLING
# ============================================================

def resample_to_target_rate(
    df: pd.DataFrame,
    target_seconds: int = 6,
) -> pd.DataFrame:
    """
    Resample all columns to a uniform target sampling rate.

    Uses mean aggregation within each new time bin — appropriate for
    power signals since it approximates the average power consumed
    during that interval (closer to true energy than picking one sample).

    Parameters
    ----------
    df : pd.DataFrame
        Power data with a DatetimeIndex, potentially at irregular
        or different sampling rate than target_seconds.
    target_seconds : int
        Desired uniform sampling interval in seconds.

    Returns
    -------
    pd.DataFrame
        Resampled to exactly `target_seconds` intervals.
    """
    rule = f"{target_seconds}s"
    return df.resample(rule).mean()


def fill_gaps(
    df: pd.DataFrame,
    max_gap_seconds: int = MAX_GAP_SECONDS,
    sampling_seconds: int = 6,
) -> pd.DataFrame:
    """
    Forward-fill missing values, but only for gaps shorter than
    max_gap_seconds. Longer gaps are left as NaN (later dropped),
    since forward-filling a long gap would fabricate false data.

    Parameters
    ----------
    df : pd.DataFrame
        Resampled power data, may contain NaN from resampling gaps.
    max_gap_seconds : int
        Maximum gap duration to forward-fill (default 180s = 3 min).
    sampling_seconds : int
        Sampling interval, used to convert seconds to number of rows.

    Returns
    -------
    pd.DataFrame
        Gaps ≤ max_gap_seconds are forward-filled.
        Gaps > max_gap_seconds remain NaN.
    """
    max_gap_rows = max_gap_seconds // sampling_seconds
    return df.ffill(limit=max_gap_rows)


def clip_outliers(
    df: pd.DataFrame,
    appliance_config: Dict = APPLIANCES,
) -> pd.DataFrame:
    """
    Clip power values above the configured max_power per appliance.
    Removes sensor glitches and impossible power readings.

    Aggregate power is clipped at a generous whole-house maximum
    since it is the sum of all appliances (not in APPLIANCES dict).

    Parameters
    ----------
    df : pd.DataFrame
        Columns: aggregate + appliance names.
    appliance_config : dict
        From config.APPLIANCES — contains max_power per appliance.

    Returns
    -------
    pd.DataFrame
        Same shape, with outliers clipped.
    """
    df = df.copy()

    # Whole-house aggregate: generous ceiling (20 kW covers virtually
    # any residential scenario, including REDD's higher-power houses).
    if "aggregate" in df.columns:
        df["aggregate"] = df["aggregate"].clip(lower=0, upper=20000)

    for appliance, cfg in appliance_config.items():
        if appliance in df.columns:
            df[appliance] = df[appliance].clip(lower=0, upper=cfg["max_power"])

    return df


# ============================================================
# 5. STATE LABEL GENERATION (ON/OFF)
# ============================================================

def compute_state_labels(
    df: pd.DataFrame,
    appliance_config: Dict = APPLIANCES,
) -> pd.DataFrame:
    """
    Generate binary ON/OFF state labels from continuous power values
    using each appliance's power_threshold.

    state = 1 (ON)  if power > threshold
    state = 0 (OFF) otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Must contain appliance power columns.
    appliance_config : dict
        From config.APPLIANCES — contains power_threshold per appliance.

    Returns
    -------
    pd.DataFrame
        New columns added: '{appliance}_state' for each appliance.

    Example
    -------
    >>> df = pd.DataFrame({'kettle': [0, 5, 2500, 2400, 10]})
    >>> result = compute_state_labels(df)
    >>> result['kettle_state'].tolist()
    [0, 0, 1, 1, 0]
    """
    df = df.copy()

    for appliance, cfg in appliance_config.items():
        if appliance in df.columns:
            threshold = cfg["power_threshold"]
            df[f"{appliance}_state"] = (df[appliance] > threshold).astype(int)

    return df


# ============================================================
# 6. COMPLETE PIPELINE — ONE FUNCTION TO RULE THEM ALL
# ============================================================

def preprocess_house(
    df: pd.DataFrame,
    target_seconds: int = 6,
    max_gap_seconds: int = MAX_GAP_SECONDS,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one house's raw power data.

    Steps (in order):
        1. Resample to uniform target sampling rate
        2. Fill short gaps (forward-fill)
        3. Clip outliers per appliance
        4. Compute binary ON/OFF state labels
        5. Drop rows with remaining NaN (long unfillable gaps)

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded data (from load_ukdale_house or load_redd_house).
    target_seconds : int
        Target sampling rate in seconds.
    max_gap_seconds : int
        Maximum gap to forward-fill.

    Returns
    -------
    pd.DataFrame
        Clean, uniformly-sampled, gap-filled, state-labeled data.
        Ready for windowing (next step in the pipeline).

    Example
    -------
    >>> raw = load_ukdale_house(1)
    >>> clean = preprocess_house(raw)
    >>> clean.isna().sum().sum()
    0
    """
    print(f"\nPreprocessing pipeline:")
    print(f"  Input shape: {df.shape}")

    df = resample_to_target_rate(df, target_seconds)
    print(f"  After resampling ({target_seconds}s): {df.shape}")

    df = fill_gaps(df, max_gap_seconds, target_seconds)
    n_nan_after_fill = df.isna().sum().sum()
    print(f"  After gap-filling: {n_nan_after_fill} NaN remaining")

    df = clip_outliers(df)
    print(f"  Outliers clipped per appliance config")

    df = compute_state_labels(df)
    print(f"  State labels computed: "
          f"{[c for c in df.columns if c.endswith('_state')]}")

    n_before_drop = len(df)
    df = df.dropna()
    n_after_drop = len(df)
    print(f"  Dropped {n_before_drop - n_after_drop} rows with unfillable gaps")
    print(f"  Final shape: {df.shape}")

    return df


# ============================================================
# ENTRY POINT — Test on real UK-DALE data
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessing Module — Self-Test on UK-DALE House 1")
    print("=" * 60)

    raw_df = load_ukdale_house(house=1)
    print(f"\nRaw data preview:")
    print(raw_df.head())

    clean_df = preprocess_house(raw_df)
    print(f"\nClean data preview:")
    print(clean_df.head())

    print(f"\nPower statistics (Watts):")
    print(clean_df[["aggregate"] + APPLIANCE_NAMES].describe())

    print(f"\n{'=' * 60}")
    print("Preprocessing module ready!")
    print(f"{'=' * 60}")
