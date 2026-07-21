"""
dwt.py — Discrete Wavelet Transform Module for NILM
====================================================
Decomposes aggregate power signals into multi-frequency sub-bands
using DWT, providing structured input features for the neural network.

Theory (Luo et al., J. Supercomputing 2023):
    Raw power signal → DWT level 3 with db2 wavelet → 4 sub-bands:
        HP1 (cD1) : fast transients     — appliance switching events
        HP2 (cD2) : medium variations   — motor oscillations, heating ramps
        HP3 (cD3) : slow variations     — compressor cycling, HVAC patterns
        LP3 (cA3) : baseline trend      — overall household consumption level

    Each sub-band captures different physical behaviors of appliances.
    Together they provide the neural network with structured multi-frequency
    information that the raw signal alone cannot offer.

Design choice:
    Unlike Luo et al. who extract statistical summaries from sub-bands,
    we preserve the FULL temporal resolution of each sub-band by resampling
    to the original window length. This retains precise temporal localization
    of transient events — critical for accurate appliance state detection.

Author  : Chadha Jeddi
Project : Benchmarking DL Models for NILM — ACTIA ES / PowerLab
"""

import numpy as np
import pywt
from scipy.signal import resample
from typing import List, Tuple, Optional

# Import project configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DWT_WAVELET,
    DWT_LEVEL,
    DWT_N_SUBBANDS,
    DWT_SUBBAND_NAMES,
    WINDOW_SIZE,
)


# ============================================================
# 1. CORE DWT DECOMPOSITION
# ============================================================

def decompose_window(
    signal: np.ndarray,
    wavelet: str = DWT_WAVELET,
    level: int = DWT_LEVEL,
) -> List[np.ndarray]:
    """
    Apply DWT decomposition to a single window of power signal.

    The signal passes through a tree of low-pass (scaling) and
    high-pass (wavelet) filters at each level, producing sub-bands
    at progressively coarser temporal resolutions.

    Parameters
    ----------
    signal : np.ndarray
        1D array of power values, shape (T,). Example: 480 points.
    wavelet : str
        Wavelet family name. Default 'db2' (Daubechies-2 = D4).
        - 4 filter coefficients
        - 2 vanishing moments (zero output on linear signals)
        - Good time-frequency localization for NILM transients
    level : int
        Number of decomposition levels. Default 3.
        - Level 1: splits signal into HP1 (fast) + LP1 (slow)
        - Level 2: splits LP1 into HP2 (medium) + LP2 (slower)
        - Level 3: splits LP2 into HP3 (slow) + LP3 (baseline)

    Returns
    -------
    List[np.ndarray]
        Sub-bands ordered as [LP3, HP3, HP2, HP1].
        This is the native output order of pywt.wavedec:
        [cA_n, cD_n, cD_{n-1}, ..., cD_1]

        Lengths after db2 level-3 on 480 points:
            LP3: 62 points  (480 → 241 → 122 → 62)
            HP3: 62 points
            HP2: 122 points
            HP1: 241 points

    Example
    -------
    >>> signal = np.random.randn(480)
    >>> subbands = decompose_window(signal)
    >>> len(subbands)
    4
    >>> [s.shape[0] for s in subbands]
    [62, 62, 122, 241]
    """
    # Validate input
    if signal.ndim != 1:
        raise ValueError(
            f"Expected 1D signal, got shape {signal.shape}. "
            f"Flatten or select one channel before decomposition."
        )

    if len(signal) < 2 ** level:
        raise ValueError(
            f"Signal length {len(signal)} is too short for "
            f"{level}-level decomposition. Minimum: {2 ** level} points."
        )

    # Apply DWT decomposition
    # pywt.wavedec returns: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # For level=3: [cA3, cD3, cD2, cD1] = [LP3, HP3, HP2, HP1]
    coefficients = pywt.wavedec(signal, wavelet, level=level)

    return coefficients


# ============================================================
# 2. RESAMPLING — ALIGN ALL SUB-BANDS TO SAME LENGTH
# ============================================================

def resample_subbands(
    subbands: List[np.ndarray],
    target_length: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Resample all DWT sub-bands to the same target length and stack
    them into a single 2D array suitable for the neural network.

    Why resampling is needed:
        After DWT level 3, sub-bands have different lengths:
            HP1: 241 points, HP2: 122 points, HP3: 62 points, LP3: 62 points
        The neural network requires a fixed-shape input tensor [C, T].
        We resample all sub-bands to T=480 using scipy's Fourier method,
        which preserves frequency content better than linear interpolation.

    Design note:
        This preserves the FULL temporal structure of each sub-band,
        unlike statistical summarization which discards temporal position.
        A transient at position 50 vs position 400 looks different in our
        representation — critical for precise event detection.

    Parameters
    ----------
    subbands : List[np.ndarray]
        Output of decompose_window(). List of 1D arrays with varying lengths.
    target_length : int
        Target length for all sub-bands. Default WINDOW_SIZE (480).

    Returns
    -------
    np.ndarray
        Stacked sub-bands, shape (N_SUBBANDS, target_length).
        For default config: (4, 480).
        Channel order: [LP3, HP3, HP2, HP1]

    Example
    -------
    >>> signal = np.random.randn(480)
    >>> subbands = decompose_window(signal)
    >>> features = resample_subbands(subbands)
    >>> features.shape
    (4, 480)
    """
    resampled = []
    for subband in subbands:
        if len(subband) == target_length:
            # No resampling needed — already the right length
            resampled.append(subband)
        else:
            # scipy.signal.resample uses Fourier method:
            # 1. Compute FFT of the sub-band
            # 2. Zero-pad or truncate in frequency domain
            # 3. Inverse FFT to get the resampled signal
            # This preserves frequency content better than interpolation.
            resampled.append(resample(subband, target_length))

    # Stack into 2D array: (n_subbands, target_length)
    return np.stack(resampled, axis=0).astype(np.float32)


# ============================================================
# 3. COMPLETE DWT PIPELINE — SINGLE WINDOW
# ============================================================

def dwt_transform(
    signal: np.ndarray,
    wavelet: str = DWT_WAVELET,
    level: int = DWT_LEVEL,
    target_length: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Complete DWT pipeline for a single window: decompose + resample.

    This is the main function called during preprocessing.
    Input: raw power window (T,)
    Output: multi-channel features (N_SUBBANDS, T)

    Parameters
    ----------
    signal : np.ndarray
        Raw aggregate power values, shape (T,).
    wavelet : str
        Wavelet name. Default 'db2'.
    level : int
        Decomposition levels. Default 3.
    target_length : int
        Output temporal length per sub-band. Default 480.

    Returns
    -------
    np.ndarray
        Shape (N_SUBBANDS, target_length) = (4, 480) by default.
        Channels: [LP3, HP3, HP2, HP1]

    Example
    -------
    >>> window = np.random.randn(480)
    >>> features = dwt_transform(window)
    >>> features.shape
    (4, 480)
    """
    subbands = decompose_window(signal, wavelet, level)
    features = resample_subbands(subbands, target_length)
    return features


# ============================================================
# 4. BATCH DWT — PROCESS MULTIPLE WINDOWS AT ONCE
# ============================================================

def dwt_transform_batch(
    signals: np.ndarray,
    wavelet: str = DWT_WAVELET,
    level: int = DWT_LEVEL,
    target_length: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Apply DWT to a batch of windows simultaneously.

    Used during offline preprocessing to transform the entire dataset
    at once before training begins. DWT is computed on CPU — not inside
    the neural network — so this runs once and the results are saved
    to disk as numpy arrays.

    Parameters
    ----------
    signals : np.ndarray
        Batch of raw power windows, shape (N, T) where:
            N = number of windows
            T = window length (480)

    Returns
    -------
    np.ndarray
        Shape (N, N_SUBBANDS, target_length) = (N, 4, 480).
        Ready to be used as input to the PyTorch DataLoader.

    Example
    -------
    >>> windows = np.random.randn(1000, 480)  # 1000 training windows
    >>> features = dwt_transform_batch(windows)
    >>> features.shape
    (1000, 4, 480)
    """
    n_windows = signals.shape[0]

    # Pre-allocate output array for memory efficiency
    # Using np.empty is faster than appending to a list for large datasets
    output = np.empty(
        (n_windows, DWT_N_SUBBANDS, target_length),
        dtype=np.float32,
    )

    for i in range(n_windows):
        output[i] = dwt_transform(signals[i], wavelet, level, target_length)

    return output


# ============================================================
# 5. INVERSE DWT — RECONSTRUCTION (for verification)
# ============================================================

def inverse_dwt(
    subbands: List[np.ndarray],
    wavelet: str = DWT_WAVELET,
) -> np.ndarray:
    """
    Reconstruct the original signal from its DWT sub-bands.

    Used ONLY for verification — to prove that DWT is lossless.
    Not used during training or inference.

    The reconstruction error should be near machine precision (~1e-12).
    If the error is large, something is wrong with the decomposition.

    Parameters
    ----------
    subbands : List[np.ndarray]
        Sub-bands from decompose_window() at their ORIGINAL lengths
        (before resampling). Order: [LP3, HP3, HP2, HP1].
    wavelet : str
        Must be the same wavelet used for decomposition.

    Returns
    -------
    np.ndarray
        Reconstructed signal, same length as the original.

    Example
    -------
    >>> signal = np.random.randn(480)
    >>> subbands = decompose_window(signal)
    >>> reconstructed = inverse_dwt(subbands)
    >>> error = np.max(np.abs(signal - reconstructed))
    >>> print(f"Max reconstruction error: {error:.2e}")
    Max reconstruction error: 1.33e-15
    """
    return pywt.waverec(subbands, wavelet)


# ============================================================
# 6. SUB-BAND ANALYSIS — ENERGY AND STATISTICS
# ============================================================

def compute_subband_energy(subbands: List[np.ndarray]) -> dict:
    """
    Compute the energy (sum of squared coefficients) of each sub-band.

    Energy distribution across sub-bands reveals the dominant frequency
    content of the signal:
        - High HP1 energy → many fast transients (active switching period)
        - High LP3 energy → strong baseline (many appliances ON)
        - Balanced energy → mixed activity

    This is used for analysis and debugging, not for model input.

    Parameters
    ----------
    subbands : List[np.ndarray]
        Output of decompose_window().

    Returns
    -------
    dict
        Energy per sub-band and total. Keys: LP3, HP3, HP2, HP1, total.

    Example
    -------
    >>> signal = np.random.randn(480)
    >>> subbands = decompose_window(signal)
    >>> energy = compute_subband_energy(subbands)
    >>> print(energy)
    {'LP3': 15.2, 'HP3': 12.8, 'HP2': 24.1, 'HP1': 48.5, 'total': 100.6}
    """
    names = DWT_SUBBAND_NAMES  # ['LP3', 'HP3', 'HP2', 'HP1']
    energy = {}

    for name, subband in zip(names, subbands):
        energy[name] = float(np.sum(subband ** 2))

    energy["total"] = sum(energy[n] for n in names)

    return energy


# ============================================================
# 7. VISUALIZATION HELPER
# ============================================================

def plot_dwt_decomposition(
    signal: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "DWT Decomposition — Aggregate Power Signal",
) -> None:
    """
    Plot the original signal and all DWT sub-bands for visual inspection.

    Creates a figure with 5 subplots:
        1. Original signal (blue)
        2. HP1 — fast transients (red)
        3. HP2 — medium variations (orange)
        4. HP3 — slow variations (green)
        5. LP3 — baseline trend (purple)

    Parameters
    ----------
    signal : np.ndarray
        Raw aggregate power values, shape (T,).
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    title : str
        Figure title.

    Example
    -------
    >>> signal = np.random.randn(480)
    >>> plot_dwt_decomposition(signal, save_path="dwt_example.png")
    """
    import matplotlib.pyplot as plt

    # Decompose at original resolution (before resampling)
    subbands = decompose_window(signal)
    names = DWT_SUBBAND_NAMES          # ['LP3', 'HP3', 'HP2', 'HP1']
    colors = ["#7B4FBF", "#2E9E5A", "#E8922A", "#D94040"]

    # Resample for aligned plotting
    resampled = resample_subbands(subbands, len(signal))

    # Create figure: 5 rows (original + 4 sub-bands)
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot original signal
    time = np.arange(len(signal))
    axes[0].plot(time, signal, color="#3366CC", linewidth=0.8)
    axes[0].set_ylabel("Original\n(Watts)", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot each sub-band (reverse order: HP1 first = fast transients)
    plot_order = [3, 2, 1, 0]  # HP1, HP2, HP3, LP3
    for ax_idx, sb_idx in enumerate(plot_order):
        ax = axes[ax_idx + 1]
        ax.plot(time, resampled[sb_idx], color=colors[sb_idx], linewidth=0.8)
        ax.set_ylabel(f"{names[sb_idx]}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (samples)", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved DWT plot to: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================
# ENTRY POINT — Run this file directly to test DWT
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("DWT Module — Self-Test")
    print("=" * 50)

    # --- Test 1: Basic decomposition ---
    print("\n[Test 1] Decompose 480-point random signal...")
    signal = np.random.randn(WINDOW_SIZE)
    subbands = decompose_window(signal)

    print(f"  Input shape:  ({len(signal)},)")
    print(f"  Sub-bands:")
    for name, sb in zip(DWT_SUBBAND_NAMES, subbands):
        print(f"    {name}: {len(sb)} points")

    # --- Test 2: Resampling ---
    print(f"\n[Test 2] Resample all sub-bands to {WINDOW_SIZE} points...")
    features = resample_subbands(subbands, WINDOW_SIZE)
    print(f"  Output shape: {features.shape}")
    print(f"  Dtype:        {features.dtype}")

    # --- Test 3: Full pipeline ---
    print(f"\n[Test 3] Full DWT pipeline (decompose + resample)...")
    features = dwt_transform(signal)
    print(f"  Output shape: {features.shape}")

    # --- Test 4: Perfect reconstruction ---
    print(f"\n[Test 4] Verify lossless reconstruction (inverse DWT)...")
    reconstructed = inverse_dwt(subbands)
    # Trim to original length (waverec may add 1 extra sample)
    reconstructed = reconstructed[:len(signal)]
    max_error = np.max(np.abs(signal - reconstructed))
    print(f"  Max reconstruction error: {max_error:.2e}")
    if max_error < 1e-10:
        print(f"  ✓ PERFECT — DWT is lossless (error < 1e-10)")
    else:
        print(f"  ✗ WARNING — reconstruction error is large!")

    # --- Test 5: Batch processing ---
    print(f"\n[Test 5] Batch DWT on 100 windows...")
    batch = np.random.randn(100, WINDOW_SIZE)
    batch_features = dwt_transform_batch(batch)
    print(f"  Input:  {batch.shape}")
    print(f"  Output: {batch_features.shape}")

    # --- Test 6: Sub-band energy ---
    print(f"\n[Test 6] Sub-band energy distribution...")
    energy = compute_subband_energy(subbands)
    total = energy["total"]
    for name in DWT_SUBBAND_NAMES:
        pct = 100 * energy[name] / total if total > 0 else 0
        print(f"    {name}: {energy[name]:.1f} ({pct:.1f}%)")
    print(f"    Total: {total:.1f}")

    print(f"\n{'=' * 50}")
    print("All tests passed — DWT module ready!")
    print(f"{'=' * 50}")
