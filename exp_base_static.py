# ============================================================================
# exp_base_static.py — Static definitions for fNIRS Pipeline
# ============================================================================
#
# CONTENTS:
#   Data I/O: CSV reading, folder loading, time-aligned merge
#   Y-only feature builder: 80+ causal features from ShortChannel (Section 2.5)
#     - Causal lags, diffs, jerk
#     - Multi-scale EMAs + EMA slopes
#     - Rolling mean/std at multiple windows
#     - Kalman filter trend bank
#     - Bandpass energy (HR/RR/trend bands)
#     - Causal envelope, zero-crossing rate
#     - Cardiac phase encoding (RETROICOR-style)
#   X feature builder: stable normalisation of 4 cardiovascular channels
#   YGivenXModel: Stage 1 teacher P(y|x) (Section 2.3)
#     - ConditionalFlow (normalizing flow)
#     - ResidualHeadStudentT (heavy-tailed correction)
#     - Teacher tap extraction + whitener fitting
# ============================================================================
# experiment_train_minimal_fixed.py
# Teacher from conditioner taps + geometry-aware X-from-Y encoder + GAN discriminator
#
# FIX PACK (8 items) applied in this file:
#   1) Stable X normalisation: compute x_norm stats on TRAIN split only; save to scaler; reuse at inference.
#   2) Remove per-run/inference X renormalisation drift: build_x_features_from_df now accepts x_norm stats.
#   3) Cond-gain anneal now honored (cond_gain_start -> cond_gain_end) and logged.
#   4) Added feature-distribution debug (train/val x,y std stats) and per-epoch config debug.
#   5) WaveNet student made strictly CAUSAL (left padding) to prevent lookahead leakage.
#   6) Teacher sensitivity gradient J computed on FULL teacher output y_mean (not y_base only).
#   7) GAN stability: one-sided label smoothing + small label noise + optional R1 penalty on real.
#   8) Extra NaN/Inf guards and richer GAN/encoder diagnostics for pointwise debugging.
#   9) PHASE FIX: Added 5 cardiac phase-encoding features to break y-only variance collapse.
from __future__ import annotations
import os, json, time, random, math
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Import flow components from core_flow.py ---
from core_flow import (
    ema_causal, lag_np_causal, kf_local_trend,
    standardize_train_stats, apply_scaling_np, invert_scaling_np,
    ConditionalFlow,
    ResidualHeadStudentT,
    student_t_nll,
    safe_autograd_grad, reset_actnorm_flags,
    _count_params
)

# --- Reproducibility: fix all random seeds ---
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Create parent directory if it doesn't exist ---
def _ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

# --- Save figure to path with tight layout, auto-create dirs ---
def _savefig(fig, path):
    _ensure_dir(path)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

# ============================================================================
# DATA I/O
# ============================================================================
# read_param_csv: load a single parameter CSV (Time + one column)
# load_folder_and_join: load all 5 CSVs and inner-join on Time
# ---------------------- Data I/O ----------------------

# --- Load one CSV: Time + parameter column, clean NaN/Inf ---
def read_param_csv(path):
    t0 = time.perf_counter()
    # Fast C-engine CSV read with memory mapping for large files
    df = pd.read_csv(path, engine="c", low_memory=False, memory_map=True)
    cols = list(df.columns)
    if len(cols) < 2: raise ValueError(f"{path} must have at least two columns: Time,<Param>")
    df = df.rename(columns={cols[0]: "Time"})
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").astype(np.float32)
    df[cols[1]] = pd.to_numeric(df[cols[1]], errors="coerce").astype(np.float32)
    before = len(df)
    # Safety: replace infinities with NaN, then drop all NaN rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    t1 = time.perf_counter()
    print(f"[data] Loaded {os.path.basename(path)}: {before}→{len(df)} rows in {t1 - t0:.2f}s")
    return df[["Time", cols[1]]], cols[1]

# --- Load all 5 CSVs from folder and inner-join on Time ---
# Expected files: ShortChannel.csv, DiastolicBP.csv, SystolicBP.csv,
#                  CardiacOutput.csv, StrokeVolume.csv
# Returns single merged DataFrame sorted by Time.
def load_folder_and_join(folder):
    names = ["ShortChannel", "DiastolicBP", "SystolicBP", "CardiacOutput", "StrokeVolume"]
    files = {n: os.path.join(folder, f"{n}.csv") for n in names}
    print("[data] Looking for:", *(f"       {n}→{p}" for n,p in files.items()), sep="\n")
    for n, p in files.items():
        if not os.path.isfile(p): raise FileNotFoundError(f"Missing expected file: {p}")
    dfs = {}
    for n, p in files.items():
        df, col = read_param_csv(p); dfs[n] = df.rename(columns={col: n})
    t0 = time.perf_counter()
    # Sequential inner join: start with ShortChannel, merge each param on Time
    # Only rows with matching timestamps across ALL files survive
    merged = dfs["ShortChannel"]
    for n in ["DiastolicBP", "SystolicBP", "CardiacOutput", "StrokeVolume"]:
        merged = pd.merge(merged, dfs[n], on="Time", how="inner", sort=False, copy=False)
    merged = merged.sort_values("Time", kind="mergesort").reset_index(drop=True)
    t1 = time.perf_counter()
    print(f"[data] Merged frame: {len(merged)} rows in {t1 - t0:.2f}s")
    return merged

# ============================================================================
# Y-ONLY FEATURE BUILDER (Section 2.5)
# ============================================================================
# Builds 80+ causal features from the ShortChannel (fNIRS) signal alone.
# These features are the input to the WaveNet student encoder (Stage 2a).
# ALL features are strictly causal: only use y[t] and earlier, never future.
# ---------------------- Light feature builder (extended y-memory) ----------------------

# --- Local wrapper for core_flow.ema_causal ---
def ema_causal_local(x, alpha=0.2):
    return ema_causal(x, alpha=alpha)

# --- Local wrapper for core_flow.kf_local_trend ---
def kf_local_trend_local(z, dt=1.0, q=1e-3, r=None, adapt=True, adapt_beta=0.10, eps=1e-9):
    return kf_local_trend(z, dt=dt, q=q, r=r, adapt=adapt, adapt_beta=adapt_beta, eps=eps)

# --- Local wrapper for core_flow.lag_np_causal ---
def lag_np_causal_local(x, k):
    return lag_np_causal(x, k)

# --- Causal rolling mean using cumulative sum trick ---
# First w-1 samples use expanding window (no lookahead).
# O(N) complexity via cumsum instead of naive O(N*w).
def _roll_mean_causal(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]
    if w <= 1 or N == 0:
        return x.copy()
    # Cumsum trick: mean[i] = (cumsum[i] - cumsum[i-w]) / w
    # Use float64 for cumsum to avoid precision loss on long signals
    c = np.cumsum(x, dtype=np.float64)
    out = np.empty(N, dtype=np.float32)
    if N < w:
        out[:] = (c / np.arange(1, N+1, dtype=np.float64)).astype(np.float32)
        return out
    out[:w-1] = (c[:w-1] / np.arange(1, w, dtype=np.float64)).astype(np.float32)
    # Steady-state: subtract cumsum[i-w] from cumsum[i] to get window sum
    tail_sum = c[w-1:] - np.concatenate(([0.0], c[:N-w]))
    out[w-1:] = (tail_sum / w).astype(np.float32)
    return out

# --- Causal rolling std: sqrt(E[x^2] - E[x]^2) ---
def _roll_std_causal(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = _roll_mean_causal(x, w)
    m2 = _roll_mean_causal(x*x, w)
    v = np.maximum(m2 - m*m, 1e-8).astype(np.float32)
    np.sqrt(v, out=v)
    return v

# --- EMA of first difference: smooth(y[t] - y[t-1]) ---
def _ema_diff(series: np.ndarray, alpha: float) -> np.ndarray:
    lag1 = lag_np_causal_local(series, 1)
    d1 = series - lag1
    return ema_causal_local(d1, alpha=alpha)

# ---- Causal oscillation-preserving helpers ----

# --- Bandpass energy: IIR bandpass -> square -> rolling mean ---
# Measures power in a frequency band (e.g. cardiac 0.5-2.5 Hz)
# Uses lfilter (causal IIR), NOT filtfilt (which is non-causal).
def _causal_bandpass_energy(y: np.ndarray, fs: float, low_hz: float, high_hz: float,
                            energy_window: int = 16) -> np.ndarray:
    """Causal bandpass energy: IIR bandpass (lfilter = forward-only) then rolling squared amplitude."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 0.005)
    high = min(high_hz / nyq, 0.995)
    if low >= high:
        return np.zeros_like(y, dtype=np.float32)
    # 2nd-order Butterworth IIR bandpass
    # lfilter applies it causally (forward-only, introduces phase lag but no lookahead)
    b, a = butter(2, [low, high], btype='band')
    filtered = lfilter(b, a, y).astype(np.float32)
    sq = filtered ** 2
    return _roll_mean_causal(sq, energy_window).astype(np.float32)

# --- Causal envelope: bandpass -> rectify -> lowpass smooth ---
# Extracts the slowly-varying amplitude of oscillations in a band.
def _causal_envelope(y: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Causal analytic envelope: bandpass -> rectify -> lowpass smooth."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 0.005)
    high = min(high_hz / nyq, 0.995)
    if low >= high:
        return np.zeros_like(y, dtype=np.float32)
    b, a = butter(2, [low, high], btype='band')
    filtered = lfilter(b, a, y).astype(np.float32)
    # Full-wave rectification: |bandpassed signal| gives instantaneous amplitude
    # Then lowpass to get smooth envelope (remove carrier frequency)
    rect = np.abs(filtered)
    env_cutoff = min(low_hz / (2.0 * nyq), 0.45)
    if env_cutoff > 0.005:
        b2, a2 = butter(2, env_cutoff, btype='low')
        return lfilter(b2, a2, rect).astype(np.float32)
    return ema_causal(rect, alpha=0.05).astype(np.float32)

# --- Causal zero-crossing rate: cheap frequency proxy ---
# Counts sign changes in a rolling window. High ZCR = high frequency.
def _causal_zero_crossing_rate(y: np.ndarray, window: int = 16) -> np.ndarray:
    """Causal rolling zero-crossing rate — cheap frequency proxy."""
    signs = np.sign(y)
    # Binary: 1 where sign changes, 0 otherwise. Rolling mean = rate.
    crossings = (np.abs(np.diff(signs, prepend=signs[0])) > 0).astype(np.float32)
    return _roll_mean_causal(crossings, window).astype(np.float32)

# --- Cardiac phase-encoding helpers (RETROICOR-style) ---
# These provide instantaneous cardiac phase to the WaveNet student.
# Critical for breaking variance collapse: phase tells the network
# WHERE in the cardiac cycle each sample is (peak vs trough vs rising).
# ---- NEW: Causal phase-encoding helpers ----

# --- Causal bandpass: returns the SIGNED oscillation (not energy/envelope) ---
# This is the raw cardiac waveform isolated from other frequencies.
def _causal_bandpass_signal(y: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Causal bandpass filter returning the SIGNED oscillating signal (not energy/envelope)."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 0.005)
    high = min(high_hz / nyq, 0.995)
    if low >= high:
        return np.zeros_like(y, dtype=np.float32)
    b, a = butter(2, [low, high], btype='band')
    return lfilter(b, a, y).astype(np.float32)

# --- RETROICOR-style causal cardiac phase estimation ---
# Algorithm:
#   1. Detect peaks in bandpassed signal (with refractory period)
#   2. Between consecutive peaks: linearly interpolate phase 0 -> 2*pi
#   3. Before first / after last peak: extrapolate using nearest interval
# Returns: cos(phase), sin(phase), cos(2*phase), sin(2*phase)
# These 4 features encode cardiac position without discontinuities.
def _causal_cardiac_phase(y_bp: np.ndarray, fs: float, refractory_sec: float = 0.4):
    """
    RETROICOR-style causal cardiac phase estimation.
    Detects peaks in bandpass-filtered signal, linearly interpolates phase between beats.
    Returns: cos_phase, sin_phase, cos_2phase, sin_2phase  (each [N] float32)
    """
    N = len(y_bp)
    # Refractory period: minimum samples between peaks (prevents double-detection)
    refractory = max(2, int(refractory_sec * fs))

    # Detect peaks causally (local max with refractory period)
    # Causal peak detection: local max that exceeds refractory spacing
    # Only looks at y_bp[i-1], y_bp[i], y_bp[i+1] — the i+1 is the next sample,
    # which is available at time i in a causal stream.
    peak_indices = []
    for i in range(1, N - 1):
        if y_bp[i] > y_bp[i - 1] and y_bp[i] >= y_bp[min(i + 1, N - 1)]:
            if len(peak_indices) == 0 or (i - peak_indices[-1]) >= refractory:
                peak_indices.append(i)

    # Not enough peaks — return zeros (safe fallback)
    # Safety: if fewer than 2 peaks found, return zeros (can't estimate phase)
    if len(peak_indices) < 2:
        z = np.zeros(N, dtype=np.float32)
        return z.copy(), z.copy(), z.copy(), z.copy()

    phase = np.zeros(N, dtype=np.float32)

    # Before first peak: extrapolate using first interval
    # Before first peak: extrapolate backwards using first inter-peak interval
    first_interval = peak_indices[1] - peak_indices[0]
    for t in range(peak_indices[0] + 1):
        samples_before = peak_indices[0] - t
        phase[t] = 2 * np.pi * (1.0 - min(samples_before / max(first_interval, 1), 1.0))

    # Between consecutive peaks: linear interpolation
    # Between consecutive peaks: linear phase interpolation 0 -> 2*pi
    # phase[p1] = 0 (at peak), phase[p2-1] = ~2*pi (just before next peak)
    for k in range(len(peak_indices) - 1):
        p1 = peak_indices[k]
        p2 = peak_indices[k + 1]
        interval = p2 - p1
        for t in range(p1, p2):
            phase[t] = 2 * np.pi * (t - p1) / max(interval, 1)

    # After last peak: extrapolate using last interval
    # After last peak: extrapolate forward using last inter-peak interval
    last_peak = peak_indices[-1]
    last_interval = peak_indices[-1] - peak_indices[-2]
    for t in range(last_peak, N):
        frac = min((t - last_peak) / max(last_interval, 1), 1.0)
        phase[t] = 2 * np.pi * frac

    # Return cos/sin of phase and 2*phase (4 features)
    # cos(phi): +1 at peak, -1 at trough
    # sin(phi): +1 rising, -1 falling
    # cos(2*phi)/sin(2*phi): sub-beat systole/diastole
    return (np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
            np.cos(2 * phase).astype(np.float32),
            np.sin(2 * phase).astype(np.float32))

# ============================================================================
# build_y_only_features — Main Y feature extraction (Section 2.5)
# ============================================================================
# Builds ALL causal features from ShortChannel for the WaveNet student.
# Feature groups: lags, diffs, EMAs, rolling stats, Kalman, oscillation, phase.
# Output: [N, ~80] float32 array + feature names list.
# ---- y-only feature builder for distillation ----
# --- Entry point: DataFrame -> (feature_array [N, D], feature_names) ---
def build_y_only_features(df, fs_hint: float = None):
    # Extract the ShortChannel (fNIRS) signal as the raw observation y
    y = df["ShortChannel"].to_numpy(np.float32)
    N = len(y)

    # ---- Estimate sampling rate ----
    if fs_hint is not None and fs_hint > 0:
        fs = float(fs_hint)
    elif "Time" in df.columns and N > 5:
        dt = float(np.median(np.diff(df["Time"].to_numpy(np.float32))))
        fs = 1.0 / max(dt, 1e-6)
    else:
        fs = 6.0  # conservative default
    nyq = fs / 2.0



    # Causal lags
    # === GROUP 1: Causal lags y[t-k] for k=1..16 ===
    # Gives the network direct access to recent history
    lags = [1,2,3,4,5,8,12,16]
    y_l = [lag_np_causal(y, k) for k in lags]

    # 1st/2nd diffs + lagged diff (all causal)
    # === GROUP 2: Causal differences ===
    # dy: first difference (velocity), ddy: second difference (acceleration)
    # dy_l1: lagged first difference (jolt/change in velocity)
    dy    = y_l[0] - y_l[1]           # y[t-1]-y[t-2]
    ddy   = y - 2.0*y_l[0] + y_l[1]
    dy_l1 = dy - lag_np_causal(dy, 1)

    # Multi-scale EMAs (all causal)
    # === GROUP 3: Multi-scale EMAs ===
    # Small alpha (0.02) = slow/smooth trend, large alpha (0.80) = fast/responsive
    # These give the network a multi-resolution temporal view of y
    alphas = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    y_emas = [ema_causal(y, a) for a in alphas]

    # EMA slopes (difference between adjacent EMA scales)
    # EMA slopes: difference between adjacent EMA scales
    # Positive slope = signal rising at that timescale
    ema_slopes = []
    for i in range(1, len(y_emas)):
        ema_slopes.append(y_emas[i] - y_emas[i-1])

    # Rolling mean/std (causal windows)
        # Local causal rolling mean (same as module-level but defined inside for scope)
    def _roll_mean_causal_local(x, w):
        x = np.asarray(x, np.float32)
        if w <= 1: return x.copy()
        c = np.cumsum(x, dtype=np.float64)
        out = np.empty_like(x)
        n = x.shape[0]
        if n < w:
            out[:] = (c / np.arange(1, n+1, dtype=np.float64)).astype(np.float32)
        else:
            out[:w-1] = (c[:w-1] / np.arange(1, w, dtype=np.float64)).astype(np.float32)
            out[w-1:] = ((c[w-1:] - np.concatenate(([0.0], c[:-w]))) / w).astype(np.float32)
        return out

        # Local causal rolling std: sqrt(E[x^2] - E[x]^2)
    def _roll_std_causal_local(x, w):
        m  = _roll_mean_causal_local(x, w)
        m2 = _roll_mean_causal_local(x*x, w)
        v = np.maximum(m2 - m*m, 1e-8).astype(np.float32)
        return np.sqrt(v, out=v)

    # === GROUP 4: Rolling mean, std, slope at multiple windows ===
    # y_mean_w: moving average at each window
    # y_std_w: moving standard deviation (local volatility)
    # y_slope_w: y - moving_average (deviation from local trend)
    wins = [8,16,32,64,128]
    y_mean_w = [ _roll_mean_causal_local(y, w) for w in wins ]
    y_std_w  = [ _roll_std_causal_local(y,  w) for w in wins ]
    y_slope_w = [ y - m for m in y_mean_w ]

    # Light Kalman trend bank (causal)
    # === GROUP 5: Kalman filter trend bank ===
    # Multiple process noise q values: small q = smooth trend, large q = responsive
    # kf_smooth: Kalman-smoothed position estimate
    # kf_innov_ema: EMA-smoothed innovations (surprise signal)
    kf_q = [5e-4, 1e-3, 2e-3, 5e-3]
    kf_smooth = []
    kf_innov_ema = []
    for q in kf_q:
        # Run Kalman filter with this q, extract smoothed state and innovations
        kf = kf_local_trend(y, q=q)
        sm = kf["xhat"].astype(np.float32)
        inv = kf["innov"].astype(np.float32)
        kf_smooth.append(sm)
        kf_innov_ema.append(ema_causal(inv, alpha=0.2))

    # Stack EXISTING features
    # Stack all existing features: lags + diffs + EMAs + rolling + Kalman
    cols = []
    cols += y_l                          # lags
    cols += [dy, ddy, dy_l1]             # diffs
    cols += y_emas + ema_slopes          # multi-scale levels + slopes
    cols += y_mean_w + y_std_w + y_slope_w
    cols += kf_smooth + kf_innov_ema

    names = [f"Y_lag{k}" for k in lags] \
          + ["Y_d1","Y_dd","Y_dacc"] \
          + [f"Y_ema_a{str(a).replace('0.','0p')}" for a in alphas] \
          + [f"Y_emaSlope_{i}" for i in range(1, len(alphas))] \
          + [f"Y_mean_w{w}" for w in wins] \
          + [f"Y_std_w{w}"  for w in wins] \
          + [f"Y_slope_w{w}" for w in wins] \
          + [f"Y_kf_q{q:.0e}" for q in kf_q] \
          + [f"Y_kf_innovEma_q{q:.0e}" for q in kf_q]

    # Track feature count for reporting
    n_existing = len(cols)

    # === GROUP 6: Oscillation-preserving features (25 features) ===
    # These preserve the actual WAVEFORM that EMAs and rolling stats destroy.
    # (A) y[t] itself + multi-scale derivatives + jerk
    # (B-C) Bandpass energy/envelope in HR, RR, and trend frequency bands
    # (D) Zero-crossing rates (frequency proxy)
    # (E) Short-window variance (local volatility at fine scale)
    # (F) EMA residuals (what the EMA misses)
    # (G) Kalman residual + raw innovation
    # ============================================================
    # OSCILLATION-PRESERVING FEATURES (25 features)
    # ============================================================

    # --- (A) y[t] itself ---
    cols.append(y)
    names.append("Y_t0")

    # --- (B) Multi-scale causal derivatives ---
    dy1 = y - lag_np_causal(y, 1)
    cols.append(dy1)
    names.append("Y_dy1")

    for step in [2, 4, 8]:
        dx_k = (y - lag_np_causal(y, step)) / float(step)
        cols.append(dx_k)
        names.append(f"Y_dy_step{step}")

    _yl1 = lag_np_causal(y, 1)
    _yl2 = lag_np_causal(y, 2)
    _yl3 = lag_np_causal(y, 3)
    # Jerk = d^3y/dt^3 via 3rd-order finite difference (causal)
    jerk = y - 3*_yl1 + 3*_yl2 - _yl3
    cols.append(jerk)
    names.append("Y_jerk")

    # --- (C) Causal bandpass energy ---
    # (C) Causal bandpass energy + envelope in cardiac band (0.5-2.5 Hz)
    # Only computed if Nyquist > 0.7 Hz (fs > 1.4 Hz)
    if nyq > 0.7:
        cols.append(_causal_bandpass_energy(y, fs, 0.5, min(2.5, nyq*0.95), energy_window=16))
        names.append("Y_bp_energy_HR")
        cols.append(_causal_envelope(y, fs, 0.5, min(2.5, nyq*0.95)))
        names.append("Y_bp_envelope_HR")

    if nyq > 0.15:
        cols.append(_causal_bandpass_energy(y, fs, 0.08, min(0.5, nyq*0.95), energy_window=32))
        names.append("Y_bp_energy_RR")
        cols.append(_causal_envelope(y, fs, 0.08, min(0.5, nyq*0.95)))
        names.append("Y_bp_envelope_RR")

    if nyq > 0.05:
        cols.append(_causal_bandpass_energy(y, fs, 0.01, min(0.1, nyq*0.95), energy_window=64))
        names.append("Y_bp_energy_Trend")

    # Total energy: rolling mean of y^2 (broadband power)
    cols.append(_roll_mean_causal(y**2, 16))
    names.append("Y_total_energy")

    # --- (D) Causal zero-crossing rate ---
    for w in [8, 16, 32]:
        cols.append(_causal_zero_crossing_rate(y, window=w))
        names.append(f"Y_zcr_w{w}")
    for w in [8, 16]:
        cols.append(_causal_zero_crossing_rate(dy1, window=w))
        names.append(f"Y_zcr_dy_w{w}")

    # --- (E) Short-window variance ---
    for w in [4, 6]:
        cols.append(_roll_std_causal(y, w))
        names.append(f"Y_std_short_w{w}")
    for w in [4, 8]:
        cols.append(_roll_std_causal(dy1, w))
        names.append(f"Y_std_dy_w{w}")

    # --- (F) EMA residuals ---
    for alpha in [0.05, 0.20]:
        cols.append(y - ema_causal(y, alpha))
        names.append(f"Y_ema_resid_{alpha:.2f}")

    # (G) Kalman residual (y - smoothed) + raw innovation
    # The innovation is what the Kalman filter didn't predict — surprise signal.
    # --- (G) Kalman residual + raw innovation ---
    # Use q=1e-3 Kalman (index 1) as the reference smoother
    kf_main = kf_smooth[1]
    cols.append(y - kf_main)
    names.append("Y_kf_resid")
    kf_raw = kf_local_trend(y, q=1e-3)
    cols.append(kf_raw["innov"].astype(np.float32))
    names.append("Y_kf_innov_raw")

    n_oscillation = len(cols) - n_existing

    # === GROUP 7: Cardiac phase-encoding features (5 features) ===
    # These break the y-only variance collapse by encoding WHERE in
    # the cardiac cycle each sample sits. Without these, the WaveNet
    # can only see amplitude, not phase — so it learns the mean.
    # ============================================================
    # NEW: CARDIAC PHASE-ENCODING FEATURES (5 features)
    # These break the variance collapse by giving the WaveNet
    # instantaneous cardiac phase information.
    # ============================================================

    if nyq > 0.7:
        # (H1) Raw bandpass-filtered cardiac signal — the signed oscillation itself
        #      This is THE critical feature: tells the network "pressure is HIGH now"
        #      vs "pressure is LOW now" at each timestep.
        # (H1) Raw cardiac oscillation: the signed bandpass signal
        # This is THE critical feature — tells network "pressure HIGH" vs "pressure LOW"
        y_bp_cardiac = _causal_bandpass_signal(y, fs, 0.5, min(2.5, nyq * 0.95))
        cols.append(y_bp_cardiac)
        names.append("Y_bp_cardiac_raw")

        # (H2-H5) cos/sin of cardiac phase from RETROICOR-style causal peak detection
        #          These provide amplitude-independent, discontinuity-free phase encoding.
        #          cos(φ): +1 at peak, -1 at trough
        #          sin(φ): +1 rising, -1 falling
        #          cos(2φ)/sin(2φ): sub-beat systole/diastole structure
        # (H2-H5) RETROICOR phase: cos(phi), sin(phi), cos(2*phi), sin(2*phi)
        # Amplitude-independent, discontinuity-free cardiac phase encoding
        cos_p, sin_p, cos_2p, sin_2p = _causal_cardiac_phase(y_bp_cardiac, fs)
        cols.append(cos_p)
        names.append("Y_cardiac_cos_phase")
        cols.append(sin_p)
        names.append("Y_cardiac_sin_phase")
        cols.append(cos_2p)
        names.append("Y_cardiac_cos_2phase")
        cols.append(sin_2p)
        names.append("Y_cardiac_sin_2phase")
    else:
        # Fallback: if fs too low for cardiac, add zero placeholders
        # Keeps feature dimension constant regardless of sampling rate
        # Below cardiac frequency range — add zeros as placeholders to keep dim constant
        for nm in ["Y_bp_cardiac_raw", "Y_cardiac_cos_phase", "Y_cardiac_sin_phase",
                    "Y_cardiac_cos_2phase", "Y_cardiac_sin_2phase"]:
            cols.append(np.zeros(N, dtype=np.float32))
            names.append(nm)

    n_phase = len(cols) - n_existing - n_oscillation

    # ============================================================
    # STACK & SAFETY
    # ============================================================
    # Stack all feature columns into [N, D] array
    # Replace any NaN/Inf with safe values (0 for NaN, ±10 for ±Inf)
    yfeat = np.stack(cols, axis=1).astype(np.float32)
    yfeat = np.nan_to_num(yfeat, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"[y_only_features] N={N} fs={fs:.2f}Hz | {n_existing} existing + {n_oscillation} oscillation + {n_phase} phase = {len(cols)} total features")

    return yfeat, names

# ============================================================================
# STABLE X FEATURE BUILDER
# ============================================================================
# Builds teacher conditioning features from the 4 cardiovascular channels.
# FIX: X normalisation computed on TRAIN split only and saved to scaler
# so train/val/inference all use the same stats (prevents drift).
# ---------------------- Stable X feature builder ----------------------

# --- Build normalised X features from DataFrame ---
# Inputs: HeartRate, RespirationRate, CO2, OtherFactor
# Output: [N,4] normalised features + stats for reuse at inference
def build_x_features_from_df(
    df: pd.DataFrame,
    x_norm_mu: np.ndarray | None = None,
    x_norm_sd: np.ndarray | None = None,
    fit_slice: slice | None = None,
    save_stats_path: str | None = None
):
    """
    Teacher inputs: ONLY use the 4 normalized raw channels (HeartRate, RespirationRate, CO2, OtherFactor).
    FIX: X normalization must be stable across train/val/inference.
      - If x_norm_mu/sd provided: use them.
      - Else: fit on df[fit_slice] (TRAIN split only) and optionally save stats.
    Returns:
      x_raw_raw: [N,4] raw
      x_feat   : [N,4] normalized with x_norm_mu/sd
      feat_names
      x_norm_stats dict
    """
    # Extract the 4 raw cardiovascular channels
    dbp = df["DiastolicBP"].to_numpy(np.float32)
    sbp = df["SystolicBP"].to_numpy(np.float32)
    co = df["CardiacOutput"].to_numpy(np.float32)
    sv = df["StrokeVolume"].to_numpy(np.float32)

    # Stack into [N,4] array — these are the raw (unnormalised) teacher inputs
    x_raw_raw = np.stack([dbp, sbp, co, sv], axis=1).astype(np.float32)

    # If no pre-computed stats: fit mean/std on TRAIN split only
    # This prevents train/val/test normalisation drift
    if x_norm_mu is None or x_norm_sd is None:
        fit_slice = fit_slice or slice(0, x_raw_raw.shape[0])
        x_fit = x_raw_raw[fit_slice]
        # Per-channel mean and std from training data
        # Floor std at 1e-6 -> 1.0 to prevent division by zero on constant channels
        mu = x_fit.mean(axis=0, keepdims=True)
        sd = x_fit.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        x_norm_mu = mu
        x_norm_sd = sd
    else:
        x_norm_mu = np.asarray(x_norm_mu, dtype=np.float32).reshape(1, -1)
        x_norm_sd = np.asarray(x_norm_sd, dtype=np.float32).reshape(1, -1)
        x_norm_sd = np.where(x_norm_sd < 1e-6, 1.0, x_norm_sd)

    # Apply standardisation: (x - mu) / sigma
    x_feat = (x_raw_raw - x_norm_mu) / x_norm_sd

    x_norm_stats = {
        "x_norm_mu": x_norm_mu.squeeze().tolist(),
        "x_norm_sigma": x_norm_sd.squeeze().tolist()
    }

    # Optionally save stats to JSON for reuse at inference
    if save_stats_path is not None:
        _ensure_dir(save_stats_path)
        with open(save_stats_path, "w") as f:
            json.dump(x_norm_stats, f, indent=2)
        print(f"[encoder] Saved X normalisation stats → {save_stats_path}")

    feat_names = ["DBP_n", "SBP_n", "CO_n", "SV_n"]
    return x_raw_raw, x_feat.astype(np.float32), feat_names, x_norm_stats

# ============================================================================
# YGivenXModel — Stage 1 Teacher: P(y|x) (Section 2.3)
# ============================================================================
# The teacher model that learns the conditional distribution of fNIRS
# observation y given cardiovascular state x.
#
# Architecture:
#   flow_y_given_x: ConditionalFlow (normalizing flow, K blocks)
#   res_y: ResidualHeadStudentT (heavy-tailed correction)
#   delta_skip: linear shortcut x -> delta_y
#
# Prediction: y_mean = flow.inverse(z=0, cond=x) + res_delta + skip(x)
#
# Teacher embedding:
#   Preferred: extract_conditioner_taps (internal activations)
#   Fallback: conditioner_hidden (h vector)
#   Last resort: orthogonal projection (frozen)
# ---------------------- Minimal model (+ teacher taps) ----------------------

# --- Stage 1 teacher model: flow + residual head ---
class YGivenXModel(nn.Module):
    """
    Teacher embedding comes from flow.extract_conditioner_taps (preferred),
    with a fallback orthogonal projection if taps aren't available.
    """
    def __init__(self, d_x_cond, d_y=1, d_h=128, K_blocks=4, K_bins=8, bound=6.0, temp=0.45,
                 slope_bias_y=1.5, cond_gain_y=12.0):
        super().__init__()
        # The normalizing flow: P(y|x) via K coupling blocks (Section 2.3)
        # d_var=1 (scalar y), d_cond=4 (normalised x features)
        self.flow_y_given_x = ConditionalFlow(
            d_var=d_y, d_cond=d_x_cond, d_h=d_h, K_blocks=K_blocks,
            K_bins=K_bins, bound=bound, temp=temp,
            slope_bias_target=slope_bias_y, cond_gain_init=cond_gain_y,
            d_rel_extra=0
        )
        # Student-t residual head: delta + log_sigma + nu correction
        # Operates on x_feat, adds to flow base prediction
        self.res_y = ResidualHeadStudentT(in_dim=d_x_cond, out_dim=d_y, hidden=320, dropout_p=0.15)
        # Skip connection: direct linear x -> delta_y (zero-init = starts at zero)
        self.delta_skip = nn.Linear(d_x_cond, d_y)
        nn.init.zeros_(self.delta_skip.weight); nn.init.zeros_(self.delta_skip.bias)

        # Fallback teacher projection (used only if taps/hidden fail)
        # Fallback teacher projection: frozen orthogonal map x -> 512-dim
        # Used only if taps and conditioner_hidden both fail
        self.proj_teacher = nn.Linear(d_x_cond, 512, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.proj_teacher.weight)
        for p in self.proj_teacher.parameters(): p.requires_grad_(False)

        # Teacher whitening stats + source tracker
        self.teacher_mu = None
        self.teacher_std = None
        self.teacher_source_mode = "unknown"  # "taps" | "hidden" | "projection"

        # Store config for serialisation / decoder pack export
        self.config = dict(d_x_cond=d_x_cond, d_y=d_y, d_h=d_h, K_blocks=K_blocks, K_bins=K_bins,
                           bound=bound, temp=temp, slope_bias_y=slope_bias_y, cond_gain_y=cond_gain_y)

    # --- Forward: x_feat -> (y_mean, y_base, delta_y, logsig_y, nu_y) ---
    def forward(self, x_feat, cond_gain_scale: float = 1.0):
        # z=0 is the mode of the standard normal base distribution
        # Inverting at z=0 gives the most likely y for this x
        z0 = torch.zeros(x_feat.size(0), 1, device=x_feat.device, dtype=torch.float32)
        # Flow inverse: z=0 -> y_base (maximum a posteriori prediction)
        # cond_gain_scale amplifies conditioning signal (annealed during training)
        y_base, _ = self.flow_y_given_x.inverse(z0, cond=x_feat * cond_gain_scale, extra=None)
        # Residual head: correction delta + observation noise (log_sigma, nu)
        delta_core, logsig_y, nu_y = self.res_y(x_feat)
        # Total correction: residual head output + skip connection
        delta_y = delta_core + self.delta_skip(x_feat)
        # Final prediction: flow base + correction
        # y_mean is the point prediction used in the Student-t NLL loss
        y_mean = y_base + delta_y
        return y_mean, y_base, delta_y, logsig_y, nu_y

    # -------- Minimal: direct taps from flow helper (with safe fallback to conditioner hidden) --------
    @torch.no_grad()
    # --- Fit whitening stats (mu, std) on teacher embedding ---
    # Called once on training data after Stage 1 is trained.
    # The whitened embedding becomes the distillation target for Stage 2.
    def fit_teacher_whitener(self, x_feat_std, cond_gain_scale: float = 1.0):
        H = None
        # Try sources in priority order: taps > hidden > projection
        self.teacher_source_mode = "projection"  # default until proven otherwise

        # (A) Preferred: taps
        if hasattr(self.flow_y_given_x, "extract_conditioner_taps"):
            # (A) Preferred: conditioner taps = internal activations from all blocks
            # Multi-scale cond_gain=[8, scale, 16] averages across gain strengths
            # z_perturb=0.1 adds small jitter for stability
            H = self.flow_y_given_x.extract_conditioner_taps(
                x_feat_std, cond_gain_scale=[8.0, cond_gain_scale, 16.0], z_perturb=0.1
            )
            if H is not None:
                self.teacher_source_mode = "taps"

        # (B) Fallback: conditioner hidden
        if H is None and hasattr(self.flow_y_given_x, "conditioner_hidden"):
            try:
                h, _ = self.flow_y_given_x.conditioner_hidden(
                    x_feat_std * float(cond_gain_scale), extra=None
                )
                if h is not None:
                    H = h
                    self.teacher_source_mode = "hidden"
                    print("[teacher] fit_teacher_whitener: taps missing → using conditioner hidden.")
            except Exception:
                H = None

        # (C) Last resort: projection (no whitener)
        # (C) Last resort: no teacher embedding available
        # Will fall back to frozen orthogonal projection in get_teacher_embed
        if H is None:
            print("[teacher] fit_teacher_whitener: no taps & no hidden → projection fallback.")
            return False

        # Compute per-dimension mean and std for whitening
        # std floored at 1e-6 to prevent division by zero
        mu = H.mean(dim=0, keepdim=False)
        sd = H.std(dim=0, keepdim=False).clamp_min(1e-6)
        self.teacher_mu = mu
        self.teacher_std = sd
        print(f"[teacher] Fitted whitener over teacher features: D={int(mu.numel())} | source={self.teacher_source_mode}")
        return True

    @torch.no_grad()
    # --- Get whitened teacher embedding for a batch of x_feat ---
    # Uses same priority: taps > hidden > projection.
    # Applies whitening: (H - mu) / std to normalise the embedding.
    def get_teacher_embed_from_xfeat(self, x_feat_std, cond_gain_scale: float = 1.0):
        H = None

        # (A) Preferred: taps
        if hasattr(self.flow_y_given_x, "extract_conditioner_taps"):
            H = self.flow_y_given_x.extract_conditioner_taps(
                x_feat_std, cond_gain_scale=[8.0, cond_gain_scale, 16.0], z_perturb=0.1
            )
            if H is not None:
                if self.teacher_source_mode == "unknown":
                    self.teacher_source_mode = "taps"

        # (B) Fallback: conditioner hidden
        if H is None and hasattr(self.flow_y_given_x, "conditioner_hidden"):
            try:
                h, _ = self.flow_y_given_x.conditioner_hidden(
                    x_feat_std * float(cond_gain_scale), extra=None
                )
                if h is not None:
                    H = h
                    if self.teacher_source_mode == "unknown":
                        self.teacher_source_mode = "hidden"
            except Exception:
                H = None

        # (C) Last resort: orthogonal projection
        # If no taps/hidden or whitener not fitted: use frozen projection
        if H is None or (self.teacher_mu is None) or (self.teacher_std is None):
            return self.proj_teacher(x_feat_std)

        # Apply whitening: centre and scale to unit variance
        # This normalised embedding is the Stage 2 distillation target
        return (H - self.teacher_mu) / self.teacher_std

    # --- Simple projection fallback: x_feat -> 512-dim via frozen orthogonal ---
    def get_teacher_embed(self, x_feat):
        return self.proj_teacher(x_feat)
