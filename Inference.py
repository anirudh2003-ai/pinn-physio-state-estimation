# ===========================================================================
# Standalone inference for fNIRS student encoder
# ===========================================================================
#
# This script runs the trained Stage 2a WaveNet student in CLOSED-LOOP mode
# on unseen data, using ONLY ShortChannel observations (no ground-truth X).
#
# Pipeline:
#   1. Load data + detrend ShortChannel (must match training, Section 2.1)
#   2. Load scalers, encoder payload, student weights from training exports
#   3. Optionally load frozen Stage 1 teacher for innovation feedback (Section 2.6)
#   4. Build y-only observation features (75 dimensions)
#   5. Window features into 128-step sliding windows for WaveNet
#   6. Run closed-loop rollout: each step feeds its own prediction as x_prev
#      to the next step, with stateful bias integration and optional innovation
#   7. Convert predictions from standardised space to physical units
#   8. Compute metrics, save CSV + diagnostic plots
#
# Key architectural details (must match training):
#   - Gate floor 0.65 prevents persistence collapse (Section 5.8)
#   - Kappa mixing blends AR proposal with y-only anchor (Section 2.7)
#   - Bias integrator b(t) tracks slow drift (Section 2.6)
#   - Innovation = y_obs - y_hat(x_prev) via frozen teacher (Section 2.6)
#   - 64-step warmup uses true state to let bias integrator converge
#
# Diagnostics included:
#   DIAG-A: Y-only backbone isolation (is it extracting signal on unseen data?)
#   DIAG-B: Kappa=0 ablation (is y-only mixing helping or hurting?)
#   Horizon R²: error growth analysis at multiple time horizons
#   Zero-centered R²: dynamics-only evaluation ignoring baseline offset
# ===========================================================================
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------- Import the corrected prior_student module ----------
# --- Import student encoder classes and utilities from prior_student.py ---
# prior_student.py is the SINGLE SOURCE OF TRUTH for inference architecture.
# Every class here must match the training code in exp_base_dynamic.py exactly.
from prior_student import (
    YMemoryWaveNet,
    YMemoryMLP,
    LagConditioner,
    WaveNetBlock,
    build_student_from_payload,
    gated_ema_bias_update,
    _ensure_kappa_vec_or_batch,
    _innov_enabled,
    set_diag_flags,
    set_spectral_norm_flag,
    _USE_SPECTRAL_NORM_AR,
)

# ---------- Import shared data builders from your training codebase ----------
# These are needed for: load_folder_and_join, build_x_features_from_df,
# build_y_only_features, apply_scaling_np, invert_scaling_np, standardize_train_stats
from exp_base_static import (
    load_folder_and_join,
    build_x_features_from_df,
    build_y_only_features,
    apply_scaling_np,
    invert_scaling_np,
    standardize_train_stats,
    YGivenXModel,
    reset_actnorm_flags,
)

from scipy.ndimage import uniform_filter1d


# ===========================================================================
# Helpers
# ===========================================================================

# --- R² and MAE computation (matches training metric functions) ---
def _r2_mae_np(y_t, y_p):
    y_t = np.asarray(y_t).ravel()
    y_p = np.asarray(y_p).ravel()
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_t - y_p)))
    return r2, mae


# --- Efficient sliding window construction via torch.unfold ---
# Creates [Nc, WIN, F] windows from [N, F] feature sequence.
# idx_center[i] = index of the last (rightmost) frame in window i.
# This means window i contains frames [idx_center[i]-WIN+1 : idx_center[i]+1].
def _make_windows_unfold(y_only_t: torch.Tensor, win: int):
    """Efficient windowing via unfold. Returns [Nc, WIN, F] and idx_center [Nc]."""
    N = int(y_only_t.size(0))
    if N < win:
        raise ValueError(f"Sequence shorter than WIN: N={N}, WIN={win}")
    y_win = y_only_t.unfold(0, win, 1).permute(0, 2, 1).contiguous()
    idx_center = torch.arange(win - 1, N, dtype=torch.long)
    return y_win, idx_center


def _warmup_y_only(model, device):
    """Warmup shim for YGivenXModel (activates actnorm etc.)."""
    model.eval()
    B = 32
    dx = model.config["d_x_cond"]
    xw = torch.randn(B, dx, device=device, dtype=torch.float32)
    yw = torch.randn(B, 1, device=device, dtype=torch.float32)
    with torch.no_grad():
        zw, _ = model.flow_y_given_x(yw, cond=xw, extra=None)
        _, _ = model.flow_y_given_x.inverse(zw, cond=xw, extra=None)
        _ = model.res_y(xw)


# ===========================================================================
# Main inference rollout
# ===========================================================================

@torch.no_grad()
# ===========================================================================
# Main inference entry point — closed-loop rollout on unseen data
# ===========================================================================
# This function orchestrates the full inference pipeline.
# At each timestep t:
#   1. Feed y-only features through WaveNet backbone -> memory vector z
#   2. Compute observation increment dmu_y from z
#   3. Compute AR increment dmu_AR from x_hat(t-1)
#   4. Update bias integrator: b(t) = EMA(b(t-1), b_hat)
#   5. If innovation ON: innovation = y_obs - y_hat(x_prev) via teacher
#   6. Build proposal: x_prop = x_hat(t-1) + dmu_y + dmu_AR + b(t) + dx_innov
#   7. Kappa mix: x_prop = (1-k)*x_prop + k*x_hat_yonly
#   8. Gate: x_hat(t) = x_hat(t-1) + g*(x_prop - x_hat(t-1))
#   9. x_hat(t) becomes x_prev for step t+1 (closed-loop)
def run_rollout_inference(
    *,
    folder: str,
    encoder_path: str,
    scaler_path: str,
    ygx_ckpt_path: str | None = None,    # Only needed if use_innovation=True
    out_dir: str = "./output/inference_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_innovation: bool = False,
    cond_gain_for_teacher: float = 20.0,
    chunk_size: int = 1024,
    max_plot_points: int = 3000,
):
    """
    Full inference pipeline:
      1. Load data (only ShortChannel is ground truth)
      2. Load scalers, encoder payload, student weights
      3. Optionally load teacher (YGivenX) for innovation
      4. Build y-only features, windows, lag tensors
      5. Run closed-loop rollout (stateful b_prev, honest lag buffer)
      6. Convert predictions to raw units
      7. Print metrics + save plots
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"[INFERENCE] Starting rollout inference")
    print(f"  folder:          {folder}")
    print(f"  encoder_path:    {encoder_path}")
    print(f"  scaler_path:     {scaler_path}")
    print(f"  ygx_ckpt_path:   {ygx_ckpt_path}")
    print(f"  use_innovation:  {use_innovation}")
    print(f"  device:          {device}")
    print(f"  out_dir:         {out_dir}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 1: Load raw fNIRS data and detrend ShortChannel
        # ==================================================================
        # CRITICAL: detrending must match training exactly (Section 2.1).
        # Training uses uniform_filter1d with size=600 (~100s at 6 Hz).
        # Mismatched detrending causes baseline offset in predictions.
    # 1. Load raw data
    # ------------------------------------------------------------------
    df = load_folder_and_join(folder)
    print(f"[DATA] Loaded {len(df)} samples from {folder}")

    # CRITICAL: detrend ShortChannel exactly as training
    y_raw = df["ShortChannel"].to_numpy(np.float32)
    y_smooth = uniform_filter1d(y_raw, size=600)
    df["ShortChannel"] = y_raw - y_smooth
    print(f"[DATA] Applied uniform_filter1d detrending (size=600)")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 2: Load standardisation scalers from Stage 1 training
        # ==================================================================
        # Contains: x_norm_mu/sigma (raw->normalised), x_feat_mu/sigma
        # (normalised->standardised), y_mu/sigma (y standardisation).
    # 2. Load scalers
    # ------------------------------------------------------------------
    with open(scaler_path, "r") as f:
        sc = json.load(f)

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 3: Load encoder payload (student weights + full config)
        # ==================================================================
        # The payload contains everything needed to reconstruct the student:
        #   - student_state: state_dict weights
        #   - Architecture params: d_mem, hidden, layers, wavenet_win
        #   - Training config: clamp_x_rollout, kappa_infer, lags
        #   - Feature scalers: y_only_mu/sigma
        #   - Flags: innov_disabled, ar_disabled, spectral_norm_ar
    # 3. Load encoder payload (contains student weights + all config)
    # ------------------------------------------------------------------
    payload = torch.load(encoder_path, map_location="cpu")
    print(f"[PAYLOAD] student_type={payload.get('student_type')} | "
          f"lags={payload.get('lags')} | d_mem={payload.get('d_mem')} | "
          f"hidden={payload.get('hidden')} | layers={payload.get('layers')}")

    # Read inference config from payload
    clamp_val = float(payload.get("clamp_x_rollout", 0.0))
    clamp_val = None if clamp_val <= 0 else clamp_val
    kappa_corr = float(payload.get("kappa_infer", 0.35))

    # Override innovation from payload if user says off
    innov_disabled = bool(payload.get("innov_disabled", False))
    if not use_innovation:
        innov_disabled = True  # force off
    innov_on = _innov_enabled(
        use_innovation_feedback=bool(payload.get("use_innovation_feedback", True)),
        innov_disabled=innov_disabled,
    )
    print(f"[CONFIG] clamp_val={clamp_val} | kappa_corr={kappa_corr} | innov_on={innov_on}")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 4: Build standardised feature arrays
        # ==================================================================
        # x_feat_std [N, 4]: base cardiovascular state in standardised space
        #   (HR, RR, CO2, Other — only used for metrics, NOT fed to student)
        # y_std [N, 1]: standardised ShortChannel (for innovation computation)
        # y_only_std [N, 75]: 75 observation features from ShortChannel alone
        #   (bandpass energies, envelopes, derivatives, Kalman features, etc.)
        #   These are the ONLY input to the student at inference time.
    # 4. Build feature arrays
    # ------------------------------------------------------------------
    x_norm_mu = np.array(sc["x_norm_mu"], dtype=np.float32)
    x_norm_sd = np.array(sc["x_norm_sigma"], dtype=np.float32)

    _, x_feat, feat_names, _ = build_x_features_from_df(
        df, x_norm_mu=x_norm_mu, x_norm_sd=x_norm_sd, fit_slice=None, save_stats_path=None
    )

    # x_feat_std [N, 4] — base state features in standardized space
    x_feat_std = apply_scaling_np(
        x_feat,
        np.array(sc["x_feat_mu"], dtype=np.float32)[:x_feat.shape[1]],
        np.array(sc["x_feat_sigma"], dtype=np.float32)[:x_feat.shape[1]],
    ).astype(np.float32)

    # y_std [N, 1]
    y_scalar = df["ShortChannel"].to_numpy(np.float32).reshape(-1, 1)
    y_mu_sc = np.array(sc["y_mu"], dtype=np.float32)
    y_sd_sc = np.array(sc["y_sigma"], dtype=np.float32)
    y_std = apply_scaling_np(y_scalar, y_mu_sc, y_sd_sc).astype(np.float32)

    # y_only features (using encoder payload stats)
    est_fs_infer = float(payload.get("est_fs", 30.0))
    y_only, yonly_names = build_y_only_features(df, fs_hint=est_fs_infer)
    y_only_mu = np.array(payload["y_only_mu"], dtype=np.float32)
    y_only_sd = np.array(payload["y_only_sigma"], dtype=np.float32)
    y_only_std = apply_scaling_np(y_only, y_only_mu, y_only_sd).astype(np.float32)

        # --- Recenter y_only features to zero mean (domain adaptation) ---
        # The student was trained on data with near-zero mean features.
        # Unseen data may have a different baseline, causing offset in predictions.
        # Subtracting the segment mean removes this baseline shift.
    # === RECENTER y_only inputs to training distribution (Mean ONLY) ===
    y_only_segment_mu = y_only_std.mean(axis=0, keepdims=True)
    y_only_std = y_only_std - y_only_segment_mu
    print(f"[RECENTER-INPUT] Shifted y_only features by segment mean "
          f"(norm={np.linalg.norm(y_only_segment_mu):.4f})")

    d_yfeat = int(y_only_std.shape[1])
    print(f"[FEATURES] y_only_std: {y_only_std.shape} | x_feat_std: {x_feat_std.shape}")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 5: Reconstruct Stage 2a WaveNet student from saved payload
        # ==================================================================
        # build_student_from_payload() sets spectral norm and diagnostic flags
        # BEFORE construction, ensuring state_dict keys match exactly.
    # 5. Build student model
    # ------------------------------------------------------------------
    student = build_student_from_payload(payload, d_yfeat=d_yfeat, device=device)

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 6: Load frozen Stage 1 teacher (only if innovation is ON)
        # ==================================================================
        # The teacher P(y|x) provides the innovation signal:
        #   innovation(t) = y_obs(t) - y_hat(x_hat(t-1))
        # If the student's state estimate is wrong, the teacher's predicted
        # observation will disagree with what was observed (Section 2.6).
        # D_COND = 10 (4 physiology + 6 Y-bandpass features).
    # 6. Build teacher model (only if innovation is ON)
    # ------------------------------------------------------------------
    ygx = None
    D_COND = len(sc["x_feat_mu"])  # 16 when teacher has lags

    if innov_on:
        assert ygx_ckpt_path is not None, "ygx_ckpt_path required when innovation is ON"
        print(f"[TEACHER] Loading YGivenX from {ygx_ckpt_path} (d_x_cond={D_COND})")

        ygx = YGivenXModel(d_x_cond=D_COND, cond_gain_y=cond_gain_for_teacher).to(device)
        _warmup_y_only(ygx, device)
        state = torch.load(ygx_ckpt_path, map_location=device)
        target_sd = ygx.state_dict()
        filtered = {k: v for k, v in state.items() if (k in target_sd and v.shape == target_sd[k].shape)}
        ygx.load_state_dict(filtered, strict=False)
        reset_actnorm_flags(ygx)
        ygx.eval()
        print(f"[TEACHER] Loaded successfully")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 7: Build 10-dim teacher conditioning array (for innovation)
        # ==================================================================
        # The teacher was trained on 10-dim input: [4 base X, 6 Y-bandpass].
        # At inference, the 4 base X dims come from the student's predictions,
        # and the 6 Y-bandpass features are recomputed from ShortChannel.
        # These must use the EXACT same bandpass extraction as training.
    # 7. Build full teacher-dim conditioning array (for innovation)
    # ------------------------------------------------------------------
    X_cond_cent_cpu = None
    x_feat_std_full = None

    if innov_on:
        from exp_base_static import _causal_bandpass_energy, _causal_envelope, _causal_bandpass_signal

        y_sc = df["ShortChannel"].to_numpy(np.float32)
        nyq = est_fs_infer / 2.0

        bp_hr_energy = _causal_bandpass_energy(y_sc, est_fs_infer, 0.5, min(2.5, nyq*0.95), energy_window=16)
        bp_hr_env    = _causal_envelope(y_sc, est_fs_infer, 0.5, min(2.5, nyq*0.95))
        bp_rr_energy = _causal_bandpass_energy(y_sc, est_fs_infer, 0.08, min(0.5, nyq*0.95), energy_window=32)
        bp_rr_env    = _causal_envelope(y_sc, est_fs_infer, 0.08, min(0.5, nyq*0.95))
        bp_trend     = _causal_bandpass_energy(y_sc, est_fs_infer, 0.01, min(0.1, nyq*0.95), energy_window=64)
        bp_cardiac   = _causal_bandpass_signal(y_sc, est_fs_infer, 0.5, min(2.5, nyq*0.95))

        y_derived = np.stack([bp_hr_energy, bp_hr_env, bp_rr_energy, bp_rr_env, bp_trend, bp_cardiac], axis=1)
        x_feat_full_raw = np.concatenate([x_feat, y_derived], axis=1)

        assert x_feat_full_raw.shape[1] == D_COND, \
            f"Teacher expects {D_COND} dims but got {x_feat_full_raw.shape[1]}"

        x_feat_std_full = apply_scaling_np(
            x_feat_full_raw,
            np.array(sc["x_feat_mu"], dtype=np.float32),
            np.array(sc["x_feat_sigma"], dtype=np.float32)
        ).astype(np.float32)

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 8: Build sliding windows and lag tensors
        # ==================================================================
        # WaveNet requires [B, WIN, F] input (128-step windows of 75 features).
        # idx_center[i] = rightmost frame index in window i.
        # If lag conditioning is configured, windows before max_lag are excluded
        # (they can't look back far enough for the earliest lag).
    # 8. Windowing + lag tensor construction
    # ------------------------------------------------------------------
    student_type = payload.get("student_type", "wavenet")
    cond_lags = list(sorted(payload.get("lags", [])))

    if student_type == "wavenet":
        WIN = int(payload["wavenet_win"])

        y_only_t = torch.from_numpy(y_only_std).float()
        y_win, idx_center = _make_windows_unfold(y_only_t, WIN)

        # Filter by max_lag (Bug 5 fix)
        if len(cond_lags) > 0:
            max_lag_export = max(cond_lags)
            valid = (idx_center >= max_lag_export)
            y_win = y_win[valid]
            idx_center = idx_center[valid]

        y_in_cpu = y_win
        print(f"[WINDOWS] WIN={WIN} | Nc={y_in_cpu.size(0)} | max_lag={max(cond_lags) if cond_lags else 0}")
    else:
        idx_center = torch.arange(len(df), dtype=torch.long)
        y_in_cpu = torch.from_numpy(y_only_std).float()

    Nc = int(y_in_cpu.size(0))

    # Aligned y_std
    y_std_cent_cpu = torch.from_numpy(y_std).float()[idx_center]

    # Aligned X_true (for metrics — we DO have it since we loaded the full data)
    X_true_cent_cpu = torch.from_numpy(x_feat_std).float()[idx_center]


    # Build aligned conditioning for innovation
    if innov_on:
        X_cond_full_cpu = torch.from_numpy(x_feat_std_full).float()
        X_cond_cent_cpu = X_cond_full_cpu[idx_center]
        print(f"[COND] X_cond_cent_cpu: {X_cond_cent_cpu.shape}")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 9: Choose initial state x0 for rollout
        # ==================================================================
        # The first prediction needs an x_prev to start from.
        # Using the mean of X_prev (standardised) as x0 centres the rollout
        # near the training distribution's mean state.
        # The 64-step warmup (step 10) uses true state to refine from here.
    # 9. Seed state (x0)
    # ------------------------------------------------------------------
    # Use mean of first few prev states as seed
    X_prev_cent_cpu = torch.from_numpy(x_feat_std).float()
    X_prev_full = torch.zeros_like(X_prev_cent_cpu)
    X_prev_full[0] = X_prev_cent_cpu[0]
    X_prev_full[1:] = X_prev_cent_cpu[:-1]
    X_prev_aligned = X_prev_full[idx_center]
    x0_cpu = X_prev_aligned.mean(dim=0)
    print(f"[SEED] x0 (mean prev): {x0_cpu.numpy()}")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 10: Run closed-loop rollout via rollout_from_y()
        # ==================================================================
        # This is the core inference loop (see prior_student.py).
        # Per step: clamp prev -> compute innovation -> y-only proposal ->
        #   main forward with AR + bias + innovation + kappa -> gate -> output
        # First 64 steps use true state (warmup) to let bias integrator converge.
        # After warmup, each step feeds its own prediction as x_prev (closed-loop).
        #
        # Returns:
        #   mu_roll [Nc, 4]: predicted state trajectory
        #   b_roll [Nc, 4]: bias state trajectory
        #   c_roll [Nc, 4]: bias increment trajectory
        #   prevfed [Nc, 4]: clamped x_prev that was actually fed
        #   prev_unclamped [Nc, 4]: raw x_prev before safety clamping
    # 10. RUN ROLLOUT
    # ------------------------------------------------------------------
    print(f"\n[ROLLOUT] Starting closed-loop rollout over {Nc} steps...")
    t0 = time.time()

    from prior_student import rollout_from_y

    print(f"[DEBUG] student type: {type(student).__name__}")
    print(f"[DEBUG] student.lags: {student.lags if hasattr(student, 'lags') else 'N/A'}")
    print(f"[DEBUG] kappa_corr={kappa_corr} | kappa_vec={student.kappa_vec().cpu().detach().numpy()}")
    print(f"[DEBUG] compound kappa_eff = {(kappa_corr * student.kappa_vec()).cpu().detach().numpy()}")
    g_floor_val = student.g_floor.item() if hasattr(student, 'g_floor') else 0.50
    print(f"[DEBUG] g_floor={g_floor_val:.2f} | gate.bias={student.gate.bias.data.cpu().numpy()}")
    print(f"[DEBUG] innov_on={innov_on} | clamp_val={clamp_val}")
    print(f"[DEBUG] x0_cpu = {x0_cpu.numpy()}")
    print(f"[DEBUG] y_in_cpu shape={y_in_cpu.shape} | y_std shape={y_std_cent_cpu.shape}")
    print(f"[DEBUG] AR weights norm: {sum(p.abs().sum().item() for p in student.ar.parameters()):.4f}")
    print(f"[DEBUG] head_dx_y weights norm: {student.head_dx_y.weight.abs().sum().item():.4f}")

    print()

    # ============================================================
        # ==================================================================
        # DIAGNOSTIC A: Y-only backbone isolation test
        # ==================================================================
        # Runs the student with x_prev=None at every step (no AR, no bias, no
        # innovation). Tests whether the Stage 2b y-only backbone extracts
        # any useful signal from observations alone on unseen data.
        # If zero-centered R² is positive, the backbone sees real physiology.
        # Cross-correlation analysis reveals if predictions are time-lagged.
    # DIAG-A: Y-only isolation on unseen segment
    # ============================================================
    print("\n[DIAG-A] Y-only backbone isolation test on unseen data...")
    import time as _dtime
    _dt0 = _dtime.time()
    _xy_all = []
    student.eval()
    with torch.no_grad():
        for _t in range(Nc):
            _yt = y_in_cpu[_t:_t+1].to(device=device, dtype=torch.float32)
            # NOTE: lag_values/lag_mask=None — no lag conditioning in y-only path
            _lv = None
            _lm = None
            _out = student(_yt, x_prev=None, b_prev=None, sample=False,
                           innov_prev=None, lag_values=_lv, lag_mask=_lm)
            _xy_all.append(_out[3].detach().cpu().numpy())
    _xy_all = np.array(_xy_all).squeeze()
    _Xt = X_true_cent_cpu.numpy() if hasattr(X_true_cent_cpu, 'numpy') else np.array(X_true_cent_cpu)

    _xy_zc = _xy_all - _xy_all.mean(axis=0, keepdims=True)
    _xt_zc = _Xt - _Xt.mean(axis=0, keepdims=True)
    print(f"[DIAG-A] Done in {_dtime.time()-_dt0:.1f}s")
    print(f"[DIAG-A] xy mean={_xy_all.mean(axis=0)} std={_xy_all.std(axis=0)}")
    print(f"[DIAG-A] Xt mean={_Xt.mean(axis=0)} std={_Xt.std(axis=0)}")

    _dim_names = ["DBP", "SBP", "CO", "SV"]
    for _j, _nm in enumerate(_dim_names):
        _corr = np.corrcoef(_xy_all[:, _j], _Xt[:, _j])[0, 1]
        _ss_r = np.sum((_xy_zc[:, _j] - _xt_zc[:, _j])**2)
        _ss_t = np.sum(_xt_zc[:, _j]**2) + 1e-12
        _r2 = 1.0 - _ss_r / _ss_t
        _vr = _xy_all[:, _j].std() / (_Xt[:, _j].std() + 1e-12)
        print(f"[DIAG-A] {_nm:>6s}: ZC_R²={_r2:+.4f} corr={_corr:+.4f} var_ratio={_vr:.3f}")

    _ss_r_all = np.sum((_xy_zc - _xt_zc)**2)
    _ss_t_all = np.sum(_xt_zc**2) + 1e-12
    print(f"[DIAG-A] OVERALL ZC_R²={1.0 - _ss_r_all/_ss_t_all:+.4f}")
    print("[DIAG-XCORR] Cross-correlation at different lags:")
    for _j, _nm in enumerate(["DBP", "SBP", "CO", "SV"]):
        _xy_j = (_xy_all[:, _j] - _xy_all[:, _j].mean())
        _xt_j = (_Xt[:, _j] - _Xt[:, _j].mean())
        _xy_j = _xy_j / (_xy_j.std() + 1e-12)
        _xt_j = _xt_j / (_xt_j.std() + 1e-12)

        max_lag = 300
        _xcorr = np.correlate(_xt_j, _xy_j, mode='full')
        _xcorr = _xcorr / len(_xt_j)
        _mid = len(_xcorr) // 2
        _xcorr_window = _xcorr[_mid - max_lag : _mid + max_lag + 1]
        _lags = np.arange(-max_lag, max_lag + 1)

        _best_idx = np.argmax(np.abs(_xcorr_window))
        _best_lag = _lags[_best_idx]
        _best_corr = _xcorr_window[_best_idx]
        _corr_at_zero = _xcorr_window[max_lag]

        print(f"[DIAG-XCORR] {_nm:>6s}: corr@lag0={_corr_at_zero:+.4f} | "
              f"best_corr={_best_corr:+.4f} @ lag={_best_lag} ({_best_lag/6:.1f}s)")
    print()

    # ============================================================
        # ==================================================================
        # DIAGNOSTIC B: Kappa=0 ablation (pure AR + bias, no y-only mixing)
        # ==================================================================
        # Runs full rollout but with kappa=0 (no y-only mixing).
        # Comparison with main rollout reveals whether kappa mixing helps:
        #   If DIAG-B R² > main R²: y-only is HURTING on unseen data
        #   If DIAG-B R² < main R²: y-only IS helping despite low solo R²
    # DIAG-B: Kappa=0 ablation (quick rollout, pure AR+bias, no y-only)
    # ============================================================
    print("[DIAG-B] Running kappa=0 ablation (no y-only mixing)...")
    _dt0 = _dtime.time()
    _mu_k0, _b_k0, _c_k0, _pf_k0, _pu_k0 = rollout_from_y(
        student=student,
        y_in_cpu=y_in_cpu,
        y_std_cpu=y_std_cent_cpu,
        x0_cpu=x0_cpu,
        clamp_val=clamp_val,
        kappa_corr=0.0,
        use_innovation=innov_on,
        ygx_model=ygx,
        cond_gain_for_teacher=cond_gain_for_teacher,
        X_cond_cent_cpu=X_cond_cent_cpu,
        d_state=4,
        device=device,
        chunk_size=chunk_size,
        use_amp=True,
        warmup_steps=64,
        X_true_cpu=X_true_cent_cpu,
    )
    _mu_k0_np = _mu_k0.numpy()
    _mu_k0_zc = _mu_k0_np - _mu_k0_np.mean(axis=0, keepdims=True)
    print(f"[DIAG-B] Done in {_dtime.time()-_dt0:.1f}s")
    print(f"[DIAG-B] mu_k0 mean={_mu_k0_np.mean(axis=0)} std={_mu_k0_np.std(axis=0)}")

    for _j, _nm in enumerate(_dim_names):
        _ss_r = np.sum((_mu_k0_zc[:, _j] - _xt_zc[:, _j])**2)
        _ss_t = np.sum(_xt_zc[:, _j]**2) + 1e-12
        _r2 = 1.0 - _ss_r / _ss_t
        print(f"[DIAG-B] {_nm:>6s}: ZC_R²={_r2:+.4f} var_ratio={_mu_k0_np[:, _j].std()/(_Xt[:, _j].std()+1e-12):.3f}")

    _ss_r_all = np.sum((_mu_k0_zc - _xt_zc)**2)
    _ss_t_all = np.sum(_xt_zc**2) + 1e-12
    print(f"[DIAG-B] OVERALL ZC_R²={1.0 - _ss_r_all/_ss_t_all:+.4f}")
    print(f"[DIAG-B] If this is BETTER than main rollout, y-only is hurting on unseen data.")
    print(f"[DIAG-B] If this is WORSE, y-only IS helping despite low R².")
    print()

    # --- YOUR EXISTING ROLLOUT CALL REMAINS HERE ---
    mu_roll, b_roll, c_roll, prevfed_roll, prev_unclamped_roll = rollout_from_y(
        student=student,
        y_in_cpu=y_in_cpu,
        y_std_cpu=y_std_cent_cpu,
        x0_cpu=x0_cpu,
        clamp_val=clamp_val,
        kappa_corr=kappa_corr,
        use_innovation=innov_on,
        ygx_model=ygx,
        cond_gain_for_teacher=cond_gain_for_teacher,
        X_cond_cent_cpu=X_cond_cent_cpu,
        d_state=4,
        device=device,
        chunk_size=chunk_size,
        use_amp=True,
        progress_every=1024,
        warmup_steps=64,
        X_true_cpu=X_true_cent_cpu,
    )

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
        # ==================================================================
        # Post-rollout diagnostics: check for divergence and drift
        # ==================================================================
        # Inspects: first 10 steps (immediate divergence), clamp saturation,
        # running R² at multiple horizons (error growth), state distribution
        # (mean/std/min/max), bias magnitude, and prevfed saturation.
    # Post-rollout step-by-step diagnostics
    # ------------------------------------------------------------------
    mu_roll_np = mu_roll.numpy()
    X_true_np = X_true_cent_cpu.numpy()

    print(f"\n{'='*60}")
    print(f"[POST-ROLLOUT DIAGNOSTICS]")
    print(f"{'='*60}")

    # Check first 10 steps for immediate divergence
    print(f"\n[FIRST 10 STEPS]")
    for t in range(min(10, Nc)):
        err = np.abs(mu_roll_np[t] - X_true_np[t])
        print(f"  t={t:4d} | mu={mu_roll_np[t]} | true={X_true_np[t]} | "
              f"|err|={err} | |b|={b_roll[t].abs().mean().item():.5f} | "
              f"prevfed={prevfed_roll[t].numpy()}")

    # Check if hitting clamp
    if clamp_val is not None:
        clamp_hits = (mu_roll.abs() >= (clamp_val - 0.01)).float().mean(dim=0).numpy()
        print(f"\n[CLAMP HITS] fraction at ±{clamp_val}: {clamp_hits}")
        print(f"  per-dim: DBP={clamp_hits[0]:.3f} SBP={clamp_hits[1]:.3f} CO={clamp_hits[2]:.3f} SV={clamp_hits[3]:.3f}")


    # Running R² at different horizons
    print(f"\n[RUNNING R² BY HORIZON]")
    checkpoints = [10, 50, 100, 200, 400, 800, 1600, 3200, 6400, Nc]
    for H in checkpoints:
        H = min(H, Nc)
        if H < 2:
            continue
        r2_h, mae_h = _r2_mae_np(X_true_np[:H], mu_roll_np[:H])
        # Also check if mean is drifting
        pred_mean = mu_roll_np[:H].mean(axis=0)
        true_mean = X_true_np[:H].mean(axis=0)
        drift = pred_mean - true_mean
        print(f"  H={H:>6d} | R²={r2_h:+8.4f} | MAE={mae_h:.4f} | "
              f"pred_mean={pred_mean} | drift={drift}")

    # State distribution summary
    print(f"\n[STATE DISTRIBUTION]")
    print(f"  mu_roll  | mean={mu_roll_np.mean(axis=0)} | std={mu_roll_np.std(axis=0)} | "
          f"min={mu_roll_np.min(axis=0)} | max={mu_roll_np.max(axis=0)}")
    print(f"  X_true   | mean={X_true_np.mean(axis=0)} | std={X_true_np.std(axis=0)} | "
          f"min={X_true_np.min(axis=0)} | max={X_true_np.max(axis=0)}")

    # Bias diagnostics
    b_np = b_roll.numpy()
    c_np = c_roll.numpy()
    print(f"\n[BIAS DIAGNOSTICS]")
    print(f"  |b| mean={np.abs(b_np).mean(axis=0)} | |c| mean={np.abs(c_np).mean(axis=0)}")
    print(f"  b final 100 steps mean={b_np[-100:].mean(axis=0)}")

    # Prevfed diagnostics (what was actually fed as x_prev)
    pf_np = prevfed_roll.numpy()
    print(f"\n[PREVFED DIAGNOSTICS]")
    print(f"  mean={pf_np.mean(axis=0)} | std={pf_np.std(axis=0)}")
    print(f"  Are prevfed values saturated at clamp? "
          f"{(np.abs(pf_np) >= (clamp_val - 0.01)).mean(axis=0) if clamp_val else 'no clamp'}")

    print(f"\n[TIMING] {Nc} steps in {elapsed:.1f}s ({Nc/elapsed:.0f} it/s)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 11: Compute evaluation metrics
        # ==================================================================
        # Two evaluation modes:
        #   1. Raw R²/MAE: direct comparison (affected by baseline offset)
        #   2. Zero-centered R²: subtract segment means from both pred and true,
        #      measuring dynamics-only accuracy (ignores baseline mismatch)
        # Per-dimension breakdown: HR, RR, CO2, Other
        # Horizon analysis: head/tail R² at [32, 128, 512, 2048, Nc] steps
    # 11. Compute metrics
    # ------------------------------------------------------------------

    r2_full, mae_full = _r2_mae_np(X_true_np, mu_roll_np)
    print(f"\n{'='*50}")
    print(f"[METRICS] FULL ROLLOUT: R²={r2_full:.4f} | MAE={mae_full:.4f}")

    # === ZERO-CENTER EVALUATION (measures dynamics, ignores baseline) ===
    pred_mu = mu_roll_np.mean(axis=0, keepdims=True)
    true_mu = X_true_np.mean(axis=0, keepdims=True)
    mu_roll_eval = mu_roll_np - pred_mu
    X_true_eval = X_true_np - true_mu

    print(f"\n[EVAL] Pred mean (removed): {pred_mu.squeeze()}")
    print(f"[EVAL] True mean (removed): {true_mu.squeeze()}")

    # Per-dim R² (Zero-Centered)
    dim_names = ["DBP", "SBP", "CO", "SV"]
    ss_res_total = 0.0
    ss_tot_total = 0.0
    print(f"\n[METRICS] ZERO-CENTERED (Dynamics Only):")
    for j, nm in enumerate(dim_names):
        ss_res_j = np.sum((X_true_eval[:, j] - mu_roll_eval[:, j]) ** 2)
        ss_tot_j = np.sum((X_true_eval[:, j] - X_true_eval[:, j].mean()) ** 2) + 1e-12
        r2_j = 1.0 - ss_res_j / ss_tot_j
        mae_j = np.mean(np.abs(X_true_eval[:, j] - mu_roll_eval[:, j]))
        print(f"  {nm:>6s}: R²={r2_j:+.4f} | MAE={mae_j:.4f}")
        ss_res_total += ss_res_j
        ss_tot_total += ss_tot_j

    r2_overall = 1.0 - ss_res_total / (ss_tot_total + 1e-12)
    mae_overall = np.mean(np.abs(X_true_eval - mu_roll_eval))
    print(f"  OVERALL: R²={r2_overall:+.4f} | MAE={mae_overall:.4f}")
    print(f"{'='*50}")

    # Horizon metrics
    horizons = [32, 128, 512, 2048, Nc]
    print(f"\n[HORIZON METRICS]")
    for H in horizons:
        H = min(H, Nc)
        if H < 2:
            continue
        r2_h, mae_h = _r2_mae_np(X_true_np[:H], mu_roll_np[:H])
        r2_t, mae_t = _r2_mae_np(X_true_np[-H:], mu_roll_np[-H:])
        print(f"  H={H:6d} | head: R²={r2_h:+.4f} MAE={mae_h:.4f} | tail: R²={r2_t:+.4f} MAE={mae_t:.4f}")

    # Bias/correction stats
    print(f"\n[BIAS STATS]")
    print(f"  mean|b|={b_roll.abs().mean().item():.5f} | mean|c|={c_roll.abs().mean().item():.5f}")
    print(f"  prevfed mean={prevfed_roll.mean(dim=0).numpy()} | std={prevfed_roll.std(dim=0).numpy()}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 12: Convert predictions from standardised to physical units
        # ==================================================================
        # Two-stage inverse transform:
        #   1. Invert x_feat standardisation: x_feat = x_std * sigma + mu
        #   2. Invert x_norm normalisation: x_raw = x_feat * norm_sigma + norm_mu
        # Result: predictions in original units (mmHg, breaths/min, etc.)
    # 12. Convert to raw units for plotting
    # ------------------------------------------------------------------
    x_feat_mu_4 = np.array(sc["x_feat_mu"], dtype=np.float32)[:4]
    x_feat_sd_4 = np.array(sc["x_feat_sigma"], dtype=np.float32)[:4]

    mu_roll_raw = invert_scaling_np(mu_roll_np, x_feat_mu_4, x_feat_sd_4)
    X_true_raw = invert_scaling_np(X_true_np, x_feat_mu_4, x_feat_sd_4)

    # Further convert from x_feat space to raw units
    x_norm_mu_4 = np.array(sc["x_norm_mu"], dtype=np.float32)[:4]
    x_norm_sd_4 = np.array(sc["x_norm_sigma"], dtype=np.float32)[:4]
    mu_roll_phys = mu_roll_raw * x_norm_sd_4 + x_norm_mu_4
    X_true_phys = X_true_raw * x_norm_sd_4 + x_norm_mu_4

    # ------------------------------------------------------------------
    # 13. Time axis
    # ------------------------------------------------------------------
    if "Time" in df.columns:
        times_full = df["Time"].to_numpy()
    else:
        times_full = np.arange(len(df), dtype=np.float64)
    times = times_full[idx_center.numpy()]

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 14: Save predictions to CSV
        # ==================================================================
        # Columns: Time, X_true/pred/res per dim (std + raw), b/c magnitudes
    # 14. SAVE CSV
    # ------------------------------------------------------------------
    import pandas as pd

    df_out = pd.DataFrame({"Time": times})
    for j, name in enumerate(dim_names):
        df_out[f"X_true_{name}"] = X_true_np[:, j]
        df_out[f"X_pred_{name}"] = mu_roll_np[:, j]
        df_out[f"X_res_{name}"] = mu_roll_np[:, j] - X_true_np[:, j]
        df_out[f"X_true_raw_{name}"] = X_true_phys[:, j]
        df_out[f"X_pred_raw_{name}"] = mu_roll_phys[:, j]
    df_out[f"b_abs_mean"] = b_roll.abs().mean(dim=1).numpy()
    df_out[f"c_abs_mean"] = c_roll.abs().mean(dim=1).numpy()

    csv_path = os.path.join(out_dir, "rollout_predictions.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[SAVE] CSV -> {csv_path}")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 15: Generate diagnostic plots
        # ==================================================================
        # Plot 1: Per-dim overlay in standardised space (direct comparison)
        # Plot 2: Per-dim overlay in physical units (interpretable)
        # Plot 3: Error growth over time (smoothed MAE vs time)
        # Plot 4: Bias state + correction increment magnitude vs time
        # Plot 5: Residual distributions (should be near-zero-mean, thin tails)
    # 15. PLOTS
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Downsample for plotting
        stride = max(1, int(math.ceil(Nc / float(max_plot_points))))
        ts = times[::stride]
        Xt = X_true_np[::stride]
        Xp = mu_roll_np[::stride]
        Xt_raw = X_true_phys[::stride]
        Xp_raw = mu_roll_phys[::stride]

        # --- Plot 1: Per-dim overlay (std-space) ---
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        for j, name in enumerate(dim_names):
            ax = axes[j]
            ax.plot(ts, Xt[:, j], label="True", alpha=0.8, linewidth=0.8)
            ax.plot(ts, Xp[:, j], label="Rollout", alpha=0.8, linewidth=0.8)
            r2_j, mae_j = _r2_mae_np(X_true_np[:, j], mu_roll_np[:, j])
            ax.set_ylabel(f"{name} (std)")
            ax.set_title(f"{name} | R²={r2_j:+.4f} MAE={mae_j:.4f}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Rollout vs Truth (std-space) | Overall R²={r2_full:.4f} | innov={'ON' if innov_on else 'OFF'}", fontsize=13)
        fig.tight_layout()
        path1 = os.path.join(out_dir, "rollout_std_space.png")
        fig.savefig(path1, dpi=160)
        plt.close(fig)
        print(f"[PLOT] {path1}")

        # --- Plot 2: Per-dim overlay (raw/physical units) ---
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        raw_names = ["DiastolicBP", "SystolicBP", "CardiacOutput", "StrokeVolume"]
        for j, name in enumerate(raw_names):
            ax = axes[j]
            ax.plot(ts, Xt_raw[:, j], label="True", alpha=0.8, linewidth=0.8)
            ax.plot(ts, Xp_raw[:, j], label="Rollout", alpha=0.8, linewidth=0.8)
            r2_j, _ = _r2_mae_np(X_true_phys[:, j], mu_roll_phys[:, j])
            ax.set_ylabel(name)
            ax.set_title(f"{name} (raw) | R²={r2_j:+.4f}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"Rollout vs Truth (physical units) | innov={'ON' if innov_on else 'OFF'}", fontsize=13)
        fig.tight_layout()
        path2 = os.path.join(out_dir, "rollout_raw_units.png")
        fig.savefig(path2, dpi=160)
        plt.close(fig)
        print(f"[PLOT] {path2}")

        # --- Plot 3: Error growth over time ---
        err_per_step = np.abs(mu_roll_np - X_true_np).mean(axis=1)  # [Nc]
        # Smooth with running mean for readability
        kernel = min(200, Nc // 10)
        if kernel > 1:
            err_smooth = np.convolve(err_per_step, np.ones(kernel)/kernel, mode='valid')
            t_smooth = times[:len(err_smooth)]
        else:
            err_smooth = err_per_step
            t_smooth = times

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t_smooth, err_smooth, linewidth=0.8, color='red')
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean |error| (std-space)")
        ax.set_title(f"Rollout Error Growth | innov={'ON' if innov_on else 'OFF'}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path3 = os.path.join(out_dir, "rollout_error_growth.png")
        fig.savefig(path3, dpi=160)
        plt.close(fig)
        print(f"[PLOT] {path3}")

        # --- Plot 4: Bias state magnitude over time ---
        b_mag = b_roll.abs().mean(dim=1).numpy()
        c_mag = c_roll.abs().mean(dim=1).numpy()

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes[0].plot(times[::stride], b_mag[::stride], linewidth=0.6, color='blue')
        axes[0].set_ylabel("mean|b_t|")
        axes[0].set_title("Bias state magnitude")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times[::stride], c_mag[::stride], linewidth=0.6, color='green')
        axes[1].set_ylabel("mean|c_t|")
        axes[1].set_title("Correction increment magnitude")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Bias/Correction Diagnostics | innov={'ON' if innov_on else 'OFF'}", fontsize=13)
        fig.tight_layout()
        path4 = os.path.join(out_dir, "rollout_bias_diagnostics.png")
        fig.savefig(path4, dpi=160)
        plt.close(fig)
        print(f"[PLOT] {path4}")

        # --- Plot 5: Residual distribution (histogram) ---
        residuals = mu_roll_np - X_true_np
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for j, name in enumerate(dim_names):
            ax = axes[j]
            ax.hist(residuals[:, j], bins=100, alpha=0.7, density=True, color='steelblue')
            mu_r = float(residuals[:, j].mean())
            sd_r = float(residuals[:, j].std())
            ax.axvline(mu_r, color='red', linestyle='--', linewidth=1)
            ax.set_title(f"{name}\nμ={mu_r:+.3f} σ={sd_r:.3f}")
            ax.set_xlabel("Residual")
        fig.suptitle("Residual Distributions (std-space)", fontsize=13)
        fig.tight_layout()
        path5 = os.path.join(out_dir, "rollout_residual_hist.png")
        fig.savefig(path5, dpi=160)
        plt.close(fig)
        print(f"[PLOT] {path5}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")

    # ------------------------------------------------------------------
        # ==================================================================
        # STEP 16: Save JSON summary with all metrics and config
        # ==================================================================
    # 16. Summary
    # ------------------------------------------------------------------
    summary = {
        "r2_full": float(r2_full),
        "mae_full": float(mae_full),
        "Nc": int(Nc),
        "use_innovation": bool(innov_on),
        "kappa_corr": float(kappa_corr),
        "clamp_val": float(clamp_val) if clamp_val else 0.0,
        "lags": cond_lags,
        "elapsed_s": float(elapsed),
        "per_dim": {
            name: {"r2": float(_r2_mae_np(X_true_np[:, j], mu_roll_np[:, j])[0]),
                   "mae": float(_r2_mae_np(X_true_np[:, j], mu_roll_np[:, j])[1])}
            for j, name in enumerate(dim_names)
        },
    }
    summary_path = os.path.join(out_dir, "rollout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary -> {summary_path}")

    print(f"\n[DONE] All outputs saved to {out_dir}/")
    return summary

# ===========================================================================
# Configuration — edit these paths before running
# ===========================================================================
# FOLDER: directory containing the fNIRS data files
# ENCODER_PATH: saved encoder payload from train_y_memory_encoder()
# SCALER_PATH: saved scaler JSON from train_y_given_x()
# YGX_CKPT_PATH: Stage 1 teacher checkpoint (only needed if USE_INNOVATION=True)
# USE_INNOVATION: True = use teacher for Kalman-inspired innovation feedback
#                 False = y-only rollout (no teacher needed)
FOLDER          = "./output"
ENCODER_PATH    = "./output/y_memory_encoder.pt"
SCALER_PATH     = "./output/scaler.json"
YGX_CKPT_PATH   = "./output/ygx_ckpt.pt"       # only used if USE_INNOVATION = True
OUT_DIR         = "./output/inference_results"

USE_INNOVATION  = True      # <--- Toggle: True = use teacher for innovation, False = y-only
COND_GAIN       = 20.0

device = "cuda" if torch.cuda.is_available() else "cpu"

summary = run_rollout_inference(
    folder=FOLDER,
    encoder_path=ENCODER_PATH,
    scaler_path=SCALER_PATH,
    ygx_ckpt_path=YGX_CKPT_PATH,
    out_dir=OUT_DIR,
    device=device,
    use_innovation=USE_INNOVATION,
    cond_gain_for_teacher=COND_GAIN,
)