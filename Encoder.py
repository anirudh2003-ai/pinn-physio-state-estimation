# PURPOSE: Recover 4 cardiovascular dims from a single fNIRS optical signal
#   dim 0: Diastolic BP | dim 1: Systolic BP | dim 2: CO | dim 3: SV
#
# ARCHITECTURE (Sections 2.3-2.7):
#   Stage 1: Conditional normalizing flow P(y|x) - FORWARD mapping
#   Stage 2a: Causal WaveNet student - INVERSE mapping y->x
#   Stage 2b: Y-only MLP+Conv1d backbone - observation anchor
#
# STATE UPDATE (Section 2.7):
#   x_prop = x_prev + dmu_y + dmu_AR + b(t) + dx_innov
#   x_prop = (1-kappa)*x_prop + kappa*x_yonly  [kappa mixing]
#   x_hat = x_prev + g*(x_prop - x_prev)      [gate >= 0.65]
#
# CURRICULUM (Section 4.2):
#   Phase 1: Teacher (18 ep, frozen after)
#   Phase 2 Stage 1: Y-only pretrain (ep 1-7)
#   Phase 2 Stage 2: Closed-loop (ep 8-30, SS 0.3->1.0)
# ============================================================================
# The one with absolute  y values

from __future__ import annotations

from contextlib import nullcontext
from exp_base_static import *
from core_flow import _count_params
from exp_base_static import _ensure_dir
import os, json, math
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
import time
from scipy.ndimage import uniform_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---- Performance toggles (safe defaults on CUDA) ----
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Safety fallback: prevents NameError if exp_base_static does not export _ensure_dir as expected
if "_ensure_dir" not in globals():
    def _ensure_dir(path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# --- Log-probability under normalizing flow (Section 2.3) ---
def _flow_logp_of_inverse(flow, z, cond):
    y, ldj_inv = flow.inverse(z, cond=cond, extra=None)
    if ldj_inv.dim() == 2:
        ldj_inv = ldj_inv.squeeze(-1)
    logp_z = -0.5 * (z ** 2).sum(dim=1)
    return y, logp_z + ldj_inv


@torch.enable_grad()
# --- MAP refinement: gradient ascent in latent space (Section 2.3) ---
def _map_decode_y_base(flow, x_feat, steps=2, lr=0.02, z_clip=0.3, verbose=False):
    device = x_feat.device
    B = x_feat.size(0)
    z = torch.zeros(B, 1, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([z], lr=lr)
    for _ in range(max(1, steps)):
        opt.zero_grad(set_to_none=True)
        _, logp = _flow_logp_of_inverse(flow, z, x_feat)
        (-logp.mean()).backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(-z_clip, z_clip)
    with torch.no_grad():
        y_star, _ = _flow_logp_of_inverse(flow, z, x_feat)
    return y_star


# ---------------------- Warm-up shim (y|x only) ----------------------

@torch.no_grad()
# --- Dummy forward+inverse to trigger lazy inits (ActNorm) ---
def _warmup_y_only(model: YGivenXModel, device):
    model.eval()
    B = 32
    dx = model.config["d_x_cond"]
    xw_feat = torch.randn(B, dx, device=device, dtype=torch.float32)
    yw = torch.randn(B, 1, device=device, dtype=torch.float32)
    zw, _ = model.flow_y_given_x(yw, cond=xw_feat, extra=None)
    _, _ = model.flow_y_given_x.inverse(zw, cond=xw_feat, extra=None)
    _ = model.res_y(xw_feat)
    model.train()


# ---------------------- Utility debug helpers ----------------------

# --- L2 norm of all gradients in a module ---
def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        v = p.grad.data
        if torch.isnan(v).any() or torch.isinf(v).any():
            return float("nan")
        total += v.pow(2).sum().item()
    return math.sqrt(total) if total > 0 else 0.0


# --- q-th percentile of flattened tensor ---
def _pct(t: torch.Tensor, q: float) -> float:
    """
    q in [0,100]. Returns percentile of flattened tensor.
    """
    if t is None:
        return float("nan")
    tt = t.detach()
    if tt.numel() == 0:
        return float("nan")
    tt = tt.float().reshape(-1)
    qq = max(0.0, min(100.0, float(q))) / 100.0
    return float(torch.quantile(tt, qq).item())


# --- True if any tensor contains NaN or Inf ---
def _nan_sentinel(*tensors):
    for t in tensors:
        if torch.isnan(t).any() or torch.isinf(t).any():
            return True
    return False


# --- Soft penalty for |x| exceeding bound (Section 4.3) ---
def _soft_bound_penalty(x: torch.Tensor, bound: float = 5.0, power: float = 2.0):
    """
    Soft penalty for |x| > bound. Returns scalar.
    Avoids hard-clamp-induced saturation while still discouraging divergence.
    """
    if bound is None or bound <= 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    excess = (x.abs() - bound).clamp_min(0.0)
    if power == 1.0:
        return excess.mean()
    return (excess.pow(power)).mean()


# --- KL(N1||N2) diagonal Gaussians - used in shift-KL (Section 4.3) ---
def diag_gauss_kl(m1, v1, m2, v2, eps=1e-6):
    # KL(N1||N2) per-dim (diagonal Gaussians)
    v1 = v1.clamp_min(eps)
    v2 = v2.clamp_min(eps)
    return 0.5 * (torch.log(v2 / v1) + (v1 + (m1 - m2).pow(2)) / v2 - 1.0)


# --- Boolean mask for history dropout (Section 3.1) ---
def _bernoulli_mask(shape, p: float, device):
    if p <= 0.0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if p >= 1.0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return (torch.rand(shape, device=device) >= p)  # True means "keep true history"


# --- Innovation magnitude tracker for rollout diagnostics ---
class InnovTracker:
    """Lightweight innovation magnitude tracker for rollout diagnostics."""

    def __init__(self):
        self.vals = []

    def record(self, innovation: torch.Tensor):
        if innovation is None:
            return
        with torch.no_grad():
            self.vals.append(float(innovation.abs().mean().item()))

    def record_batch(self, innovation: torch.Tensor):
        """Record per-sample magnitudes (for finer percentiles)."""
        if innovation is None:
            return
        with torch.no_grad():
            per_sample = innovation.abs().mean(dim=-1)  # [B]
            self.vals.extend(per_sample.cpu().tolist())

    def summary(self) -> dict:
        if len(self.vals) == 0:
            return {"n": 0}
        v = np.array(self.vals, dtype=np.float32)
        return {
            "n": len(v),
            "mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "frac_above_0.5": float((v > 0.5).mean()),
            "frac_above_1.0": float((v > 1.0).mean()),
        }

    def print_summary(self, tag: str = ""):
        s = self.summary()
        if s["n"] == 0:
            print(f"[INNOV_DIST]{tag} no samples recorded")
            return
        print(
            f"[INNOV_DIST]{tag} n={s['n']} | "
            f"mean={s['mean']:.4f} p50={s['p50']:.4f} "
            f"p90={s['p90']:.4f} p95={s['p95']:.4f} "
            f"p99={s['p99']:.4f} max={s['max']:.4f} | "
            f"frac>0.5={s['frac_above_0.5']:.3f} "
            f"frac>1.0={s['frac_above_1.0']:.3f}",
            flush=True
        )

    def reset(self):
        self.vals = []


# --- Normalise kappa to consistent shape for mixing (Section 2.7) ---
def _ensure_kappa_vec_or_batch(
        k: torch.Tensor | float | None,
        *,
        x_prop: torch.Tensor,  # the tensor we will mix into; defines device/dtype/target batch
        D: int,
) -> torch.Tensor:
    """
    Ensures kappa is never empty and is one of:
      - (D,)         per-dim vector
      - (1, D)       broadcastable vector
      - (B, D)       per-sample batch
    Also collapses (N, D) "per-match" kappa into (D,) by mean, unless it already matches (B, D).
    Neutral fallback is zeros(D) => no mixing with x_yonly_prop.
    """
    device = x_prop.device
    dtype = x_prop.dtype
    B = x_prop.size(0)

    # None => neutral
    if k is None:
        return torch.zeros(D, device=device, dtype=dtype)

    # Convert scalars
    if not torch.is_tensor(k):
        return torch.full((D,), float(k), device=device, dtype=dtype)

    # Tensor => move to correct device/dtype
    k = k.to(device=device, dtype=dtype)

    # Empty => neutral
    if k.numel() == 0:
        return torch.zeros(D, device=device, dtype=dtype)

    # Handle shapes
    if k.dim() == 0:
        return k.expand(D)

    if k.dim() == 1:
        if k.numel() == D:
            return k
        if k.numel() == 1:
            return k.expand(D)
        raise RuntimeError(f"[kappa] expected (D,) or scalar; got {tuple(k.shape)} (D={D})")

    if k.dim() == 2:
        # If already per-sample batch (B,D), keep it
        if k.shape == (B, D):
            return k

        # If (0,D) handled above; if (1,D) keep it
        if k.shape[0] == 1 and k.shape[1] == D:
            return k

        # If looks like (N,D) per-match/per-geom, collapse N -> D
        if k.shape[1] == D and k.shape[0] > 1:
            return k.mean(dim=0)

        raise RuntimeError(f"[kappa] bad 2D shape {tuple(k.shape)}; expected (B,D), (1,D), or (N,D) with D={D}")

    raise RuntimeError(f"[kappa] bad rank {k.dim()} for shape {tuple(k.shape)}")


# --- Shift-invariant contiguous batch sampler (Section 3.1) ---
class RandomContiguousBatchSampler(Sampler):
    """
    Preserves contiguity within each batch (needed for rollout unrolls that treat batch as time),
    but randomizes start offsets each epoch (shift-invariant training).

    Works for datasets where index i corresponds to timestep i in a single long sequence
    (which is exactly your _make_windows(...) output order).
    """

    def __init__(self, n: int, batch_size: int, drop_last: bool = True, generator: torch.Generator | None = None):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.gen = generator
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __iter__(self):
        n = self.n
        b = self.batch_size
        if n < b:
            return iter([])

        n_batches = n // b if self.drop_last else (n + b - 1) // b
        g = self.gen
        starts = torch.randint(0, n - b + 1, (n_batches,), generator=g)
        for s in starts.tolist():
            idx = list(range(s, s + b))
            if len(idx) < b and self.drop_last:
                continue
            yield idx

    def __len__(self):
        n = self.n
        b = self.batch_size
        if self.drop_last:
            return n // b
        return (n + b - 1) // b


# ---------------------- Loaders ----------------------

# --- Load data, build 10-dim teacher inputs (Section 2.1, 4.1) ---
def make_loaders_from_folder(folder, batch=128, val_frac=0.2, x_norm_stats_in: dict | None = None):
    df = load_folder_and_join(folder)
    # Detrend ShortChannel
    y_raw = df["ShortChannel"].to_numpy(np.float32)
    y_smooth = uniform_filter1d(y_raw, size=800)
    df["ShortChannel"] = y_raw - y_smooth

    # ---- Restore feature extraction (accidentally deleted) ----
    N_initial = len(df)
    n_val_initial = int(max(1, round(N_initial * val_frac)))
    tr_slice_initial = slice(0, max(0, N_initial - n_val_initial))

    if x_norm_stats_in is None:
        x_raw_raw, x_feat, feat_names, x_norm_stats = build_x_features_from_df(
            df, x_norm_mu=None, x_norm_sd=None, fit_slice=tr_slice_initial, save_stats_path="./output/x_norm_stats.json"
        )
    else:
        x_raw_raw, x_feat, feat_names, x_norm_stats = build_x_features_from_df(
            df,
            x_norm_mu=np.array(x_norm_stats_in["x_norm_mu"], dtype=np.float32),
            x_norm_sd=np.array(x_norm_stats_in["x_norm_sigma"], dtype=np.float32),
            fit_slice=None, save_stats_path=None
        )

    # ---- NEW: Build Teacher Input using Y-only Bandpass Features ----
    N = len(df)
    if "Time" in df.columns and len(df) > 5:
        dt = float(np.median(np.diff(df["Time"].to_numpy(np.float32))))
        fs_est = 1.0 / max(dt, 1e-6)
    else:
        fs_est = 30.0

    # Extract Y bandpass features to provide temporal context
    y_only_features, y_feat_names = build_y_only_features(df, fs_hint=fs_est)
    bp_names = ['Y_bp_energy_HR', 'Y_bp_envelope_HR', 'Y_bp_energy_RR',
                'Y_bp_envelope_RR', 'Y_bp_energy_Trend', 'Y_bp_cardiac_raw']
    bp_indices = [y_feat_names.index(n) for n in bp_names if n in y_feat_names]
    y_bp_feats = y_only_features[:, bp_indices].astype(np.float32)

    # Concatenate base X (4 dims) with Y bandpass features
    x_feat = np.concatenate([x_feat, y_bp_feats], axis=1)
    feat_names = feat_names + [y_feat_names[i] for i in bp_indices]
    d_x_cond = x_feat.shape[1]
    print(f"[Loader] Teacher input: {d_x_cond} dims (4 base X + {len(bp_indices)} Y-derived)")

    y = df[["ShortChannel"]].to_numpy(np.float32)

    n_val_expected = int(round(N * val_frac))
    if n_val_expected > 0 and val_frac > 0:
        val_step = max(1, int(round(1.0 / val_frac)))
        va_mask = (np.arange(N) % val_step == 0)
        tr_mask = ~va_mask
        n_val = int(va_mask.sum())
    else:
        va_mask = np.zeros(N, dtype=bool)
        tr_mask = np.ones(N, dtype=bool)
        n_val = 0

    x_feat_tr, x_feat_va = (x_feat[tr_mask], x_feat[va_mask]) if n_val > 0 else (x_feat, x_feat[0:0])
    y_tr, y_va = (y[tr_mask], y[va_mask]) if n_val > 0 else (y, y[0:0])

    x_feat_mu, x_feat_sd = standardize_train_stats(x_feat_tr)
    y_mu, y_sd = standardize_train_stats(y_tr)

    x_feat_tr_std = apply_scaling_np(x_feat_tr, x_feat_mu, x_feat_sd)
    y_tr_std = apply_scaling_np(y_tr, y_mu, y_sd)
    x_feat_va_std = apply_scaling_np(x_feat_va, x_feat_mu, x_feat_sd) if n_val > 0 else x_feat_va
    y_va_std = apply_scaling_np(y_va, y_mu, y_sd) if n_val > 0 else y_va

    # feature distribution debug
    def _stat(msg, arr):
        arr = np.asarray(arr, dtype=np.float32)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        mu_s = ", ".join(f"{v:+.3f}" for v in mu.ravel()[:8])
        sd_s = ", ".join(f"{v:.3f}" for v in sd.ravel()[:8])
        print(f"[stats] {msg} | mean[:]={mu_s} | std[:]={sd_s}")

    _stat("x_feat_tr_std", x_feat_tr_std)
    _stat("y_tr_std", y_tr_std)
    if n_val > 0:
        _stat("x_feat_va_std", x_feat_va_std)
        _stat("y_va_std", y_va_std)

    ds_tr = TensorDataset(torch.from_numpy(x_feat_tr_std), torch.from_numpy(y_tr_std))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=False, num_workers=0)
    dl_va = None
    if n_val > 0:
        ds_va = TensorDataset(torch.from_numpy(x_feat_va_std), torch.from_numpy(y_va_std))
        dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    scaler = {
        "x_norm_mu": x_norm_stats["x_norm_mu"],
        "x_norm_sigma": x_norm_stats["x_norm_sigma"],
        "x_feat_mu": x_feat_mu.squeeze().tolist(),
        "x_feat_sigma": x_feat_sd.squeeze().tolist(),
        "y_mu": y_mu.squeeze().tolist(),
        "y_sigma": y_sd.squeeze().tolist(),
        "feat_names": feat_names
    }
    dims = {"d_x_cond": x_feat.shape[1]}
    return df, dl_tr, dl_va, scaler, dims


# ---------------------- Calibration helpers ----------------------

# --- Affine calibration y_true = a*y_pred + b via least-squares ---
def _affine_from_xy(y_pred, y_true):
    var_hat = float(np.var(y_pred) + 1e-12)
    cov = float(np.mean((y_pred - y_pred.mean()) * (y_true - y_true.mean())))
    a = cov / var_hat
    b = float(y_true.mean() - a * y_pred.mean())
    return a, b


# --- Apply calibration (affine, isotonic, or blend) ---
def _apply_calib_blob(y_pred, calib):
    if calib.get("type") == "blend_iso":
        xk = np.asarray(calib["xk"], dtype=np.float64)
        yk = np.asarray(calib["yk"], dtype=np.float64)
        a = float(calib["a"])
        b = float(calib.get("b", 0.0))
        alpha = float(calib.get("alpha", 0.25))
        iso_y = np.interp(y_pred, xk, yk)
        return alpha * iso_y + (1.0 - alpha) * (a * y_pred + b)
    elif calib.get("type") == "isotonic":
        xk = np.asarray(calib["xk"], dtype=np.float64)
        yk = np.asarray(calib["yk"], dtype=np.float64)
        return np.interp(y_pred, xk, yk)
    elif calib.get("type") == "affine":
        a = float(calib.get("a", 1.0))
        b = float(calib.get("b", 0.0))
        return a * y_pred + b
    else:
        return y_pred


# --- R-squared and MAE ---
def _r2_mae_np(y_t, y_p):
    y_t = np.asarray(y_t).ravel()
    y_p = np.asarray(y_p).ravel()
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_t - y_p)))
    return r2, mae


def _slope(y_t, y_p):
    y_t = np.asarray(y_t).ravel()
    y_p = np.asarray(y_p).ravel()
    var_p = float(np.var(y_p) + 1e-12)
    cov = float(np.mean((y_p - y_p.mean()) * (y_t - y_t.mean())))
    return cov / var_p


# --- Calibration guardrails: variance, monotonicity, slope, R2 ---
def _calib_guardrails(y_true, y_pred_raw, calib_candidate, min_var_ratio=0.85, min_bins=10, slope_must_not_drop=True):
    y_cal = _apply_calib_blob(y_pred_raw, calib_candidate)
    var_ratio = float(np.var(y_cal) / (np.var(y_pred_raw) + 1e-12))
    slope_raw = _slope(y_true, y_pred_raw)
    slope_cal = _slope(y_true, y_cal)
    xs = np.linspace(np.min(y_pred_raw), np.max(y_pred_raw), 256)
    ys = _apply_calib_blob(xs, calib_candidate)
    diffs = np.abs(np.diff(ys))
    bins = int(np.sum(diffs > 1e-6))
    r2_raw, _ = _r2_mae_np(y_true, y_pred_raw)
    r2_cal, _ = _r2_mae_np(y_true, y_cal)
    ok_var = var_ratio >= min_var_ratio
    ok_bins = bins >= min_bins
    ok_slope = (not slope_must_not_drop) or (slope_cal >= slope_raw - 1e-6)
    ok_r2 = r2_cal >= r2_raw - 1e-6
    return {
        "ok": bool(ok_var and ok_bins and ok_slope and ok_r2),
        "var_ratio": var_ratio,
        "bins": bins,
        "slope_raw": slope_raw,
        "slope_cal": slope_cal,
        "r2_raw": r2_raw,
        "r2_cal": r2_cal
    }


# --- Bagged 3-fold isotonic + affine blend ---
def _bagged_isotonic(y_pred_all, y_true_all, iso_knots=64, alpha=0.25, a_aff=1.0, b_aff=0.0):
    from sklearn.isotonic import IsotonicRegression
    yhat = y_pred_all.ravel()
    ytru = y_true_all.ravel()
    n = len(yhat)
    thirds = np.array_split(np.arange(n), 3)
    xk = np.quantile(yhat, np.linspace(0, 1, iso_knots)).astype(np.float64)
    yk_list = []
    for idx in thirds:
        if len(idx) < 10:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(yhat[idx], ytru[idx])
        yk_list.append(iso.predict(xk).astype(np.float64))
    if not yk_list:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(yhat, ytru)
        yk = iso.predict(xk).astype(np.float64)
    else:
        yk = np.mean(np.stack(yk_list, axis=0), axis=0)
    return {"type": "blend_iso", "xk": xk.tolist(), "yk": yk.tolist(), "a": float(a_aff), "b": float(b_aff),
            "alpha": float(alpha)}


# ---------------------- Cosine LR with warmup ----------------------

# --- Cosine LR with linear warmup ---
def _build_warmup_cosine(optimizer, steps_per_epoch, epochs, warmup_epochs=1, min_lr_mult=0.1):
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = steps_per_epoch * max(1, warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / float(max(1, warmup_steps)))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cos_mult = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_mult + (1 - min_lr_mult) * cos_mult

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------- Train loop (R² early stop, opt-in calibration) ----------------------

# ============================================================================
# Stage 1: Train P(y|x) teacher (Section 2.3, 4.1)
# K=8 coupling layers + Student-t residual. 10-dim input, gain=20.
# Loss: Student-t NLL + Jacobian sensitivity + sigma priors + 68%% coverage.
# Frozen after training.
# ============================================================================
def train_y_given_x(folder="./output", ckpt_out="./output/ygx_ckpt.pt", scaler_out="./output/scaler.json",
                    epochs=40, lr_flow=3e-5, lr_res=1e-5, batch=128, val_frac=0.2, device=None,
                    max_grad_norm=5.0,
                    lambda_sens_start=1e-2, lambda_sens_end=1e-4,
                    lambda_sigma_prior=5e-3,
                    lambda_sigma_cov=2e-3,
                    lambda_l1=1e-3,
                    gate_floor_start=0.52, gate_floor_peak=0.58, gate_floor_tail=0.50,
                    cond_gain_start=12.0, cond_gain_end=12.0,
                    es_patience=4,
                    use_isotonic=True,
                    iso_alpha=0.25, iso_knots=64,
                    guard_var_ratio=0.85, guard_bins=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(scaler_out) or ".", exist_ok=True)

    df, dl_tr, dl_va, scaler, dims = make_loaders_from_folder(folder, batch=batch, val_frac=val_frac)

    model = YGivenXModel(d_x_cond=dims["d_x_cond"], cond_gain_y=cond_gain_start).to(device)
    print(f"Using device: {device}")
    print(f"Model trainable params: {_count_params(model):,}")

    _warmup_y_only(model, device)
    reset_actnorm_flags(model)

    opt_flow = torch.optim.AdamW(model.flow_y_given_x.parameters(), lr=lr_flow, weight_decay=2e-5, betas=(0.9, 0.98))
    opt_res = torch.optim.AdamW(list(model.res_y.parameters()) + list(model.delta_skip.parameters()),
                                lr=lr_res, weight_decay=2e-5, betas=(0.9, 0.98))

    steps_per_epoch = len(dl_tr) if dl_tr is not None else 1
    sch_flow = _build_warmup_cosine(opt_flow, steps_per_epoch, epochs, warmup_epochs=1, min_lr_mult=0.1)
    sch_res = _build_warmup_cosine(opt_res, steps_per_epoch, epochs, warmup_epochs=1, min_lr_mult=0.1)

    best_r2 = -1e9
    best_tail_r2 = -1e9
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_epoch = 0
    since_improve = 0
    calib_blob = {"type": "affine", "a": 1.0, "b": 0.0, "notes": "identity-by-default"}

    y_mu = np.array(scaler["y_mu"], dtype=np.float32)
    y_sd = np.array(scaler["y_sigma"], dtype=np.float32)

    def _lin_anneal(a, b, t):
        return a + (b - a) * t

    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        n_tr = 0

        t_ep0 = time.time()
        batches_total = len(dl_tr)

        # --- Gate floor schedule: ramp up/down/constant ---
        # gate_floor schedule
        if ep <= 10:
            gate_floor = _lin_anneal(gate_floor_start, gate_floor_peak, ep / 10.0)
        elif ep <= 20:
            gate_floor = _lin_anneal(gate_floor_peak, gate_floor_tail, (ep - 10) / 10.0)
        else:
            gate_floor = gate_floor_tail
        for blk in model.flow_y_given_x.blocks:
            blk.gate_floor = float(gate_floor)

        # --- Sensitivity schedule: constant 8ep then anneal ---
        # sensitivity schedule
        if ep <= 8:
            lam_sens = lambda_sens_start
        else:
            t = (ep - 8) / max(1, (epochs - 8))
            lam_sens = _lin_anneal(lambda_sens_start, lambda_sens_end, t)

        # l1 schedule
        if ep >= int(0.6 * epochs):
            t_l1 = (ep - int(0.6 * epochs)) / max(1, epochs - int(0.6 * epochs))
            lam_l1 = lambda_l1 * (1.0 - 0.7 * t_l1)
        else:
            lam_l1 = lambda_l1

        # cond-gain anneal honored
        if epochs <= 1:
            cond_gain_scale = float(cond_gain_start)
        else:
            t_cg = (ep - 1) / float(max(1, epochs - 1))
            cond_gain_scale = float(_lin_anneal(cond_gain_start, cond_gain_end, t_cg))

        print(
            f"[train][ep {ep:03d}] gate_floor={gate_floor:.3f} λ_sens={lam_sens:.2e} λ_l1={lam_l1:.2e} cond_gain={cond_gain_scale:.3f}")
        t_ep0 = time.time()
        batches_total = len(dl_tr)

        for bi, (xb_feat, yb) in enumerate(dl_tr, start=1):
            xb_feat = xb_feat.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            xb_feat = torch.nan_to_num(xb_feat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)

            opt_flow.zero_grad(set_to_none=True)
            opt_res.zero_grad(set_to_none=True)

            # Flow inverse at z=0: mode prediction y_base (Section 2.3)
            z0 = torch.zeros_like(yb)
            y_base, _ = model.flow_y_given_x.inverse(z0, cond=xb_feat * cond_gain_scale, extra=None,
                                                     collect_stats=False)

            delta_core, logsig_y, nu_y = model.res_y(xb_feat)
            delta_y = delta_core + model.delta_skip(xb_feat)

            # Student-t NLL on residual (Section 2.3)
            e_y_star = yb - y_base
            nll_y = student_t_nll(e_y_star, delta_y, logsig_y, nu_y)
            l1 = F.smooth_l1_loss(y_base + delta_y, yb, beta=0.2, reduction='mean')

            # Jacobian sensitivity computation (Section 2.3)
            xb_req = xb_feat.detach().requires_grad_(True)
            y_jac, _ = model.flow_y_given_x.inverse(torch.zeros_like(yb), cond=xb_req * cond_gain_scale, extra=None)
            (J_yx,) = safe_autograd_grad(y_jac.sum(), xb_req, create_graph=True, retain_graph=True)
            J_yx = J_yx.clamp(-5, 5)
            sens_loss = (J_yx ** 2).sum(dim=1).mean()

            with torch.no_grad():
                target_sig_y = e_y_star.abs().mean().clamp_min(1e-6)
                log_sig_target_y = target_sig_y.log()
            # Sigma prior: variance-calibrated (Section 2.3)
            sig_prior = (logsig_y - log_sig_target_y).pow(2).mean()
            sigma_y = torch.exp(logsig_y)
            err_y = (e_y_star - delta_y).abs()
            cov_est_y = torch.sigmoid(3.0 * (sigma_y - err_y)).mean()
            # 68%% coverage regularisation (Section 2.3)
            cov_reg_y = (cov_est_y - 0.68).pow(2)

            loss = nll_y + lam_l1 * l1 + lam_sens * sens_loss + lambda_sigma_prior * sig_prior + lambda_sigma_cov * cov_reg_y
            loss.backward()

            g_flow_pre = _grad_norm(model.flow_y_given_x)
            g_res_pre = _grad_norm(model.res_y) + _grad_norm(model.delta_skip)

            nn.utils.clip_grad_norm_(list(model.parameters()), max_grad_norm)
            opt_flow.step()
            opt_res.step()

            g_flow_post = _grad_norm(model.flow_y_given_x)
            g_res_post = _grad_norm(model.res_y) + _grad_norm(model.delta_skip)

            tr_loss_sum += float(nll_y.detach()) * xb_feat.size(0)
            n_tr += xb_feat.size(0)
            global_step += 1
            sch_flow.step()
            sch_res.step()

            if (bi == 1) or (bi % 20 == 0):
                with torch.no_grad():
                    y_pred_std = (y_base + delta_y).detach().cpu().numpy()
                    y_true_std = yb.detach().cpu().numpy()
                    y_pred = invert_scaling_np(y_pred_std, y_mu, y_sd)
                    y_true = invert_scaling_np(y_true_std, y_mu, y_sd)
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    elapsed = time.time() - t_ep0
                    pct = 100.0 * bi / max(1, batches_total)

                    print(
                        f"[ep {ep:03d} | b {bi:04d}/{batches_total} {pct:5.1f}% | {elapsed:6.1f}s] use_map=0  "
                        f"NLL(z0)={float(nll_y.mean().item()):.6f}  "
                        f"||J||²={float(sens_loss.item()):.4e}  "
                        f"σ_prior={float(sig_prior.item()):.4e}  "
                        f"cov68_err={float(cov_reg_y.item()):.4e}  "
                        f"MAE(z0)={mae:.4f}  "
                        f"sig_mean={float(torch.exp(logsig_y).mean().item()):.4f}  "
                        f"grad||flow|| pre={g_flow_pre:.4f}→post={g_flow_post:.4f}  "
                        f"grad||res(+skip)|| pre={g_res_pre:.4f}→post={g_res_post:.4f}  "
                        f"nan?={_nan_sentinel(y_base, delta_y, logsig_y)}"
                    )

        model.eval()
        va_loss_sum = 0.0
        n_va = 0
        vy_true_list = []
        vy_pred_list = []
        with torch.no_grad():
            if dl_va is not None:
                for xb_feat, yb in dl_va:
                    xb_feat = xb_feat.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device, dtype=torch.float32)
                    xb_feat = torch.nan_to_num(xb_feat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
                    yb = torch.nan_to_num(yb, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)

                    z0 = torch.zeros_like(yb)
                    y_base, _ = model.flow_y_given_x.inverse(z0, cond=xb_feat * cond_gain_scale, extra=None)
                    e_y_star = yb - y_base
                    delta_core, logsig_y, nu_y = model.res_y(xb_feat)
                    delta_y = delta_core + model.delta_skip(xb_feat)
                    nll_y = student_t_nll(e_y_star, delta_y, logsig_y, nu_y)
                    va_loss_sum += float(nll_y.item()) * xb_feat.size(0)
                    n_va += xb_feat.size(0)
                    y_pred = y_base + delta_y
                    vy_true_list.append(invert_scaling_np(yb.cpu().numpy(), y_mu, y_sd))
                    vy_pred_list.append(invert_scaling_np(y_pred.cpu().numpy(), y_mu, y_sd))

        tr_loss = tr_loss_sum / max(1, n_tr)
        if n_va > 0:
            va_loss = va_loss_sum / max(1, n_va)
            y_true_all = np.concatenate(vy_true_list, axis=0)
            y_pred_all = np.concatenate(vy_pred_list, axis=0)
            r2_raw, mae = _r2_mae_np(y_true_all, y_pred_all)
            bias = float(np.mean(y_pred_all - y_true_all))
            denom = (np.std(y_pred_all) * np.std(y_true_all) + 1e-12)
            corr = float(np.mean((y_pred_all - y_pred_all.mean()) * (y_true_all - y_true_all.mean())) / denom)
            k_tail = max(1, int(0.05 * len(y_true_all)))
            r2_tail, mae_tail = _r2_mae_np(y_true_all[-k_tail:], y_pred_all[-k_tail:])

            print(f"ep {ep:03d} | train NLL(z0) {tr_loss:.6f} | valid NLL(z0) {va_loss:.6f} | "
                  f"VAL y: MAE={mae:.4f} R^2_raw={r2_raw:.4f} bias={bias:.4f} corr={corr:.4f}")
            print(f"[tail] last5%: MAE={mae_tail:.4f} R^2_tail={r2_tail:.4f}")

            a, b = _affine_from_xy(y_pred_all, y_true_all)
            calib_aff = {"type": "affine", "a": float(a), "b": float(b)}
            best_local_calib = calib_aff
            y_aff = _apply_calib_blob(y_pred_all, calib_aff)
            r2_aff, _ = _r2_mae_np(y_true_all, y_aff)
            best_local_r2 = r2_aff

            if use_isotonic:
                try:
                    calib_iso_blend = _bagged_isotonic(y_pred_all, y_true_all, iso_knots=iso_knots, alpha=iso_alpha,
                                                       a_aff=a, b_aff=b)
                    guards = _calib_guardrails(y_true_all, y_pred_all, calib_iso_blend,
                                               min_var_ratio=guard_var_ratio, min_bins=guard_bins)
                    y_blend = _apply_calib_blob(y_pred_all, calib_iso_blend)
                    r2_blend, _ = _r2_mae_np(y_true_all, y_blend)
                    print(f"[calib] affine R^2={r2_aff:.4f} | blend_iso R^2={r2_blend:.4f} "
                          f"| var_ratio={guards['var_ratio']:.3f} bins={guards['bins']} "
                          f"| slope_raw={guards['slope_raw']:.3f} slope_cal={guards['slope_cal']:.3f}")
                    if guards["ok"] and (r2_blend >= best_local_r2 - 1e-6):
                        best_local_calib = calib_iso_blend
                        best_local_r2 = r2_blend
                except Exception as e:
                    print(f"[calib] isotonic failed, keeping affine. reason: {e}")

            improved = (best_local_r2 > best_r2 + 1e-6)
            tail_better = False
            if not improved and abs(best_local_r2 - best_r2) <= 1e-6:
                if r2_tail > best_tail_r2 + 1e-3:
                    tail_better = True

            if improved or tail_better:
                best_r2 = best_local_r2
                best_tail_r2 = r2_tail
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = ep
                since_improve = 0
                calib_blob = best_local_calib
                print(
                    f"[best] epoch {ep}: R^2_cal={best_local_r2:.4f} (saved) with calib type={calib_blob['type']} | R^2_tail={r2_tail:.4f}")
            else:
                since_improve += 1
        else:
            print(f"ep {ep:03d} | train NLL(z0) {tr_loss:.6f}")

        if since_improve >= es_patience and ep >= 6:
            print(
                f"[early-stop] No val calibrated R^2 improvement for {es_patience} epochs; stopping at epoch {ep}. Best at {best_epoch} (R^2_cal={best_r2:.4f}).")
            break

    # <--- FIX: Restore the best weights into the live model before returning it --->
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save(best_state, ckpt_out)
    scaler_out_obj = {
        "x_norm_mu": scaler["x_norm_mu"],
        "x_norm_sigma": scaler["x_norm_sigma"],
        "x_feat_mu": scaler["x_feat_mu"],
        "x_feat_sigma": scaler["x_feat_sigma"],
        "y_mu": scaler["y_mu"],
        "y_sigma": scaler["y_sigma"],
        "feat_names": scaler["feat_names"],
        "calibration": calib_blob
    }
    with open(scaler_out, "w") as f:
        json.dump(scaler_out_obj, f, indent=2)
    print(f"Saved ckpt → {ckpt_out}\nSaved scaler+calibration → {scaler_out}")
    return model, scaler_out_obj


# ---------------------- Geometry-aware Y→X encoder (fast + x_prev priority + geom ramp) ----------------------

# --- Gaussian NLL with clamped log-sigma (Stage 2a) ---
def _gaussian_nll(x_true, x_mu, x_logsig):
    x_logsig = x_logsig.clamp(-6.0, 2.0)
    inv_var = torch.exp(-2.0 * x_logsig)
    nll = 0.5 * ((x_true - x_mu) ** 2) * inv_var + x_logsig
    return nll.sum(dim=-1)  # [B]


# --- Save X trajectory plots (true/TF/rollout) + CSV ---
def _save_x_diagnostics(
        out_dir,
        ep,
        times,
        X_true_np,
        X_pred_np,
        dim_names,
        max_points_plot=2000,
        X_pred_tf_np=None,
        X_pred_roll_np=None,
        save_csv: bool = True
):
    """
    Backwards-compatible:
      - If called with the old signature, X_pred_np is used as rollout pred.
      - If X_pred_tf_np is provided but X_pred_roll_np is None, it will produce TF-only plots.
      - If both TF and rollout are provided, it plots all three.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os, math
    import pandas as pd
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # Backwards compatibility:
    # if no explicit roll provided, treat X_pred_np as rollout UNLESS TF-only is intended.
    tf_only = (X_pred_tf_np is not None) and (X_pred_roll_np is None)
    if X_pred_roll_np is None and not tf_only:
        X_pred_roll_np = X_pred_np

    df_out = pd.DataFrame({"Time": times})

    for j, name in enumerate(dim_names):
        df_out[f"X_true_{name}"] = X_true_np[:, j]
        if X_pred_roll_np is not None:
            df_out[f"X_pred_roll_{name}"] = X_pred_roll_np[:, j]
            df_out[f"X_res_roll_{name}"] = X_pred_roll_np[:, j] - X_true_np[:, j]
        if X_pred_tf_np is not None:
            df_out[f"X_pred_tf_{name}"] = X_pred_tf_np[:, j]
            df_out[f"X_res_tf_{name}"] = X_pred_tf_np[:, j] - X_true_np[:, j]

    if save_csv:
        csv_path = os.path.join(out_dir, f"x_true_vs_pred_ep{ep:03d}.csv")
        df_out.to_csv(csv_path, index=False)

    # downsample for plotting
    n = len(times)
    stride = max(1, int(math.ceil(n / float(max_points_plot))))
    ts = times[::stride]
    Xt = X_true_np[::stride]
    Xtf = X_pred_tf_np[::stride] if X_pred_tf_np is not None else None
    Xr = X_pred_roll_np[::stride] if X_pred_roll_np is not None else None

    for j, name in enumerate(dim_names):
        plt.figure(figsize=(14, 4))
        plt.plot(ts, Xt[:, j], label="True")
        if Xtf is not None:
            plt.plot(ts, Xtf[:, j], label="Pred(TF)")
        if Xr is not None:
            plt.plot(ts, Xr[:, j], label="Pred(Rollout)")
        plt.title(f"X trajectory (teacher space) — {name} — epoch {ep}")
        plt.xlabel("Time")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"x_traj_{name}_ep{ep:03d}.png"), dpi=160)
        plt.close()


# ---- Strictly CAUSAL WaveNet blocks (left pad only) ----
# --- Causal WaveNet block: tanh*sigmoid gating (Section 2.4) ---
# Dilations [1,2,4,8,16,32] -> ~42s receptive field at 6Hz
class WaveNetBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.dilation = int(dilation)
        self.conv_f = nn.Conv1d(ch, ch, kernel_size=3, dilation=self.dilation, padding=0)
        self.conv_g = nn.Conv1d(ch, ch, kernel_size=3, dilation=self.dilation, padding=0)
        self.proj = nn.Conv1d(ch, ch, kernel_size=1)

    def _causal_conv(self, conv, x):
        # Causal left-pad: output[t] depends only on input[<=t]
        pad_left = 2 * self.dilation
        x = F.pad(x, (pad_left, 0))
        return conv(x)

    def forward(self, x):
        f_in = self._causal_conv(self.conv_f, x)
        g_in = self._causal_conv(self.conv_g, x)

        # Gated activation: tanh=filter, sigmoid=gate (Section 2.4)
        f = torch.tanh(f_in)
        g = torch.sigmoid(g_in)
        h = f * g
        h = self.proj(h)
        return (x + h) * (1.0 / math.sqrt(2.0))


# --- Constants: DELTA_SCALE=2(tanh bound), LOGSIG_SCALE=0.5, GATE_BIAS=1.5 ---
_DELTA_SCALE = 2.0
_LOGSIG_SCALE = 0.5
_GATE_BIAS_INIT = 1.5
_USE_SPECTRAL_NORM_AR = True
_DISABLE_AR_DIAG = False
_DISABLE_INNOV_DIAG = False
print(f"[CONFIG] AR_disabled={_DISABLE_AR_DIAG} INNOV_disabled={_DISABLE_INNOV_DIAG}", flush=True)

_FORCE_GATE_OPEN = False
_SAT_THRESH = 2.0


# --- Single truth: innovation active? (Section 2.6) ---
def _innov_enabled(use_innovation_feedback: bool, innov_disabled: bool) -> bool:
    # single source of truth
    return bool(use_innovation_feedback) and (not bool(innov_disabled))


# --- Optional spectral norm on AR sub-net (Section 2.4) ---
def _sn(linear: nn.Linear):
    return nn.utils.parametrizations.spectral_norm(linear) if _USE_SPECTRAL_NORM_AR else linear


# --- Gated EMA bias update (Section 2.6): b=(1-g)*b_prev + g*b_hat ---
def gated_ema_bias_update(b_prev, b_hat, g_logits, *, g_min=0.01, g_max=0.50, b_max=10.0):
    # Sigmoid in fp32 for stability
    g01 = torch.sigmoid(g_logits.float())  # compute in fp32
    # Map to [g_min, g_max]
    g = g_min + (g_max - g_min) * g01  # fp32
    g = g.to(dtype=b_prev.dtype)  # back to model dtype
    # EMA update: slow integration, tracks persistent offset
    b_next = (1.0 - g) * b_prev + g * b_hat
    b_next = torch.clamp(b_next, -b_max, b_max)
    return b_next, g


# ============================================================================
# Stage 2a+2b: Causal WaveNet Student (Sections 2.4-2.7)
# 2a: Conv1d->6 blocks->z, 5 heads, AR sub-net
# 2b: Separate MLP+Conv1d y-only backbone
# State: x = x_prev + g*(proposal - x_prev), g>=0.65
# ============================================================================
class YMemoryWaveNet(nn.Module):
    """
    Strictly causal WaveNet student with explicit drift correction + gated/leaky update + optional innovation feedback.
    Outputs:
      (z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t)
    """

    def __init__(
            self,
            d_yfeat,
            d_mem=512,
            hidden=112,
            layers=6,
            dilations=None,
            d_x=4,

    ):
        super().__init__()

        # Input projection: Conv1d(75->112)
        self.inp = nn.Conv1d(d_yfeat, hidden, kernel_size=1)
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128][:layers]

        # 6 causal blocks: dilations [1,2,4,8,16,32]
        self.blocks = nn.ModuleList([WaveNetBlock(hidden, d) for d in dilations])

        # Output projection: Conv1d(112->112->512) -> memory z
        self.out = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1), nn.GELU(),
            nn.Conv1d(hidden, d_mem, kernel_size=1)
        )

        # Head 1: observation increment dmu_y [B,4]
        self.head_dx_y = nn.Linear(d_mem, d_x)
        # Head 2: log-sigma (uncertainty) [B,4]
        self.head_logsig = nn.Linear(d_mem, d_x)
        # Innovation embedding: scalar->512-dim (Section 2.6), zero-init
        self.innov_embed = nn.Linear(1, d_mem)
        nn.init.zeros_(self.innov_embed.weight)
        nn.init.zeros_(self.innov_embed.bias)

        # Innovation direct MLP: 8-step buffer+x_prev->dx_innov, zero-init
        self.innov_buffer_size = 8
        self.innov_direct = nn.Sequential(
            nn.Linear(self.innov_buffer_size + d_x, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, d_x)
        )
        nn.init.zeros_(self.innov_direct[-1].weight)
        nn.init.zeros_(self.innov_direct[-1].bias)

        # Head 3: bias estimate b_hat (Section 2.6)
        self.head_corr = nn.Linear(d_mem, d_x)  # will output b_hat logits/raw
        # Head 4: bias gate g_b logits (Section 2.6)
        self.head_corr_gate = nn.Linear(d_mem, d_x)  # g_logits for EMA gate
        nn.init.normal_(self.head_corr_gate.weight, mean=0.0, std=1e-3)

        # Start around g≈0.10 (before affine-map into [g_min,g_max]).
        # logit(0.10) ≈ -2.197
        nn.init.constant_(self.head_corr_gate.bias, -0.224)

        nn.init.zeros_(self.head_corr.weight);
        nn.init.zeros_(self.head_corr.bias)

        nn.init.zeros_(self.head_dx_y.weight);
        nn.init.zeros_(self.head_dx_y.bias)
        nn.init.zeros_(self.head_logsig.weight);
        nn.init.zeros_(self.head_logsig.bias)

        # AR sub-net: 2-layer, spectral norm (Section 3.3)
        self.ar = nn.Sequential(
            _sn(nn.Linear(d_x, 64)), nn.GELU(), nn.LayerNorm(64),
            _sn(nn.Linear(64, 2 * d_x))
        )
        for m in self.ar.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight);
                nn.init.zeros_(m.bias)

        # Head 5: state gate g, floor 0.65 (Section 2.7)
        self.gate = nn.Linear(d_mem, d_x)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(_GATE_BIAS_INIT))

        self.d_x = d_x
        # AR dropout p=0.4: prevents persistence collapse (Section 3.3)
        self.ar_dropout_p = 0.4

        # Per-dim kappa logits: sigmoid->clamp(0.85) (Section 2.7)
        self.kappa_logits = nn.Parameter(torch.full((d_x,), 1.7))

        # === Stage 2b: Y-Only Backbone (Section 2.5) ===
        # Separate MLP 75->256->128->64 + temporal Conv1d
        # === SEPARATE Y-ONLY BACKBONE (fully decoupled from WaveNet) ===
        self.yonly_backbone = nn.Sequential(
            nn.Linear(d_yfeat, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
        )
        # Lightweight causal temporal encoder (own params, NOT shared with WaveNet)
        self.yonly_temporal_inp = nn.Conv1d(d_yfeat, 64, kernel_size=1)
        self.yonly_temporal_convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=3, dilation=d, padding=0)
            for d in [1, 4, 16]
        ])
        self.yonly_temporal_out = nn.Conv1d(64, 64, kernel_size=1)
        # Stage 2b output: MLP(64)+temporal(64)->4 dims, bounded by 2*tanh
        self.head_dx_y_separate = nn.Linear(64 + 64, d_x)
        nn.init.normal_(self.head_dx_y_separate.weight, std=0.01)

        nn.init.zeros_(self.head_dx_y_separate.bias)

        # Gate floor 0.65 (Section 5.8)
        self.register_buffer('g_floor', torch.tensor(0.65))

    # kappa_vec: sigmoid(logits) clamped >= 0.85
    def kappa_vec(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_logits).clamp(min=0.85)

    def forward(
            self, y_feat_only, x_prev=None, b_prev=None, sample=False,
            x_yonly_prop=None, kappa: float | torch.Tensor | None = None,
            innov_prev: torch.Tensor | None = None,  # <--- RENAMED
            return_parts: bool = False,
    ):
        # --- WaveNet forward (Section 2.4) ---
        x = y_feat_only.transpose(1, 2)  # [B,F,T]
        h = self.inp(x)

        # 6 causal dilated blocks
        for blk in self.blocks:
            h = blk(h)

        # Last timestep -> memory z [B,512]
        last = h[:, :, -1:]  # [B,C,1]
        z = self.out(last).squeeze(-1)  # [B,d_mem]

        d1, d4 = None, None

        if innov_prev is not None:
            innov_prev = innov_prev.to(device=z.device, dtype=z.dtype)
            if innov_prev.dim() == 1:
                innov_prev = innov_prev.unsqueeze(-1)  # [B] -> [B,1]
            # innov_embed expects scalar [B,1] — extract latest from buffer
            # Innovation: clamp+/-3, embed to modulate z (Section 2.6)
            innov_scalar = innov_prev[:, -1:] if innov_prev.size(-1) > 1 else innov_prev
            z_innov = z + self.innov_embed(innov_scalar.clamp(-3.0, 3.0))
        else:
            z_innov = z

        # dmu_y from innovation-modulated z; logsig from raw z
        dmu_y_raw = self.head_dx_y(z_innov)  # <--- Only dx head uses innovation
        dls_y_raw = self.head_logsig(z)
        with torch.no_grad():
            # fraction of elements where the pre-tanh magnitude is large (proxy for saturation pressure)
            sat_y = (dmu_y_raw.abs() > _SAT_THRESH).float().mean().item()

        # Bound: 2*tanh -> [-2,+2] sigma
        dmu_y = _DELTA_SCALE * torch.tanh(dmu_y_raw)
        dls_y = _LOGSIG_SCALE * torch.tanh(dls_y_raw)

        # Scale z gradient to protect wavenet backbone (same 0.015 as z_temporal) --->
        # Bias heads get 10%% gradient (Section 3.6)
        _z_bias = z.detach() + 0.10 * (z - z.detach())
        b_hat_raw = self.head_corr(_z_bias)
        g_logits = self.head_corr_gate(_z_bias)

        if b_prev is None:
            b_prev = torch.zeros((z.size(0), self.d_x), device=z.device, dtype=z.dtype)

        b_prev0 = b_prev  # keep for c_t definition below

        # map to bounded bias estimate in same space as x_feat_std
        b_max = 0.3
        # Clamp bias to +/-0.3 sigma
        b_hat = b_hat_raw.clamp(-b_max, b_max)

        # stable "catch-up" update
        # EMA bias update: g in [0.05,0.20], b in [-0.3,0.3]
        b_t, g_b = gated_ema_bias_update(
            b_prev=b_prev0,
            b_hat=b_hat,
            g_logits=g_logits,
            g_min=0.05,
            g_max=0.20,
            b_max=b_max,
        )

        # (now it's the EMA step delta, not an unbounded increment accumulator)
        # Correction increment c(t) = b(t) - b(t-1)
        c_t = b_t - b_prev0

        # std/cent-space magnitudes should be O(1..10), not O(1e3)
        with torch.no_grad():
            if torch.isfinite(c_t).all():
                if c_t.abs().mean().item() > 50:
                    raise RuntimeError("c_t magnitude suggests space mismatch (not in std/cent space).")

        # --- Y-only path (no x_prev): Section 2.5 ---
        sat_ar = 0.0
        if x_prev is None:
            if y_feat_only.dim() == 3:
                y_last = y_feat_only[:, -1, :]
            else:
                y_last = y_feat_only

            # Stage 2b MLP features
            z_yonly_sep = self.yonly_backbone(y_last)

            # Own temporal encoder (fully decoupled from WaveNet backbone)
            if y_feat_only.dim() == 3:
                xt = y_feat_only.transpose(1, 2)  # [B, F, T]
                ht = self.yonly_temporal_inp(xt)
                for conv in self.yonly_temporal_convs:
                    pad = 2 * conv.dilation[0]
                    ht_padded = F.pad(ht, (pad, 0))
                    ht = ht + torch.tanh(conv(ht_padded))
                ht = self.yonly_temporal_out(ht)
                z_yonly_temp = ht[:, :, -1]  # [B, 64]
            else:
                z_yonly_temp = torch.zeros(y_last.size(0), 64, device=y_last.device, dtype=y_last.dtype)

            # Concat MLP(64) + temporal(64)
            z_combined = torch.cat([z_yonly_sep, z_yonly_temp], dim=-1)

            # Stage 2b output: 2*tanh bounded
            dmu_y_sep = _DELTA_SCALE * torch.tanh(self.head_dx_y_separate(z_combined))
            x_prop = dmu_y_sep

            x_logsig = dls_y
            x_mu = x_prop
        else:
            # --- AR path (Section 2.4): x_prev available ---
            # --- DIAG: disable AR(x_prev) ---
            if _DISABLE_AR_DIAG:
                dmu_ar = torch.zeros_like(dmu_y)
                dls_ar = torch.zeros_like(dls_y)
                with torch.no_grad():
                    sat_ar = 0.0
            else:
                ar_params = self.ar(x_prev)
                with torch.no_grad():
                    sat_ar = (ar_params[:, :self.d_x].abs() > _SAT_THRESH).float().mean().item()
                dmu_ar = _DELTA_SCALE * torch.tanh(ar_params[:, :self.d_x])
                dls_ar = _LOGSIG_SCALE * torch.tanh(ar_params[:, self.d_x:])

            # AR dropout: zero 40%% of steps (Section 3.3)
            if self.training and self.ar_dropout_p > 0:
                if torch.rand(()).item() < self.ar_dropout_p:
                    dmu_ar = torch.zeros_like(dmu_ar)
            # --------------------------------
            x_logsig = dls_y + dls_ar

            # <--- REVERTED: Standard proposal first, then gentle kappa mix --->
            # <--- REVERTED: Standard proposal first, then gentle kappa mix --->
            # Innovation direct MLP: buffer+x_prev->dx_innov (Section 2.6)
            dx_innov = torch.zeros_like(dmu_y)
            if innov_prev is not None:
                ip = innov_prev.to(device=dmu_y.device, dtype=dmu_y.dtype)
                if ip.dim() == 1:
                    ip = ip.unsqueeze(-1)
                # --- NEW: Expand scalar to fill buffer if needed ---
                if ip.size(-1) == 1:
                    ip = ip.expand(-1, self.innov_buffer_size)
                # 8-step buffer + detached x_prev -> correction
                innov_input = torch.cat([ip.clamp(-3.0, 3.0), x_prev.detach()], dim=-1)
                dx_innov = self.innov_direct(innov_input)
            # Proposal: x_prev + dmu_y + dmu_AR + b(t) + dx_innov
            x_prop = x_prev + (dmu_y + dmu_ar) + b_t + dx_innov

            if (x_yonly_prop is not None):
                k = self.kappa_vec() if (kappa is None) else kappa
                k = _ensure_kappa_vec_or_batch(k, x_prop=x_prop, D=self.d_x)

                if not torch.is_tensor(k):
                    k = torch.tensor(float(k), device=x_prop.device, dtype=x_prop.dtype)
                if k.dim() == 0:
                    k = k.view(1, 1).expand_as(x_prop)
                elif k.dim() == 1:
                    k = k.view(1, -1).expand_as(x_prop)
                else:
                    k = k.to(device=x_prop.device, dtype=x_prop.dtype)
                    if k.shape[0] == 1 and k.shape[1] == x_prop.shape[1]:
                        k = k.expand_as(x_prop)
                    elif k.shape != x_prop.shape:
                        raise RuntimeError(f"[kappa] expected {tuple(x_prop.shape)} got {tuple(k.shape)}")

                k = k.clamp(0.0, 1.0)
                # Kappa mixing: blend with y-only (Section 2.7)
                x_prop = (1.0 - k) * x_prop + k * x_yonly_prop.detach()

            # ---------------------------------------------------------------------

            # Gate g in [0.65, 1.0] (Section 2.7)
            g_raw = torch.sigmoid(self.gate(z).float())
            g = self.g_floor + (1.0 - self.g_floor) * g_raw
            # WHY floor 0.65: without it gate shuts, model ignores corrections
            if _FORCE_GATE_OPEN:
                g = torch.ones_like(g)

            # STATE UPDATE: x(t) = x(t-1) + g*(proposal - x(t-1))
            x_mu = x_prev + g * (x_prop - x_prev)
        # Safety clamp +/-10 sigma
        x_mu = torch.clamp(x_mu, -10.0, 10.0)

        x_logsig = x_logsig.clamp(-6.0, 2.0)

        if sample:
            x_hat = x_mu + torch.exp(x_logsig) * torch.randn_like(x_mu)
        else:
            x_hat = x_mu

        with torch.no_grad():
            # dx_total and dx_bias are only meaningful when x_prev exists
            if x_prev is not None:
                dx_total = (x_mu - x_prev)
                dx_bias = g * b_t
                dx_total_abs_mean = float(dx_total.detach().abs().mean().item())
                dx_bias_abs_mean = float(dx_bias.detach().abs().mean().item())
                bias_frac = float(dx_bias_abs_mean / max(1e-9, dx_total_abs_mean))
            else:
                dx_total_abs_mean = float("nan")
                dx_bias_abs_mean = float("nan")
                bias_frac = float("nan")

            gb = g_b.detach()
            self.last_diag = {
                "sat_y": float(sat_y),
                "sat_ar": float(sat_ar),
                "ct_abs_mean": float(c_t.detach().abs().mean().item()),
                "bt_abs_mean": float(b_t.detach().abs().mean().item()),
                "gb_mean": float(gb.mean().item()),
                "gb_p10": _pct(gb, 10),
                "gb_p50": _pct(gb, 50),
                "gb_p90": _pct(gb, 90),
                "dx_total_abs_mean": dx_total_abs_mean,
                "dx_bias_abs_mean": dx_bias_abs_mean,
                "bias_fraction": bias_frac,
            }

        # Define a dummy g for the x_prev is None case
        if x_prev is None:
            g = torch.ones_like(x_mu)

        if return_parts:
            dx = (x_mu - x_prev) if (x_prev is not None) else x_mu
            return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, dx

        # CRITICAL FIX: Return x_prop and g as the 9th and 10th elements
        return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, x_prop, g


# ---------------------- GAN Discriminator on X (multi-channel, conditional) ----------------------

# --- GAN discriminator for X states (Section 4.3) ---
class XDiscriminator(nn.Module):
    def __init__(self, in_dim_x=4, in_dim_y=0, hidden=128, depth=3, drop_p=0.10):
        super().__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        d = in_dim_x + in_dim_y
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(drop_p)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, y_cond=None):
        if self.in_dim_y > 0:
            if y_cond is None:
                raise ValueError("y_cond must be provided when in_dim_y > 0")
            if y_cond.dim() == 3:
                y_cond = y_cond[:, -1, :]
            h_in = torch.cat([x, y_cond], dim=-1)
        else:
            h_in = x
        h = self.backbone(h_in)
        logits = self.out(h).squeeze(-1)
        return logits


# --- Moving-average smoothing for teacher labels ---
def _smooth_time(tensor, k=3):
    if k <= 1:
        return tensor
    pad = (k - 1) // 2
    xpad = torch.cat([tensor[0:1].repeat(pad, 1), tensor, tensor[-1:].repeat(pad, 1)], dim=0)
    out = []
    for i in range(tensor.size(0)):
        out.append(xpad[i:i + k].mean(dim=0))
    return torch.stack(out, dim=0)


# --- Per-sample importance weights from Y dynamics ---
def _make_hard_weights_from_y(df, power=0.5, eps=1e-3):
    y = df["ShortChannel"].to_numpy(np.float32)
    y_l1 = lag_np_causal(y, 1)
    y_l2 = lag_np_causal(y, 2)
    dy = y - y_l1
    ddy = y - 2 * y_l1 + y_l2
    s = np.abs(dy) + 0.5 * np.abs(ddy)
    s = (s - s.min()) / max(s.max() - s.min(), 1e-6)
    w = (eps + s) ** power
    return torch.from_numpy(w.astype(np.float32))


# --- Sliding windows from time-series tensor ---
def _make_windows(arr_t, win):
    N = arr_t.size(0)
    if N < win:
        return None, None
    xs = []
    idxs = []
    for t in range(win - 1, N):
        xs.append(arr_t[t - win + 1:t + 1])
        idxs.append(t)
    return torch.stack(xs, dim=0), torch.tensor(idxs, dtype=torch.long)


# ---------------------- FIXED: Y->X training ----------------------

# ============================================================================
# Stage 2: Train WaveNet student (Sections 2.4-2.7, 4.2)
# Phase 2 Stage 1: Y-only pretrain (ep 1-7)
# Phase 2 Stage 2: Closed-loop (ep 8-30)
# Losses: Huber + delta + proposal + gate + obs-y + GAN + teacher
#   + rollout (multi-scale, variance, spectral) + shift-KL + Jacobian
# ============================================================================
def train_y_memory_encoder(model_yx: YGivenXModel,
                           df: pd.DataFrame,
                           scaler_obj: dict,
                           epochs: int = 20,
                           lr: float = 1e-3,
                           batch: int = 256,
                           device: str = "cpu",
                           save_path: str = "./output/y_memory_encoder.pt",
                           w_geom: float = 0.25,
                           w_dyn: float = 0.3,
                           w_var: float = 0.1,
                           w_teacher: float = 1.0,
                           w_obs_y: float = 5.0,
                           cond_gain_for_teacher: float = 20.0,
                           label_smooth_k: int = 1,
                           USE_WAVENET_STUDENT: bool = False,
                           WAVENET_WIN: int = 128,
                           w_gan: float = 0.10,
                           gan_lr: float = 5e-4,
                           gan_label_smooth: float = 0.1,
                           gan_label_noise: float = 0.05,
                           gan_r1_gamma: float = 5.0,
                           eval_every: int = 3,
                           # scheduled sampling knobs
                           ss_p_start: float = 0.0,
                           ss_p_end: float = 0.5,
                           ss_warmup_epochs: int = 1,
                           x_prev_noise_std: float = 0.05,
                           # geometry ramp cap
                           geom_mult_start: float = 0.15,
                           geom_mult_end: float = 0.50,
                           geom_ramp_epochs: int = 6,
                           # diagnostics
                           diag_dir: str = "./output/x_diagnostics",
                           diag_max_points_plot: int = 2000,
                           # Stage-1 controls
                           recon_epochs: int = 3,
                           freeze_sigma_epochs: int = 5,
                           fixed_logsig: float = -2.0,
                           w_mu: float = 100.0,
                           w_delta: float = 5.0,
                           w_drift: float = 2.0,
                           enable_aux_after_recon: bool = True,
                           # bounds
                           clamp_x_rollout: float = 5.0,
                           stage1_min_tf_r2: float = 0.30,
                           # fix pack
                           # NEW: periodic re-anchoring (reset that actually corrects drift)
                           reanchor_every: int = 32,  # 0 disables
                           reanchor_alpha: float = 0.50,  # mix strength into y-only estimate
                           reanchor_reset_b: float = 0.50,  # damp bias state on reanchor (0..1)
                           reanchor_innov_thr: float = 2.0,
                           yonly_pretrain_epochs: int = 20,  # also reanchor when innovation spikes

                           hist_p_start: float = 0.0,
                           hist_p_end: float = 0.6,
                           hist_warmup_epochs: int = 1,
                           w_yonly_start: float = 0.0,
                           w_yonly_end: float = 5.0,
                           yonly_ramp_epochs: int = 8,
                           w_roll: float = 10.0,
                           roll_gamma: float = 1,
                           w_shift: float = 0.05,
                           shift_kl_use_unclamped: bool = True,
                           w_jac: float = 0.02,
                           jac_samples: int = 16,
                           w_bound: float = 2.0,
                           bound_soft: float = 5.0,
                           bound_power: float = 2.0,
                           detach_hist_pred: bool = True,
                           detach_rollout_prev: bool = False,
                           corr_kappa_max: float = 0.3,
                           corr_ramp_epochs: int = 6,
                           aux_start_epoch: int = 5,
                           aux_min_tf_r2: float = 0.50,
                           clamp_x_mode: str = "std",
                           clamp_x_raw_bounds: dict | None = None,
                           # NEW: innovation feedback enable (3.3)
                           use_innovation_feedback: bool = True,
                           w_x: float = 1.0,
                           # NEW: teacher distill on rollout (4.1(d))
                           w_teacher_roll_mult: float = 1.0,
                           # NEW: rollout stochasticity (process noise)
                           rollout_sample_p: float = 0.20,  # probability to sample x_hat instead of x_mu
                           rollout_noise_std: float = 0.02,  # additional Gaussian noise on fed prev

                           ):
    """
    Fix pack includes:
      - Shift-invariant sampling (random contiguous batch segments).
      - Correct std-space mixing for scheduled sampling + history dropout.
      - Explicit drift correction: dx + c_t, with stateful bias b_t.
      - Rollout loss inside objective (discounted).
      - Correction regularizers on c_t (magnitude + smoothness) using rollout sequence.
      - Soft bound penalty (objective), clamp only for feeding.
      - Optional innovation feedback (Kalman-like): innovation = y_obs - y_hat(prev_x).
      - Teacher distillation applied on rollout horizon as well (optional but enabled here).
    """
    print("[mem] === Building geometry-aware Y→X distillation dataset ===")
    DEBUG_PRINTS = True  # master switch
    DEBUG_EARLY_EPS = 1  # treat first N epochs as "early"
    # Detrend ShortChannel: 600-sample moving average (Section 2.1)
    y_raw = df["ShortChannel"].to_numpy(np.float32)
    y_smooth = uniform_filter1d(y_raw, size=600)
    df["ShortChannel"] = y_raw - y_smooth

    # ---------------------- SCHEDULE DEBUG (PATCH) ----------------------
    SCHED_DEBUG = True  # set False to silence later
    # Optional hard-force via env vars (do NOT commit enabled)
    _env_force_ss = os.getenv("FORCE_SS_P", "").strip()
    _env_force_hist = os.getenv("FORCE_HIST_P", "").strip()
    FORCE_SS_P = float(_env_force_ss) if _env_force_ss != "" else None
    FORCE_HIST_P = float(_env_force_hist) if _env_force_hist != "" else None

    if SCHED_DEBUG:
        print(
            "[SCHED][ARGS] "
            f"epochs={epochs} recon_epochs={recon_epochs} freeze_sigma_epochs={freeze_sigma_epochs} | "
            f"ss_p_start={ss_p_start} ss_p_end={ss_p_end} ss_warmup_epochs={ss_warmup_epochs} | "
            f"hist_p_start={hist_p_start} hist_p_end={hist_p_end} hist_warmup_epochs={hist_warmup_epochs} | "
            f"FORCE_SS_P={FORCE_SS_P} FORCE_HIST_P={FORCE_HIST_P}",
            flush=True
        )
    # -------------------------------------------------------------------

    x_norm_mu = np.array(scaler_obj["x_norm_mu"], dtype=np.float32)
    x_norm_sd = np.array(scaler_obj["x_norm_sigma"], dtype=np.float32)

    _, x_feat, _feat_names, _ = build_x_features_from_df(
        df, x_norm_mu=x_norm_mu, x_norm_sd=x_norm_sd, fit_slice=None, save_stats_path=None
    )

    # Standardise X features using Stage 1 scaler
    x_feat_std = apply_scaling_np(
        x_feat,
        np.array(scaler_obj["x_feat_mu"], dtype=np.float32)[:x_feat.shape[1]],
        np.array(scaler_obj["x_feat_sigma"], dtype=np.float32)[:x_feat.shape[1]]
    ).astype(np.float32)

    d_teacher = len(scaler_obj["x_feat_mu"])  # 16 when teacher has lags

    def _pad_for_teacher(x4):
        if x4.shape[-1] == d_teacher:
            return x4
        pad = torch.zeros(*x4.shape[:-1], d_teacher - x4.shape[-1], device=x4.device, dtype=x4.dtype)
        return torch.cat([x4, pad], dim=-1)

    def _np_stat(tag, arr):
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        print(f"[XSTAT] {tag}: mean[:4]={mu[:4]} std[:4]={sd[:4]}")

    _np_stat("x_feat (pre apply_scaling)", x_feat.astype(np.float32))
    _np_stat("x_feat_std (post apply_scaling)", x_feat_std.astype(np.float32))

    # x_prev: shifted sequence x_prev[t]=x[t-1]
    x_prev_std = np.zeros_like(x_feat_std, dtype=np.float32)
    x_prev_std[0] = x_feat_std[0]
    x_prev_std[1:] = x_feat_std[:-1]

    y_scalar = df["ShortChannel"].to_numpy(np.float32).reshape(-1, 1)
    y_mu = np.array(scaler_obj["y_mu"], dtype=np.float32)
    y_sd = np.array(scaler_obj["y_sigma"], dtype=np.float32)
    y_std = apply_scaling_np(y_scalar, y_mu, y_sd).astype(np.float32)

    est_fs = 30.0
    if "Time" in df.columns and len(df) > 5:
        dt = float(np.median(np.diff(df["Time"].to_numpy(np.float32))))
        if dt > 0:
            est_fs = 1.0 / dt

    # Build 75-dim Y-only features (no cardiovascular info)
    y_only, yonly_names = build_y_only_features(df, fs_hint=est_fs)
    y_only_mu, y_only_sd = standardize_train_stats(y_only)
    y_only_std = apply_scaling_np(y_only, y_only_mu, y_only_sd).astype(np.float32)

    x_feat_t_cpu = torch.from_numpy(x_feat_std)
    x_prev_t_cpu = torch.from_numpy(x_prev_std)
    y_only_t_cpu = torch.from_numpy(y_only_std)
    y_std_t_cpu = torch.from_numpy(y_std)
    hard_w_cpu = _make_hard_weights_from_y(df)

    with torch.no_grad():
        X_true_np = x_feat_t_cpu.numpy()
        X_prev_np = x_prev_t_cpu.numpy()
        r2_base, mae_base = _r2_mae_np(X_true_np, X_prev_np)
        corr_dims = []
        for j in range(X_true_np.shape[1]):
            a = X_true_np[1:, j]
            b = X_true_np[:-1, j]
            denom = (a.std() * b.std() + 1e-12)
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / denom)
            corr_dims.append(corr)
        corr_s = ", ".join(f"{c:+.3f}" for c in corr_dims)
        print(f"[diag] Baseline X̂[t]=X[t-1]: R2={r2_base:.3f} MAE={mae_base:.4f} | lag1 corr per-dim: {corr_s}")

    model_yx.eval()
    for p in model_yx.parameters():
        p.requires_grad_(False)

    # === BUILD FULL Teacher Input Array (X_base + Y_bp) ===
    # Extract the exact same Y_bp features used during teacher training
    bp_names = ['Y_bp_energy_HR', 'Y_bp_envelope_HR', 'Y_bp_energy_RR',
                'Y_bp_envelope_RR', 'Y_bp_energy_Trend', 'Y_bp_cardiac_raw']
    bp_indices = [yonly_names.index(n) for n in bp_names if n in yonly_names]
    y_bp_raw = y_only[:, bp_indices]

    x_teacher_raw = np.concatenate([x_feat, y_bp_raw], axis=1)

    # Standardize using the full scaler from YGivenX training
    x_feat_full_std = apply_scaling_np(
        x_teacher_raw,
        np.array(scaler_obj["x_feat_mu"], dtype=np.float32),
        np.array(scaler_obj["x_feat_sigma"], dtype=np.float32)
    ).astype(np.float32)

    x_feat_full_t_cpu = torch.from_numpy(x_feat_full_std)

    # 1. Fix offline embedder to use the FULL 16-dim array
    x_feat_t_dev = x_feat_full_t_cpu.to(device)
    # Fit whitener on frozen teacher taps for distillation
    used_whitener = model_yx.fit_teacher_whitener(x_feat_t_dev, cond_gain_scale=cond_gain_for_teacher)

    with torch.no_grad():
        if used_whitener:
            T_raw_dev = model_yx.get_teacher_embed_from_xfeat(x_feat_t_dev, cond_gain_scale=cond_gain_for_teacher)
            src = getattr(model_yx, "teacher_source_mode", "taps")
            print(f"[mem] Teacher source: {src} (whitened).")
        else:
            T_raw_dev = model_yx.get_teacher_embed(x_feat_t_dev)
            print("[mem] Teacher source: FALLBACK orthogonal projection (no whitener).")

    # 2. Define the new dynamic helper functions
    x_feat_full_t_dev = x_feat_full_t_cpu.to(device)  # Keep on device for fast lookup

    def _teacher_input_from_pred(x4_pred, t_indices):
        if x4_pred.shape[-1] == x_feat_full_t_dev.shape[-1]:
            return x4_pred
        lag_dims = x_feat_full_t_dev[t_indices, 4:]
        if lag_dims.shape[0] != x4_pred.shape[0]:
            lag_dims = lag_dims.expand(x4_pred.shape[0], -1)
        return torch.cat([x4_pred, lag_dims], dim=-1)

    def _teacher_input_from_indices(t_indices):
        return x_feat_full_t_dev[t_indices]

    @torch.no_grad()
    # Innovation: y_obs - y_hat(x_prev) (Section 2.6)
    # Measures if previous state was correct (Kalman analogy)
    def _compute_innovation_scalar(y_obs, x_prev_fed, t_idx_slice, model_yx, cond_gain, teacher_input_fn):
        """
        Returns SCALAR innovation: y_obs - H(x_pred). Shape [B, 1].
        No Jacobian needed — the WaveNet learns the x-space mapping internally.
        """
        teacher_in = teacher_input_fn(x_prev_fed, t_idx_slice)
        y_hat = model_yx(teacher_in, cond_gain_scale=cond_gain)[0]  # [B, 1]
        return (y_obs - y_hat).detach()  # [B, 1]

    def _compute_innov_scalar(y_obs, x_prev_fed, t_idx_slice):
        return _compute_innovation_scalar(
            y_obs, x_prev_fed, t_idx_slice,
            model_yx, cond_gain_for_teacher, _teacher_input_from_pred
        )

    T_raw = T_raw_dev.detach().cpu()
    T_mean = T_raw.mean(dim=0, keepdim=True)
    T_std = T_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    T = (T_raw - T_mean) / T_std
    if label_smooth_k and label_smooth_k > 1:
        T = _smooth_time(T, k=label_smooth_k)

    print(f"[mem] Teacher taps dim: {T.shape[1]} | y-only feat dim: {y_only_t_cpu.shape[1]}")
    idx_center = None
    d_yfeat = y_only_t_cpu.shape[1]
    dim_names = ["DBP", "SBP", "CO", "SV"]

    # Domain-aware clamping: student operates in x_feat_std
    x_feat_mu_4 = torch.tensor(np.array(scaler_obj["x_feat_mu"], dtype=np.float32)[:4], device=device)
    x_feat_sd_4 = torch.tensor(np.array(scaler_obj["x_feat_sigma"], dtype=np.float32)[:4], device=device).clamp_min(
        1e-6)
    x_norm_mu_4 = torch.tensor(np.array(scaler_obj["x_norm_mu"], dtype=np.float32)[:4], device=device)
    x_norm_sd_4 = torch.tensor(np.array(scaler_obj["x_norm_sigma"], dtype=np.float32)[:4], device=device).clamp_min(
        1e-6)

    def _clamp_feed(x: torch.Tensor) -> torch.Tensor:
        if not (clamp_x_rollout and clamp_x_rollout > 0):
            return x
        mode = str(clamp_x_mode).lower().strip()
        if mode == "std":
            return x.clamp(-float(clamp_x_rollout), float(clamp_x_rollout))
        if mode == "raw":
            if clamp_x_raw_bounds is None:
                raise RuntimeError("[CLAMP] clamp_x_mode='raw' requires clamp_x_raw_bounds")
            x_feat = x * x_feat_sd_4 + x_feat_mu_4
            x_raw = x_feat * x_norm_sd_4 + x_norm_mu_4
            lo = []
            hi = []
            for nm in dim_names:
                if nm not in clamp_x_raw_bounds:
                    raise RuntimeError(f"[CLAMP] missing raw bound for dim '{nm}' in clamp_x_raw_bounds")
                lo.append(float(clamp_x_raw_bounds[nm][0]))
                hi.append(float(clamp_x_raw_bounds[nm][1]))
            lo_t = torch.tensor(lo, device=device, dtype=x.dtype).view(1, -1)
            hi_t = torch.tensor(hi, device=device, dtype=x.dtype).view(1, -1)
            x_raw = torch.max(torch.min(x_raw, hi_t), lo_t)
            x_feat = (x_raw - x_norm_mu_4) / x_norm_sd_4
            return (x_feat - x_feat_mu_4) / x_feat_sd_4
        raise RuntimeError(f"[CLAMP] unknown clamp_x_mode='{clamp_x_mode}'")

    if USE_WAVENET_STUDENT:

        # Build student configured with lags
        # Build student: 6 blocks, 112 hidden (Section 2.4)
        student = YMemoryWaveNet(
            d_yfeat=d_yfeat,
            d_mem=512,
            hidden=112,
            layers=6,
            d_x=4,
        ).to(device)

        # Build windows (these index the target time t via idx_center)
        # Sliding windows [N-128+1, 128, 75]
        Xwin_cpu, idx_center = _make_windows(y_only_t_cpu, WAVENET_WIN)
        assert Xwin_cpu is not None, "Sequence shorter than WAVENET_WIN."


        # Targets aligned to idx_center (time t)
        T_win_cpu = T[idx_center]
        W_win_cpu = hard_w_cpu[idx_center]
        X_tgt_cpu = x_feat_t_cpu[idx_center]  # X_t
        Xprev_cpu = x_prev_t_cpu[idx_center]  # X_{t-1} (your existing prev)
        y_std_tgt_cpu = y_std_t_cpu[idx_center]

        # IMPORTANT: dataset order updated (lag tensors included)
        ds = TensorDataset(
            Xwin_cpu, T_win_cpu, X_tgt_cpu, Xprev_cpu,
            y_std_tgt_cpu, W_win_cpu, idx_center
        )

        print(f"[mem] Student=WaveNet | win={WAVENET_WIN} | samples={Xwin_cpu.size(0)}")
    else:
        student = YMemoryMLP(d_yfeat=d_yfeat, d_mem=T.shape[1]).to(device)
        ds = TensorDataset(y_only_t_cpu, T, x_feat_t_cpu, x_prev_t_cpu, y_std_t_cpu, hard_w_cpu)
        print(f"[mem] Student=MLP | samples={y_only_t_cpu.size(0)}")

    disc = XDiscriminator(in_dim_x=4, in_dim_y=d_yfeat, hidden=128, depth=3, drop_p=0.10).to(device)

    # Shift-invariant contiguous random segments (per epoch)
    gen = torch.Generator()
    gen.manual_seed(1234)
    # Shift-invariant sampler: contiguous segments, random start
    batch_sampler = RandomContiguousBatchSampler(n=len(ds), batch_size=batch, drop_last=True, generator=gen)
    dl = DataLoader(ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    dl_eval = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    # <--- FIX 1: Add shuffled pretrain DataLoader --->
    dl_pretrain = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

    # --- SURGICAL FIX: Robust batch unpacker ---
    def _unpack_batch(batch_items, *, device=None):
        """Returns: y_in, Tb, Xb, XprevT, ystd, wb, y_cond, tidx"""
        (xb_win_cpu, Tb_cpu, Xb_cpu, Xprev_cpu,
         ystd_cpu, wb_cpu, tidx_cpu) = batch_items

        if device is None:
            y_in, Tb, Xb, XprevT = xb_win_cpu, Tb_cpu, Xb_cpu, Xprev_cpu
            ystd, wb, tidx = ystd_cpu, wb_cpu, tidx_cpu
        else:
            y_in = xb_win_cpu.to(device, non_blocking=True)
            Tb = Tb_cpu.to(device, non_blocking=True)
            Xb = Xb_cpu.to(device, non_blocking=True)
            XprevT = Xprev_cpu.to(device, non_blocking=True)
            ystd = ystd_cpu.to(device, non_blocking=True)
            wb = wb_cpu.to(device, non_blocking=True)
            tidx = tidx_cpu.to(device, non_blocking=True)

        y_cond = y_in[:, -1, :]
        return y_in, Tb, Xb, XprevT, ystd, wb, y_cond, tidx


    @torch.no_grad()
    def _sat_frac(x: torch.Tensor, clamp_val: float, sat_ratio: float = 0.95) -> torch.Tensor:
        """
        x: [B,D] or [1,D] in std-space
        returns: scalar tensor in [0,1] = fraction of elements near clamp boundary
        """
        if clamp_val is None or clamp_val <= 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        thr = sat_ratio * float(clamp_val)
        return (x.abs() >= thr).float().mean()

    # ---------------------- Long-horizon drift probe (drop-in) ----------------------

    @torch.no_grad()
    def _xprev_rows_clean_mask(
            x_unclamped: torch.Tensor,  # [B,D] in std-space (before clamp)
            x_clamped: torch.Tensor | None,  # [B,D] in std-space (after clamp), can be None
            clamp_val: float | None,
            *,
            abs_mean_thr: float = 2.5,  # per-row mean |x| must be <= this
            abs_max_thr: float = 6.0,  # per-row max |x| must be <= this
            max_sat_frac: float = 0.15,  # if too many elems are near clamp, mark unclean
    ) -> torch.Tensor:
        """
        Returns mask [B,1] (True = clean) to avoid batch-wide disabling.
        This is NOT the same as "standard normal-like"; it's a conservative per-row sanity check.
        """
        # magnitude checks (unclamped)
        row_abs_mean = x_unclamped.abs().mean(dim=1, keepdim=True)  # [B,1]
        row_abs_max = x_unclamped.abs().amax(dim=1, keepdim=True)  # [B,1]
        ok_mag = (row_abs_mean <= abs_mean_thr) & (row_abs_max <= abs_max_thr)

        # clamp saturation checks (clamped)
        if (x_clamped is None) or (clamp_val is None) or (clamp_val <= 0):
            ok_sat = torch.ones_like(ok_mag)
        else:
            thr = 0.95 * float(clamp_val)
            sat = (x_clamped.abs() >= thr).float().mean(dim=1, keepdim=True)  # [B,1]
            ok_sat = (sat <= max_sat_frac)

        return ok_mag & ok_sat

    @torch.no_grad()
    def _closedloop_predprev_clean_frac(
            *,
            student: nn.Module,
            y_seq: torch.Tensor,  # [B,F] or [B,WIN,F]
            xprev0: torch.Tensor,  # [1,D] seed prev (std-space)
            ystd_seq: torch.Tensor | None,  # [B,1] for innovation
            clamp_feed_fn,
            clamp_val: float | None,
            model_yx: nn.Module,
            cond_gain_for_teacher: float,
            kappa_eff: float | torch.Tensor | None,
            innov_on: bool,
            H: int = 16,  # short horizon for ROI
            abs_mean_thr: float = 1.5,
            abs_max_thr: float = 4.0,
            max_sat_frac: float = 0.25,
            t_indices_seq: torch.Tensor | None = None,
    ) -> float:
        """
        Rolls the student CLOSED-LOOP for H steps (prev <- x_mu),
        then computes cleanliness on the *fed prev stream*.
        Returns scalar clean fraction in [0,1].
        """
        B = int(y_seq.size(0))
        H = int(max(2, min(H, B)))
        D = int(xprev0.size(1))

        prev = xprev0.to(device=y_seq.device, dtype=torch.float32)
        prev = clamp_feed_fn(prev) if (clamp_val is not None) else prev
        b_prev = torch.zeros((1, D), device=y_seq.device, dtype=torch.float32)

        prev_stream_unclamped = []
        prev_stream_clamped = []

        for t in range(H):
            prev_fed = clamp_feed_fn(prev) if (clamp_val is not None) else prev

            prev_stream_unclamped.append(prev.detach())
            prev_stream_clamped.append(prev_fed.detach())

            innovation_t = None
            if innov_on and (ystd_seq is not None):
                innovation_t = _compute_innov_scalar(ystd_seq[t:t + 1], prev_fed, t_indices_seq[t:t + 1])
            xy_t = None

            if kappa_eff is not None:
                out_yonly = student(
                    y_seq[t:t + 1],
                    x_prev=None,
                    b_prev=None,
                    sample=False,
                    innov_prev=None
                )
                xy_t = out_yonly[3]

            # per-row kappa slice if kappa_eff is [B,D]
            k_t = kappa_eff
            if torch.is_tensor(k_t) and k_t.dim() == 2:
                k_t = k_t[t:t + 1]

            out = student(
                y_seq[t:t + 1],
                x_prev=prev_fed,
                b_prev=b_prev,
                sample=False,
                x_yonly_prop=xy_t,
                kappa=k_t,
                innov_prev=innovation_t
            )

            xmu = out[3]
            b_prev = out[6].detach()
            prev = xmu.detach()  # CLOSED-LOOP

        X_u = torch.cat(prev_stream_unclamped, dim=0)  # [H,D]
        X_c = torch.cat(prev_stream_clamped, dim=0)  # [H,D]

        ok = _xprev_rows_clean_mask(
            x_unclamped=X_u,
            x_clamped=X_c,
            clamp_val=clamp_val,
            abs_mean_thr=abs_mean_thr,
            abs_max_thr=abs_max_thr,
            max_sat_frac=max_sat_frac,
        )
        return float(ok.float().mean().item())

    # <--- FIX: Removed "innov_proj" from bias_names --->
    bias_names = ("head_corr", "head_corr_gate", "innov_gain_logits", "innov_gate_gain")
    kappa_params = []
    bias_params = []
    yonly_sep_params = []
    innov_params = []  # <--- FIX: New list for innov_proj
    base_params = []

    for n, p in student.named_parameters():
        if 'kappa_logits' in n:
            kappa_params.append(p)
        elif 'yonly_backbone' in n or 'head_dx_y_separate' in n or 'yonly_temporal' in n:
            yonly_sep_params.append(p)
        elif 'innov_embed' in n or 'innov_direct' in n:
            innov_params.append(p)
        elif any(k in n for k in bias_names):
            bias_params.append(p)
        else:
            base_params.append(p)

    # Optimiser (Section 3.6): base 0.5x | bias 1x | innov 1x | 2b 2x | kappa 50x
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr * 0.5, "weight_decay": 1e-4},
            {"params": bias_params, "lr": lr * 1.0, "weight_decay": 0.0},
            {"params": innov_params, "lr": lr * 1.0, "weight_decay": 0.0},
            # <--- FIX 3: Slower LR, 100x stronger weight decay --->
            {"params": yonly_sep_params, "lr": lr * 2.0, "weight_decay": 1e-2},
        # 50x LR: kappa must converge fast or persistence collapses
            {"params": kappa_params, "lr": lr * 50.0, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
    )
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=gan_lr, weight_decay=1e-4)

    print(f"[mem] Student params: {_count_params(student):,} | Discriminator params: {_count_params(disc):,}")
    print(
        f"[mem] GAN config: w_gan={w_gan:.4f} | gan_lr={gan_lr:.2e} | smooth={gan_label_smooth:.2f} noise={gan_label_noise:.2f} r1={gan_r1_gamma:.2f}")

    times_full = df["Time"].to_numpy() if "Time" in df.columns else np.arange(len(df), dtype=np.int64)

    def _lin(a, b, t):
        return float(a + (b - a) * float(min(1.0, max(0.0, t))))

    def _ss_prob(ep):
        if ep <= ss_warmup_epochs:
            return float(ss_p_start)
        t = (ep - ss_warmup_epochs) / max(1, (epochs - ss_warmup_epochs))
        return _lin(ss_p_start, ss_p_end, t)

    def _hist_prob(ep):
        if ep <= hist_warmup_epochs:
            return float(hist_p_start)
        t = (ep - hist_warmup_epochs) / max(1, (epochs - hist_warmup_epochs))
        return _lin(hist_p_start, hist_p_end, t)

    def _yonly_weight(ep):
        if ep <= (recon_epochs + yonly_pretrain_epochs):
            return 100.0
        elif ep <= (recon_epochs + yonly_pretrain_epochs + 7):
            return 10.0
        else:
            return 1.0

    def _geom_eff(ep):
        if geom_ramp_epochs <= 1:
            mult = geom_mult_end
        else:
            t = float(min(1.0, max(0.0, ep / float(geom_ramp_epochs))))
            mult = float(geom_mult_start + (geom_mult_end - geom_mult_start) * t)
        return float(w_geom * mult), float(mult)

    def _w_huber(x_mu, x_true, w, beta=0.5):
        per = F.smooth_l1_loss(x_mu, x_true, beta=beta, reduction="none").sum(dim=1)
        return (per * w).mean()

    def _w_mse(x_hat, x_true, w):
        per = F.mse_loss(x_hat, x_true, reduction="none").sum(dim=1)
        return (per * w).mean()

    def _rollout_k(ep: int, Bsz: int) -> int:
        """
        Curriculum + random horizon length (prevents overfitting one K).
        NOTE: capped by Bsz because you unroll along batch/time axis.
        """
        # 10 → 50 → 200 → 1000 style curriculum (mapped onto epochs)
        frac = (ep - 1) / float(max(1, epochs - 1))
        if frac < 0.25:
            k_max = int(round(10 + (50 - 10) * (frac / 0.25)))
        elif frac < 0.50:
            k_max = int(round(50 + (200 - 50) * ((frac - 0.25) / 0.25)))
        elif frac < 0.75:
            k_max = int(round(200 + (1000 - 200) * ((frac - 0.50) / 0.25)))
        else:
            k_max = 1000

        k_max = int(max(2, min(k_max, Bsz)))

        # random horizon in [ceil(0.6*k_max), k_max]
        k_min = int(max(2, math.ceil(0.6 * k_max)))
        if k_min >= k_max:
            return min(k_max, 150)
        k = int(torch.randint(low=k_min, high=k_max + 1, size=(1,), device=device).item())
        k = min(k, 150)
        if last_full_roll_r2 < -1.0:
            k = min(k, 40)
        elif last_full_roll_r2 < 0.0:
            k = min(k, 80)
        return k

    def _corr_kappa(ep: int, stage1: bool, w_yonly_now: float) -> float:
        if stage1 or corr_kappa_max <= 0:
            return 0.0
        if ep <= (recon_epochs + yonly_pretrain_epochs):
            return 0.0
        # Ramp kappa from 0.1 to corr_kappa_max over corr_ramp_epochs after pretrain
        eps_since = ep - (recon_epochs + yonly_pretrain_epochs)
        if corr_ramp_epochs <= 1:
            return float(corr_kappa_max)
        t = min(1.0, eps_since / float(corr_ramp_epochs))
        return 0.1 + (float(corr_kappa_max) - 0.1) * t

        # --- High-ROI drift regularizers (rollout-unroll) ---

    w_ct_dc = 1.0  # penalize DC component in correction increments c_t across rollout horizon
    w_b_state = 5.0  # penalize large bias-state magnitude b_t across rollout horizon

    tf_r2_prev = -1e9
    last_full_roll_r2 = -1e9

    def _student_forward_consistent(y_in, x_prev, kappa_corr: float, innov_prev=None):
        # Build an effective kappa (works for both MLP and WaveNet)
        if hasattr(student, "kappa_vec"):
            kappa_eff_local = float(kappa_corr) * student.kappa_vec().view(1, -1)
        else:
            kappa_eff_local = float(kappa_corr)

        xy = None
        if (float(kappa_corr) > 0.0):
            out_yonly = student(
                y_in,
                x_prev=None,
                b_prev=None,
                sample=False,
                innov_prev=None,
            )
            xy = out_yonly[3]

        return student(
            y_in,
            x_prev=x_prev,
            b_prev=None,
            sample=False,
            x_yonly_prop=xy,
            kappa=kappa_eff_local,
            innov_prev=innov_prev,
        )

    def _tf_forward_stateful_b(
            y_seq: torch.Tensor,  # [B,F] or [B,WIN,F]
            xprev_seq: torch.Tensor,  # [B,D] (already clamped feed-space)
            ystd_seq: torch.Tensor | None,  # [B,1] (for innovation)
            *,
            kappa_eff: float | torch.Tensor | None,
            use_innov: bool,
            detach_b: bool = False,
            t_indices_seq: torch.Tensor | None = None,

    ):
        """
        Teacher-forced roll-in (uses provided xprev_seq per row) but with STATEFUL b_prev carried across rows.
        This removes the TF-vs-export mismatch on the bias integrator.
        """
        B = int(xprev_seq.size(0))
        D = int(xprev_seq.size(1))
        b_prev = torch.zeros((1, D), device=xprev_seq.device, dtype=xprev_seq.dtype)

        mu_list, logsig_list, b_list, c_list, z_list = [], [], [], [], []

        for t in range(B):
            yi = y_seq[t:t + 1]
            xprev_t = xprev_seq[t:t + 1]

            innovation_t = None
            if use_innov and (ystd_seq is not None):
                innovation_t = _compute_innov_scalar(ystd_seq[t:t + 1], xprev_t, t_indices_seq[t:t + 1])

            xy_t = None
            if kappa_eff is not None:
                # y-only proposal must NOT use innovation
                out_yonly = student(
                    yi,
                    x_prev=None,
                    b_prev=None,
                    sample=False,
                    innov_prev=None
                )

                xy_t = out_yonly[3]

            # per-row kappa slice if tensor
            k_t = kappa_eff
            if torch.is_tensor(k_t) and k_t.dim() == 2:
                k_t = k_t[t:t + 1]

            out_t = student(
                yi,
                x_prev=xprev_t,
                b_prev=b_prev,
                sample=False,
                x_yonly_prop=xy_t,
                kappa=k_t,
                innov_prev=innovation_t
            )

            z_list.append(out_t[0])
            mu_list.append(out_t[3])
            logsig_list.append(out_t[4])
            b_list.append(out_t[6])
            c_list.append(out_t[7])

            b_prev = out_t[6]
            if detach_b:
                b_prev = b_prev.detach()

        return (
            torch.cat(z_list, dim=0),
            torch.cat(mu_list, dim=0),
            torch.cat(logsig_list, dim=0),
            torch.cat(b_list, dim=0),
            torch.cat(c_list, dim=0),
        )

    def _rollin_forward_stateful_b(
            y_seq: torch.Tensor,  # [B,F] or [B,WIN,F]
            xprev0: torch.Tensor,  # [1,D] initial prev (feed space)
            x_true_seq: torch.Tensor,  # [B,D] true X in std-space
            ystd_seq: torch.Tensor | None,  # [B,1]
            *,
            kappa_eff: float | torch.Tensor | None,
            use_innov: bool,
            ss_p_eff: float,
            process_noise_std: float,
            t_indices_seq: torch.Tensor | None = None,
    ):
        """
        CLOSED-LOOP ROLL-IN:
          prev_{t+1} = x_hat_t with prob ss_p_eff else x_true_t
        Carries b_prev statefully across time so training matches export behavior.
        Returns:
          z_seq, xmu_seq, logsig_seq, b_seq, c_seq, prevfed_seq, ss_use_rate
        """
        B = int(x_true_seq.size(0))
        D = int(x_true_seq.size(1))

        prev = xprev0.to(device=device, dtype=torch.float32)
        prev = _clamp_feed(prev) if (clamp_x_rollout and clamp_x_rollout > 0) else prev

        b_prev = torch.zeros((1, D), device=device, dtype=torch.float32)

        z_list, mu_list, ls_list, b_list, c_list, prevfed_list = [], [], [], [], [], []
        xprop_list, g_list = [], []  # <--- NEW LISTS

        ss_eligible = 0
        ss_use = 0
        is_pred = False
        for t in range(B):
            yi = y_seq[t:t + 1]

            prev_fed = _clamp_feed(prev) if (clamp_x_rollout and clamp_x_rollout > 0) else prev
            prevfed_list.append(prev_fed)

            innovation_t = None
            if use_innov and (ystd_seq is not None):
                innovation_t = _compute_innov_scalar(ystd_seq[t:t + 1], prev_fed, t_indices_seq[t:t + 1])

            xy_t = None
            if kappa_eff is not None:
                out_yonly = student(
                    yi,
                    x_prev=None,
                    b_prev=None,
                    sample=False,
                    innov_prev=None
                )

                xy_t = out_yonly[3]

            if is_pred:
                k_t = kappa_eff  # predicted prev might drift → use full kappa
            else:
                k_t = 0.0  # teacher prev is truth → no correction needed
                xy_t = None  # ← force fallback dynamics path so AR/gate/bias get gradients

            if torch.is_tensor(k_t) and k_t.dim() == 2:
                k_t = k_t[t:t + 1]

            out = student(
                yi,
                x_prev=prev_fed,
                b_prev=b_prev,
                sample=False,
                x_yonly_prop=xy_t,
                kappa=k_t,
                innov_prev=innovation_t
            )
            z_list.append(out[0])
            mu_list.append(out[3])
            ls_list.append(out[4])
            b_list.append(out[6])
            c_list.append(out[7])
            xprop_list.append(out[8])  # <--- CAPTURE x_prop
            g_list.append(out[9])

            # advance bias state
            b_prev = out[6]

            # choose next prev: predicted vs teacher
            if t < (B - 1):
                ss_eligible += 1
                use_pred = (torch.rand((), device=device).item() < float(ss_p_eff))
                ss_use += int(use_pred)

                # <--- FIX 3: Store decision for the next timestep's kappa toggle --->
                is_pred = use_pred

                if use_pred:
                    nxt = out[3].detach() if detach_rollout_prev else out[3]
                    if (process_noise_std is not None) and (process_noise_std > 0.0):
                        nxt = nxt + float(process_noise_std) * torch.randn_like(nxt)
                    prev = nxt
                else:
                    # teacher: true state at t becomes prev for t+1
                    prev = x_true_seq[t:t + 1].detach()

        z_seq = torch.cat(z_list, dim=0)
        xmu_seq = torch.cat(mu_list, dim=0)
        logsig_seq = torch.cat(ls_list, dim=0)
        b_seq = torch.cat(b_list, dim=0)
        c_seq = torch.cat(c_list, dim=0)
        prevfed_seq = torch.cat(prevfed_list, dim=0)
        xprop_seq = torch.cat(xprop_list, dim=0)  # <--- ADDED THIS
        g_seq = torch.cat(g_list, dim=0)  # <--- ADDED THIS

        ss_use_rate = float(ss_use / max(1, ss_eligible))
        # CRITICAL FIX: Return the pre-gate proposal and gate values
        return z_seq, xmu_seq, logsig_seq, b_seq, c_seq, prevfed_seq, ss_use_rate, xprop_seq, g_seq

    @torch.no_grad()
    def _tf_metrics_small(kappa_corr: float):
        innov_on = _innov_enabled(use_innovation_feedback, _DISABLE_INNOV_DIAG)

        eval_batches = min(5, len(dl_eval))
        X_list, Xmu_list = [], []
        for ei, batch_items in enumerate(dl_eval, start=1):
            if ei > eval_batches:
                break
            y_in, Tb, Xb, XprevT, ystd, wb, y_cond, t_indices = _unpack_batch(batch_items, device=device)

            innovation = None
            if innov_on:
                innovation = _compute_innov_scalar(ystd, XprevT, t_indices)

            out = _student_forward_consistent(
                y_in, XprevT, 0.0,
                innov_prev=innovation,
            )

            Xmu_b = out[3]
            X_list.append(Xb.cpu())
            Xmu_list.append(Xmu_b.cpu())

        X_eval = torch.cat(X_list, dim=0).numpy()
        X_hat = torch.cat(Xmu_list, dim=0).numpy()
        return _r2_mae_np(X_eval, X_hat)

    @torch.no_grad()
    def _collect_tf_and_roll_for_plots(
            dl_source,
            kappa_corr: float,
            *,
            rollout_streams: int = 1,
            use_amp: bool = True,
            progress_every: int = 10,
            max_batches: int | None = None,
    ):
        """
        Returns time-ordered arrays over dl_source:
          X_true, X_tf (teacher forcing), X_roll (closed-loop rollout)
        """

        student.eval()
        innov_on = _innov_enabled(use_innovation_feedback, _DISABLE_INNOV_DIAG)

        innov_tracker_roll = InnovTracker()

        X_true_all = []
        X_tf_all = []
        X_roll_all = []

        last_prev_roll = None  # carry across batches for continuity

        # Compute compound kappa matching training
        if hasattr(student, 'kappa_vec') and kappa_corr > 0.0:
            kappa_eff = float(kappa_corr) * student.kappa_vec().view(1, -1)
        else:
            kappa_eff = float(kappa_corr)

        for bi_eval, batch_items in enumerate(dl_source, start=1):
            if (max_batches is not None) and (bi_eval > max_batches):
                break
            if (progress_every is not None) and (progress_every > 0):
                if (bi_eval == 1) or (bi_eval % progress_every == 0):
                    print(f"[eval] batch {bi_eval}/{len(dl_source)} (max_batches={max_batches})")

            y_in, Tb, X_true, XprevT, ystd, wb, y_cond, t_indices = _unpack_batch(batch_items,
                                                                                                        device=device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if (use_amp and torch.cuda.is_available() and (str(device).startswith("cuda") or device == "cuda"))
                else nullcontext()
            )

            # --- Teacher forcing prediction (use true prev) ---
            with amp_ctx:
                innovation_tf = None
                if innov_on:
                    innovation_tf = _compute_innov_scalar(ystd, _clamp_feed(XprevT), t_indices)

                out_tf = _student_forward_consistent(
                    y_in,
                    _clamp_feed(XprevT),
                    0.0,
                    innov_prev=innovation_tf,
                )

                X_tf = out_tf[3]

            # --- Rollout prediction (closed-loop) ---
            Bsz = X_true.size(0)
            X_roll = torch.empty_like(X_true)

            # Choose number of parallel streams
            S = int(max(1, min(int(rollout_streams), Bsz)))
            L = int(Bsz // S)  # steps per stream
            T_use = S * L  # number of samples we will rollout with parallel streams

            if T_use == 0:
                X_roll.copy_(X_true)
            else:
                if USE_WAVENET_STUDENT:
                    y_seq = y_in[:T_use].view(S, L, *y_in.shape[1:])
                else:
                    y_seq = y_in[:T_use].view(S, L, *y_in.shape[1:])

                Xprev_seq = XprevT[:T_use].view(S, L, X_true.size(1))
                ystd_seq = ystd[:T_use].view(S, L, 1)

                prev = Xprev_seq[:, 0, :]  # [S, d_x]

                if last_prev_roll is not None:
                    prev0 = last_prev_roll.squeeze(0) if last_prev_roll.dim() == 2 else last_prev_roll
                    prev = prev.clone()
                    prev[0:1] = prev0.view(1, -1)

                prev = _clamp_feed(prev)

                b_prev = torch.zeros((S, X_true.size(1)), device=device, dtype=X_true.dtype)
                mu_roll = torch.empty((S, L, X_true.size(1)), device=device, dtype=X_true.dtype)
                innov_buffer_eval = torch.zeros((S, 8), device=device, dtype=X_true.dtype)

                with amp_ctx:
                    for t in range(L):
                        yi_t = y_seq[:, t]
                        yobs_t = ystd_seq[:, t]

                        innovation_t = None
                        if innov_on:
                            innovation_t = _compute_innov_scalar(yobs_t, prev, t_indices[t:t + 1])

                            innov_tracker_roll.record(innovation_t)
                            if innovation_t is not None:
                                innov_buffer_eval = torch.cat([innov_buffer_eval[:, 1:], innovation_t.view(S, 1)],
                                                              dim=1)

                        xy_t = None
                        if kappa_corr > 0.0:
                            out_yonly = student(
                                yi_t,
                                x_prev=None,
                                b_prev=None,
                                sample=False,
                                innov_prev=None
                            )

                            xy_t = out_yonly[3]

                        out_t = student(
                            yi_t,
                            x_prev=prev,
                            b_prev=b_prev,
                            sample=False,
                            x_yonly_prop=xy_t,
                            kappa=kappa_eff,
                            innov_prev=innov_buffer_eval
                        )

                        xmu_t = out_t[3]
                        b_prev = out_t[6]

                        # === WARM-START ===
                        warmup_steps_eval = 32
                        if t < warmup_steps_eval:
                            # Use true state during warmup, but keep model's b_prev
                            prev = _clamp_feed(Xprev_seq[:, min(t + 1, L - 1), :])
                        else:
                            prev = _clamp_feed(xmu_t)

                        mu_roll[:, t, :] = xmu_t

                X_roll[:T_use] = mu_roll.reshape(T_use, X_true.size(1))

                if T_use < Bsz:
                    prev_tail = _clamp_feed(XprevT[T_use:T_use + 1])
                    b_tail = torch.zeros((1, X_true.size(1)), device=device, dtype=X_true.dtype)
                    innov_buffer_tail = torch.zeros((1, 8), device=device, dtype=X_true.dtype)

                    for i in range(T_use, Bsz):
                        yi = y_in[i:i + 1]

                        innovation_i = None
                        if innov_on:
                            y_obs_i = ystd[i:i + 1]
                            with torch.no_grad():
                                y_prev_hat_i, *_ = model_yx(_teacher_input_from_pred(prev_tail, t_indices[i:i + 1]),
                                                            cond_gain_scale=cond_gain_for_teacher)
                            innovation_i = (y_obs_i - y_prev_hat_i).detach()
                            if innovation_i is not None:
                                innov_buffer_tail = torch.cat([innov_buffer_tail[:, 1:], innovation_i.view(1, 1)],
                                                              dim=1)

                        xy_i = None
                        if kappa_corr > 0.0:

                            out_yonly = student(
                                yi,
                                x_prev=None,
                                b_prev=None,
                                sample=False,
                                innov_prev=None,
                            )
                            xy_i = out_yonly[3]


                        out_i = student(
                            yi,
                            x_prev=prev_tail,
                            b_prev=b_tail,
                            sample=False,
                            x_yonly_prop=xy_i,
                            kappa=float(kappa_corr),
                            innov_prev=innov_buffer_tail,
                        )

                        xmu_i = out_i[3]
                        b_tail = out_i[6]
                        prev_tail = _clamp_feed(xmu_i)
                        X_roll[i:i + 1] = xmu_i

                last_prev_roll = _clamp_feed(mu_roll[0, -1, :].view(1, -1)).detach()

            # Keep these append lines exactly
            X_true_all.append(X_true.detach().cpu())
            X_tf_all.append(X_tf.detach().cpu())
            X_roll_all.append(X_roll.detach().cpu())

        X_true_np = torch.cat(X_true_all, dim=0).numpy()
        X_tf_np = torch.cat(X_tf_all, dim=0).numpy()
        X_roll_np = torch.cat(X_roll_all, dim=0).numpy()

        # ONE transfer back to CPU (fast)
        innov_tracker_roll.print_summary(tag=f" [EVAL_ROLL]")
        return X_true_np, X_tf_np, X_roll_np

    @torch.no_grad()
    def _collect_yonly_diagnostic(dl_source):
        student.eval()
        X_true_all, X_yonly_all = [], []
        for bi_eval, batch_items in enumerate(dl_source, start=1):
            y_in, Tb, X_true, XprevT, ystd, wb, y_cond, t_indices = _unpack_batch(batch_items,
                                                                                                        device=device)
            out_yonly = student(y_in, x_prev=None, b_prev=None, sample=False, innov_prev=None)
            X_true_all.append(X_true.detach().cpu())
            X_yonly_all.append(out_yonly[3].detach().cpu())
        return torch.cat(X_true_all, 0).numpy(), torch.cat(X_yonly_all, 0).numpy()

    @torch.no_grad()
    def _print_yonly_diagnostic(X_true_np, X_yonly_np, ep):
        r2_all, mae_all = _r2_mae_np(X_true_np, X_yonly_np)
        print(f"\n{'=' * 80}")
        print(f"[YONLY_DIAG] ep={ep:02d} Overall: R2={r2_all:+.4f} MAE={mae_all:.4f}")
        for j, nm in enumerate(["DBP", "SBP", "CO", "SV"]):
            r2_j, mae_j = _r2_mae_np(X_true_np[:, j], X_yonly_np[:, j])
            std_t = float(np.std(X_true_np[:, j]))
            std_y = float(np.std(X_yonly_np[:, j]))
            vr = (std_y ** 2) / max(std_t ** 2, 1e-12)
            tag = "OSCILLATES" if vr > 0.25 else "FLAT"
            print(
                f"[YONLY_DIAG]   {nm:6s}: R2={r2_j:+.4f} std_true={std_t:.3f} std_yonly={std_y:.3f} var_ratio={vr:.3f} => {tag}")
        print(f"{'=' * 80}\n")

        # ------------------------------------------------------------------

    # Long-horizon drift probe (single-stream; horizon-by-horizon metrics)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Long-horizon drift probe (single-stream; horizon-by-horizon metrics)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _gather_eval_sequences(dl_source, max_steps: int | None = 20000):
        """
        Returns time-ordered tensors on CPU:
          y_in_cpu, X_true_cpu, X_prev_tf_cpu, y_std_cpu, t_indices_cpu
        Works for both WaveNet and MLP student datasets.
        """
        ys, xs, xprevs, ystds, tidxs = [], [], [], [], []

        n = 0
        for batch_items in dl_source:
            y_in, Tb, Xb, XprevT, ystd, wb, y_cond, t_indices = _unpack_batch(batch_items, device=None)

            ys.append(y_in.cpu())
            xs.append(Xb.cpu())
            xprevs.append(XprevT.cpu())
            ystds.append(ystd.cpu())
            tidxs.append(t_indices.cpu())

            n += int(Xb.size(0))
            if (max_steps is not None) and (n >= int(max_steps)):
                break

        y_in_cpu = torch.cat(ys, dim=0)
        X_true_cpu = torch.cat(xs, dim=0)
        X_prev_tf_cpu = torch.cat(xprevs, dim=0)
        y_std_cpu = torch.cat(ystds, dim=0)
        t_indices_cpu = torch.cat(tidxs, dim=0)

        if (max_steps is not None) and (y_in_cpu.size(0) > max_steps):
            y_in_cpu = y_in_cpu[:max_steps]
            X_true_cpu = X_true_cpu[:max_steps]
            X_prev_tf_cpu = X_prev_tf_cpu[:max_steps]
            y_std_cpu = y_std_cpu[:max_steps]
            t_indices_cpu = t_indices_cpu[:max_steps]

        return y_in_cpu, X_true_cpu, X_prev_tf_cpu, y_std_cpu, t_indices_cpu

    @torch.no_grad()
    def _rollout_single_stream(
            y_in_cpu: torch.Tensor,  # [N,F] or [N,WIN,F]
            X_prev_tf_cpu: torch.Tensor,  # [N,d]
            y_std_cpu: torch.Tensor,  # [N,1]
            *,
            kappa_corr: float,
            use_innov: bool,
            b_mode: str,  # "stateful" | "reset" | "no_rho"
            clamp_val: float | None,
            max_steps: int | None = None,
            use_amp: bool = True,
            t_indices_cpu: torch.Tensor | None = None,
            warmup_steps: int = 0,
            X_true_cpu: torch.Tensor | None = None,
    ):
        """
        Produces a true single-stream closed-loop rollout over N steps.
        """
        student.eval()
        N = int(y_in_cpu.size(0)) if max_steps is None else int(min(int(max_steps), int(y_in_cpu.size(0))))
        d_x = int(X_prev_tf_cpu.size(1))

        # seed prev from TF prev of step 0 (this is the correct “start of stream” in training)
        prev = X_prev_tf_cpu[0:1].to(device=device, dtype=torch.float32)
        if (clamp_val is not None) and (clamp_val > 0):
            prev = prev.clamp(-float(clamp_val), float(clamp_val))

        b_prev = torch.zeros((1, d_x), device=device, dtype=torch.float32)

        X_mu = torch.empty((N, d_x), device=device, dtype=torch.float32)
        B_mu = torch.empty((N, d_x), device=device, dtype=torch.float32)
        C_mu = torch.empty((N, d_x), device=device, dtype=torch.float32)
        prevfed = torch.empty((N, d_x), device=device, dtype=torch.float32)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
            else nullcontext()
        )

        innov_tracker = InnovTracker()
        innov_buffer = torch.zeros((1, 8), device=device, dtype=torch.float32)

        # <--- FIX 2A: Compute compound kappa matching training --->
        if hasattr(student, 'kappa_vec') and kappa_corr > 0.0:
            kappa_eff_probe = float(kappa_corr) * student.kappa_vec().view(1, -1)
        else:
            kappa_eff_probe = float(kappa_corr)

        for t in range(N):
            # clamp only for feeding
            prev_fed = prev
            if (clamp_val is not None) and (clamp_val > 0):
                prev_fed = prev_fed.clamp(-float(clamp_val), float(clamp_val))
            prevfed[t] = prev_fed.squeeze(0)

            # innovation = y_obs - y_hat(prev)
            innovation = None
            if use_innov:
                innovation = _compute_innov_scalar(
                    y_std_cpu[t:t + 1].to(device=device, dtype=torch.float32),
                    prev_fed,
                    t_indices_cpu[t:t + 1].to(device=device)
                )
                innov_tracker.record(innovation)
                if innovation is not None:
                    innov_buffer = torch.cat([innov_buffer[:, 1:], innovation.view(1, 1)], dim=1)

            y_t = y_in_cpu[t:t + 1].to(device=device, dtype=torch.float32)

            # y-only proposal if correction mixing is enabled
            xy = None
            if kappa_corr > 0.0:
                with amp_ctx:
                    out_yonly = student(
                        y_t, x_prev=None, b_prev=None, sample=False, innov_prev=None
                    )
                xy = out_yonly[3]

            # choose b_prev behavior
            if b_mode == "reset":
                b_in = None
            elif b_mode == "no_rho":
                b_in = b_prev * 0.0
            else:
                b_in = b_prev

            with amp_ctx:
                out = student(
                    y_t,
                    x_prev=prev_fed,
                    b_prev=b_in,
                    sample=False,
                    x_yonly_prop=xy,
                    kappa=kappa_eff_probe,
                    innov_prev=innov_buffer
                )

            xmu = out[3]
            b_next = out[6]
            c_t = out[7]

            X_mu[t] = xmu.squeeze(0)
            B_mu[t] = b_next.squeeze(0)
            C_mu[t] = c_t.squeeze(0)

            # === WARM-START: use true state during warmup ===
            if (warmup_steps > 0) and (t < warmup_steps) and (X_true_cpu is not None):
                prev = X_true_cpu[t:t + 1].to(device=device, dtype=torch.float32)
                # Keep b_prev from model (it's learning the bias regime)
            else:
                prev = xmu  # closed-loop
            b_prev = b_next

        innov_tracker.print_summary(tag=f" [SINGLE_STREAM]")

        return X_mu.cpu(), B_mu.cpu(), C_mu.cpu(), prevfed.cpu()

    def _mae_r2_on_prefixes(X_true_cpu, X_pred_cpu, horizons=(32, 128, 512, 2048)):
        """
        Returns dict horizon -> (R2, MAE) for first H and last H.
        """
        out = {}
        X_true_np = X_true_cpu.numpy()
        X_pred_np = X_pred_cpu.numpy()
        N = int(X_true_cpu.size(0))
        for H in horizons:
            H = int(min(int(H), N))
            if H < 2:
                continue
            r2_h, mae_h = _r2_mae_np(X_true_np[:H], X_pred_np[:H])
            r2_t, mae_t = _r2_mae_np(X_true_np[-H:], X_pred_np[-H:])
            out[H] = (float(r2_h), float(mae_h), float(r2_t), float(mae_t))
        return out

    def _print_horizon_table(tag: str, metrics: dict):
        items = sorted(metrics.items(), key=lambda kv: kv[0])
        for H, (r2_h, mae_h, r2_t, mae_t) in items:
            print(
                f"[DRIFT][{tag}] H={H:5d} | head: R2={r2_h:+.4f} MAE={mae_h:.4f} | tail: R2={r2_t:+.4f} MAE={mae_t:.4f}",
                flush=True)

    @torch.no_grad()
    def _drift_probe_report(ep: int, kappa_corr: float, horizons=(32, 128, 512, 2048), max_steps=3000):
        """
        Runs ablations that decisively tell you whether drift is:
          - long-horizon instability (error grows with H),
          - integrator-driven (stateful b_prev is the differentiator),
          - innovation-driven (innov on/off changes drift),
          - or something else.
        """
        clamp_val = float(clamp_x_rollout) if (clamp_x_rollout and clamp_x_rollout > 0) else None
        innov_on = _innov_enabled(use_innovation_feedback, _DISABLE_INNOV_DIAG)

        # Unpack all 7 return values (including lags and indices)
        y_in_cpu, X_true_cpu, X_prev_tf_cpu, y_std_cpu, t_indices_cpu = _gather_eval_sequences(dl_eval,
                                                                                               max_steps=int(max_steps))

        # Variant A: true export-like rollout (stateful b_prev, innovation as configured)
        X_a, B_a, C_a, prevfed_a = _rollout_single_stream(
            y_in_cpu, X_prev_tf_cpu, y_std_cpu,
            kappa_corr=float(kappa_corr),
            use_innov=bool(innov_on),
            b_mode="stateful",
            clamp_val=clamp_val,
            max_steps=int(max_steps),
            use_amp=True,
            t_indices_cpu=t_indices_cpu,
            warmup_steps=32,
            X_true_cpu=X_true_cpu[:max_steps]
        )
        m_a = _mae_r2_on_prefixes(X_true_cpu, X_a, horizons=horizons)
        _print_horizon_table(f"ep{ep:02d}:A_stateful_innov{int(innov_on)}", m_a)

        # Variant D: innovation OFF (only if innovation is currently enabled)
        if innov_on:
            X_d, B_d, C_d, prevfed_d = _rollout_single_stream(
                y_in_cpu, X_prev_tf_cpu, y_std_cpu,
                kappa_corr=float(kappa_corr),
                use_innov=False,
                b_mode="stateful",
                clamp_val=clamp_val,
                max_steps=int(max_steps),
                use_amp=True,
                t_indices_cpu=t_indices_cpu,
                warmup_steps=32,
                X_true_cpu=X_true_cpu[:max_steps]
            )
            m_d = _mae_r2_on_prefixes(X_true_cpu, X_d, horizons=horizons)
            _print_horizon_table(f"ep{ep:02d}:D_stateful_innov0", m_d)

        # High-signal summaries
        with torch.no_grad():
            print(
                f"[DRIFT][SUM] ep={ep:02d} | "
                f"mean|c| (A)={C_a.abs().mean().item():.5f} mean|b| (A)={B_a.abs().mean().item():.5f} | ",
                # f"mean|c| (B)={C_b.abs().mean().item():.5f} mean|b| (B)={B_b.abs().mean().item():.5f}",
                flush=True
            )

    last_prev_train = None

    for ep in range(1, epochs + 1):
        last_prev_train = None

        # Logic: Eval on Epoch 1, then every N epochs, and always on the final epoch

        t_ep0 = time.time()

        # epoch-gated recon
        # --- Stage determination (Section 4.2) ---
        stage1_by_epoch = (ep <= int(recon_epochs))
        stage1 = bool(stage1_by_epoch)

        stage_freeze_sigma = bool(ep <= freeze_sigma_epochs)
        yonly_pretrain = bool((not stage1) and (ep <= recon_epochs + yonly_pretrain_epochs))

        # NEW: delay bias activation so backbone learns dynamics first
        freeze_bias = bool((not stage1) and (not yonly_pretrain) and (ep <= recon_epochs + yonly_pretrain_epochs + 1))
        if freeze_bias:
            for pn, p in student.named_parameters():
                if any(k in pn for k in ("head_corr", "head_corr_gate")):
                    p.requires_grad_(False)
        else:
            for pn, p in student.named_parameters():
                if any(k in pn for k in ("head_corr", "head_corr_gate")):
                    p.requires_grad_(True)

        # <--- FIX 2 & 4: Set Active DataLoader and Batch Total --->
        dl_active = dl_pretrain if yonly_pretrain else dl
        batches_total = len(dl_active)

        ss_p = 0.0 if stage1 else _ss_prob(ep)
        hist_p = 0.0 if stage1 else _hist_prob(ep)
        w_yonly = 0.0 if stage1 else _yonly_weight(ep)

        # ---------------------- SCHEDULE PROVENANCE (PATCH) ----------------------
        if SCHED_DEBUG and ((ep <= 5) or (ep % 5 == 0)):
            # Recompute the raw schedule outputs explicitly (without stage1 gate)
            ss_base = float(_ss_prob(ep))
            hist_base = float(_hist_prob(ep))

            # Also show the exact normalized t used inside the schedule
            if ep <= ss_warmup_epochs:
                ss_t = None
            else:
                ss_t = (ep - ss_warmup_epochs) / float(max(1, (epochs - ss_warmup_epochs)))

            if ep <= hist_warmup_epochs:
                hist_t = None
            else:
                hist_t = (ep - hist_warmup_epochs) / float(max(1, (epochs - hist_warmup_epochs)))

            print(
                f"[SCHED][BASE] ep={ep:02d} stage1={stage1} "
                f"| ss_base={ss_base:.6f} (t={ss_t}) -> ss_p(after_stage1_gate)={float(ss_p):.6f} "
                f"| hist_base={hist_base:.6f} (t={hist_t}) -> hist_p(after_stage1_gate)={float(hist_p):.6f}",
                flush=True
            )
        # ------------------------------------------------------------------------

        # --- PATCH: floors so SS/HIST cannot get crushed into a never-used regime ---
        if not stage1:
            ss_p_floor = 0.02
            hist_p_floor = 0.02
            ss_p = float(max(ss_p, ss_p_floor))
            hist_p = float(max(hist_p, hist_p_floor))

            # ---------------------- POST-SCALING TRACE (PATCH) ----------------------
        if SCHED_DEBUG and ((ep <= 5) or (ep % 5 == 0)):
            print(
                f"[SCHED][SCALED] ep={ep:02d} "
                f"tf_r2_prev={tf_r2_prev:+.4f} stage1_min_tf_r2={stage1_min_tf_r2:.3f} "
                f"last_full_roll_r2={last_full_roll_r2:+.4f} "
                f"-> ss_p={float(ss_p):.6f} hist_p={float(hist_p):.6f}",
                flush=True
            )
        # -----------------------------------------------------------------------

        kappa_corr = _corr_kappa(ep, stage1=stage1, w_yonly_now=w_yonly)

        if stage1:
            w_geom_eff = 0.0
            w_gan_eff = 0.0
            w_teacher_eff = 0.0
            w_dyn_eff = 0.0
            w_var_eff = 0.0
            w_x_eff = 0.0
            w_obs_y_eff = float(w_obs_y)
            w_roll_eff = 0.0
            w_shift_eff = 0.0
            w_jac_eff = 0.0
            w_bound_eff = float(w_bound)
        else:
            aux_ok = (
                    (ep >= int(aux_start_epoch)) and
                    (tf_r2_prev >= float(aux_min_tf_r2))
                # REMOVED: roll R² gate — GAN needed before rollout is good
            )
            if enable_aux_after_recon and aux_ok:
                w_geom_eff, _ = _geom_eff(ep)
                w_gan_eff = float(w_gan)
                w_teacher_eff = float(w_teacher)
            else:
                w_geom_eff = 0.0
                w_gan_eff = 0.0
                w_teacher_eff = 0.0

            w_dyn_eff = float(w_dyn)
            w_var_eff = float(w_var)
            w_x_eff = float(w_x)
            w_obs_y_eff = float(w_obs_y)
            w_roll_eff = float(w_roll)
            w_bound_eff = float(w_bound)
            w_shift_eff = float(w_shift)
            w_jac_eff = float(w_jac)

        innov_on = _innov_enabled(use_innovation_feedback, _DISABLE_INNOV_DIAG)
        print(
            f"[INNOV][TRAIN] ep={ep:02d} use_innovation_feedback={use_innovation_feedback} "
            f"_DISABLE_INNOV_DIAG={_DISABLE_INNOV_DIAG} => innov_on={innov_on}",
            flush=True
        )

        print(
            f"| freeze_sigma={stage_freeze_sigma} | ss_p={ss_p:.3f} hist_p={hist_p:.3f} w_yonly={w_yonly:.2f} kappa_corr={kappa_corr:.3f} "
            f"| w_mu={w_mu:.1f} w_delta={w_delta:.1f} w_obs_y={w_obs_y_eff:.2f} w_roll={w_roll_eff:.2f} w_jac={w_jac_eff:.3f} w_bound={w_bound_eff:.2f} "
            f"| w_x={w_x_eff:.2f} w_geom={w_geom_eff:.3f} w_teacher={w_teacher_eff:.2f} w_gan={w_gan_eff:.2f} w_shift={w_shift_eff:.2e}")

        # --- DEBUG: GAN gate and w_yonly schedule ---
        if True:  # always print
            print(
                f"[SCHED][WEIGHTS] ep={ep:02d} w_yonly={w_yonly:.1f} w_gan_eff={w_gan_eff:.4f} "
                f"w_jac_eff={w_jac_eff:.3f} w_obs_y_eff={w_obs_y_eff:.2f} "
                f"aux_ok={'N/A' if stage1 else aux_ok} "
                f"tf_r2_prev={tf_r2_prev:+.4f} last_full_roll_r2={last_full_roll_r2:+.4f}",
                flush=True
            )

        # === NEW: Kappa diagnostic ===
        with torch.no_grad():
            # Calculate active effective kappa for this epoch
            # <--- FIX 2: Use student.kappa_vec() to respect the clamp floor --->
            kappa_vals = (float(kappa_corr) * student.kappa_vec()).tolist()
            print(f"[KAPPA] ep={ep:02d} per-dim: DBP={kappa_vals[0]:.3f} SBP={kappa_vals[1]:.3f} "
      f"CO={kappa_vals[2]:.3f} SV={kappa_vals[3]:.3f}", flush=True)

        student.train()
        disc.train()

        loss_sum = 0.0
        n = 0

        gan_loss_sum = 0.0
        disc_loss_sum = 0.0
        disc_real_mean_sum = 0.0
        disc_fake_mean_sum = 0.0
        n_gan = 0

        for bi, batch_items in enumerate(dl_active, start=1):
            y_in, Tb, Xb, XprevT, ystd, wb, y_cond, t_indices = _unpack_batch(batch_items, device=device)
            if hasattr(student, "kappa_vec"):
                kappa_eff = float(kappa_corr) * student.kappa_vec().view(1, -1)
            else:
                kappa_eff = float(kappa_corr)

            # <--- FIX 3A: Create detached kappa for TF path --->
            kappa_eff_detached = kappa_eff.detach() if torch.is_tensor(kappa_eff) else kappa_eff
            Bsz = Xb.size(0)

            # == Y-only pretrain (Section 4.2 Phase 2 Stage 1) ==
            # === Y-ONLY PRETRAIN: force backbone to learn phase decoding ===
            if yonly_pretrain:
                out_yonly_pt = student(
                    y_in, x_prev=None, b_prev=None, sample=False,
                    innov_prev=None
                )
                X_mu_pt = out_yonly_pt[3]
                w_batch = wb.clamp_min(1e-3)

                # Huber: robust reconstruction (Section 2.5)
                loss_mu_pt = _w_huber(X_mu_pt, Xb, w_batch, beta=0.5)

                eps_v = 1e-6
                var_true_pt = Xb.var(dim=0, unbiased=False).detach()  # <--- FIX: Changed to Xb
                var_pred_pt = X_mu_pt.var(dim=0, unbiased=False)
                # <--- FIX: Two-sided variance penalty --->
                loss_var_pt = ((var_pred_pt / (var_true_pt + eps_v)) - 1.0).pow(2).mean()

                # <--- FIX 3: Per-dim correlation loss (directly targets R²) --->
                pred_c = X_mu_pt - X_mu_pt.mean(dim=0, keepdim=True)
                true_c = Xb - Xb.mean(dim=0, keepdim=True)  # <--- FIX: Changed to Xb
                corr_per_dim = (pred_c * true_c).sum(dim=0) / (
                        pred_c.pow(2).sum(dim=0).sqrt() * true_c.pow(2).sum(dim=0).sqrt() + 1e-6
                )
                loss_corr_pt = (1.0 - corr_per_dim).mean()

                #  Delay variance loss during early pretrain --->
                #  Active from first pretrain epoch to prevent var_ratio overshoot --->
                var_weight = 10.0

                #  Absolute Variance MSE Penalty --->
                loss_var_match = F.mse_loss(var_pred_pt, var_true_pt)

                # Combine all pretrain losses
                loss_pt = (w_mu * loss_mu_pt
                           + var_weight * loss_var_pt
                           + 15.0 * loss_corr_pt
                           + 0.5 * loss_var_match)

                if not torch.isfinite(loss_pt):
                    opt.zero_grad(set_to_none=True)
                    continue

                opt.zero_grad(set_to_none=True)
                loss_pt.backward()

                # Gradient decomposition
                if ep <= 5 or bi == 1:
                    with torch.no_grad():
                        grad_head_path = 0.0
                        if student.head_dx_y_separate.weight.grad is not None:
                            grad_head_path = student.head_dx_y_separate.weight.grad.pow(2).sum().sqrt().item()

                        grad_b_path = 0.0
                        if student.head_corr.weight.grad is not None:
                            grad_b_path = student.head_corr.weight.grad.pow(2).sum().sqrt().item()

                        total_grad_path = grad_head_path + grad_b_path + 1e-8
                        print(
                            f"  grad_paths: head_path={grad_head_path:.4f} b_path={grad_b_path:.4f} head_share={grad_head_path / total_grad_path:.3f}")
                # === DIAGNOSTICS (every batch during pretrain — only 6 batches so no spam) ===
                with torch.no_grad():
                    # 1. Per-dim var ratio (THE critical metric)
                    std_pred = X_mu_pt.std(dim=0)
                    std_true = Xb.std(dim=0).clamp_min(1e-6)
                    vr = (std_pred / std_true).tolist()
                    vr_sq = [(v ** 2) for v in vr]  # actual variance ratio

                    # 2. Per-dim R² (is prediction correlated with truth?)
                    r2_dims = []
                    for j in range(X_mu_pt.size(1)):
                        ss_res = ((Xb[:, j] - X_mu_pt[:, j]) ** 2).sum()
                        ss_tot = ((Xb[:, j] - Xb[:, j].mean()) ** 2).sum().clamp_min(1e-12)
                        r2_dims.append(float((1.0 - ss_res / ss_tot).item()))

                    # 3. Per-dim mean of predictions (stuck at zero = not learning)
                    pred_means = X_mu_pt.mean(dim=0).tolist()

                    # 4. Pre-tanh activation stats (saturation check)
                    # Recompute through the ACTUAL separate backbone to get true activations
                    if y_in.dim() == 3:
                        y_last_diag = y_in[:, -1, :]
                    else:
                        y_last_diag = y_in

                    # <--- FIX 4: Compute combined Z to prevent dimension crash --->
                    z_sep = student.yonly_backbone(y_last_diag)  # [B, 64]

                    # Compute yonly temporal features (decoupled from WaveNet)
                    with torch.no_grad():
                        if hasattr(student, 'yonly_temporal_inp') and y_in.dim() == 3:
                            xt_diag = y_in.transpose(1, 2)
                            ht_diag = student.yonly_temporal_inp(xt_diag)
                            for conv in student.yonly_temporal_convs:
                                pad_d = 2 * conv.dilation[0]
                                ht_diag = ht_diag + torch.tanh(conv(F.pad(ht_diag, (pad_d, 0))))
                            z_temp_diag = student.yonly_temporal_out(ht_diag)[:, :, -1]
                        else:
                            z_temp_diag = None

                    if z_temp_diag is not None:
                        z_combined_diag = torch.cat([z_sep, z_temp_diag], dim=-1)
                    else:
                        z_combined_diag = z_sep

                    dmu_raw = student.head_dx_y_separate(z_combined_diag)

                    dmu_abs_mean = dmu_raw.abs().mean(dim=0).tolist()
                    dmu_max = dmu_raw.abs().max().item()

                    # 5. Gradient norms: backbone vs head
                    grad_yonly_bb = 0.0
                    grad_yonly_head = 0.0
                    grad_wavenet = 0.0
                    for name, p in student.named_parameters():
                        if p.grad is None:
                            continue
                        gn = p.grad.data.pow(2).sum().item()
                        if 'head_dx_y_separate' in name:
                            grad_yonly_head += gn
                        elif 'yonly_backbone' in name:
                            grad_yonly_bb += gn
                        elif 'inp' in name or 'blocks' in name or 'out.' in name:
                            grad_wavenet += gn
                    grad_yonly_bb = math.sqrt(grad_yonly_bb)
                    grad_yonly_head = math.sqrt(grad_yonly_head)
                    grad_wavenet = math.sqrt(grad_wavenet)

                    # 6. Separate head weight norm
                    head_w_norm = float(student.head_dx_y_separate.weight.data.norm().item())
                    head_b_norm = float(student.head_dx_y_separate.bias.data.norm().item())

                dim_names_pt = ["DBP", "SBP", "CO", "SV"]
                vr_s = " ".join(f"{dim_names_pt[j]}={vr_sq[j]:.4f}" for j in range(len(vr_sq)))
                r2_s = " ".join(f"{dim_names_pt[j]}={r2_dims[j]:+.4f}" for j in range(len(r2_dims)))
                mu_s = " ".join(f"{dim_names_pt[j]}={pred_means[j]:+.3f}" for j in range(len(pred_means)))
                raw_s = " ".join(f"{dim_names_pt[j]}={dmu_abs_mean[j]:.4f}" for j in range(len(dmu_abs_mean)))

                print(
                    f"[YONLY_PT][ep {ep:02d}][b{bi:02d}] "
                    f"loss={float(loss_pt.item()):.4f} "
                    f"(mu={float(loss_mu_pt.item()):.4f} var={float(loss_var_pt.item()):.4f})"
                    # <--- FIX: Removed undefined obs
                )
                print(f"  var_ratio: {vr_s}")
                print(f"  R2:        {r2_s}")
                print(f"  pred_mean: {mu_s}")
                print(f"  |dmu_raw|: {raw_s} (max={dmu_max:.4f})")
                print(
                    f"  grad: yonly_bb={grad_yonly_bb:.6f} yonly_head={grad_yonly_head:.6f} wavenet={grad_wavenet:.6f} | "
                    f"||W_head_sep||={head_w_norm:.6f} ||b_head_sep||={head_b_norm:.6f}",
                    flush=True
                )

                # <--- NEW: b_t magnitude and x_prop split --->
                bt_pretrain = out_yonly_pt[6]
                ct_pretrain = out_yonly_pt[7]
                dmu_y_sep_diag = 2.0 * torch.tanh(dmu_raw)  # Recreate actual dmu using _DELTA_SCALE

                dmu_contrib = dmu_y_sep_diag.abs().mean().item()
                bt_contrib = bt_pretrain.abs().mean().item()
                ratio = dmu_contrib / max(dmu_contrib + bt_contrib, 1e-6)

                print(
                    f"  b_t: mean|b|={bt_contrib:.4f} max|b|={bt_pretrain.abs().max().item():.4f} mean|c|={ct_pretrain.abs().mean().item():.4f}\n"
                    f"  x_prop split: |dmu_y_sep|={dmu_contrib:.4f} |b_t|={bt_contrib:.4f} dmu_fraction={ratio:.3f}"
                )
                if bt_contrib > 0.8:
                    print(f"  [WARN] b_t approaching b_max=1.0 during pretrain!")

                # <--- NEW: Head weight partition (Updated for 128-dim backbone) --->

                W = student.head_dx_y_separate.weight.data
                if W.size(1) > 64:
                    w_bb_part = W[:, :64].norm().item()
                    w_temp_part = W[:, 64:].norm().item()
                    temp_frac = w_temp_part / max(w_bb_part + w_temp_part, 1e-6)
                    print(
                        f"  head_W split: ||W_bb||={w_bb_part:.4f} ||W_temp||={w_temp_part:.4f} temp_fraction={temp_frac:.3f}")
                else:
                    print(f"  head_W: ||W||={W.norm().item():.4f}")

                # <--- NEW: z feature quality --->
                # Temporal feature quality
                z_out = z_temp_diag.detach() if z_temp_diag is not None else z_sep.detach()
                print("  temp_quality: ", end="")
                for j, nm in enumerate(["DBP", "SBP", "CO", "SV"]):
                    corrs = []
                    target_j = Xb[:, j]
                    target_c = target_j - target_j.mean()
                    target_std = target_c.pow(2).sum().sqrt().clamp(min=1e-6)
                    for zd in range(min(z_out.size(1), 32)):
                        z_c = z_out[:, zd] - z_out[:, zd].mean()
                        z_std = z_c.pow(2).sum().sqrt().clamp(min=1e-6)
                        corrs.append(abs(float((z_c * target_c).sum().item() / (z_std * target_std).item())))
                    best_corr = max(corrs) if corrs else 0.0
                    if j == 0:  # Only print HR to save space
                        print(f"best_temp_corr_with_{nm}={best_corr:.4f}", end="")
                print("", flush=True)  # Final newline to cleanly cap the block

                # <--- FIX: Remove head-specific clip, global clip = 2.0 --->
                nn.utils.clip_grad_norm_(student.parameters(), 5.0)

                opt.step()

                loss_sum += float(loss_pt.item()) * Bsz
                n += Bsz

                continue  # skip TF/rollout/GAN entirely

            K_target = int(_rollout_k(ep, Bsz=Bsz))

            K = int(min(Bsz, K_target))

            if DEBUG_PRINTS and (bi == 1):
                print(f"[ROLLOUT][K] ep={ep:02d} K_target={K_target} K_actual={K} Bsz={Bsz}", flush=True)

            rollout_scale = float(K) / float(max(1, K_target))
            rollout_scale = float(max(0.0, min(1.0, rollout_scale)))

            # choose a contiguous window of length K
            if (not stage1) and (K >= 2) and (Bsz > K):
                start = int(torch.randint(0, Bsz - K + 1, (1,), device=device).item())
            else:
                start = 0
            end = start + K

            # noise on x_prev (disabled in Stage-1)
            x_prev_noise = (x_prev_noise_std if (not stage1) else 0.0)
            Xprev_noisy = XprevT + (x_prev_noise * torch.randn_like(XprevT) if x_prev_noise > 0 else 0.0)

            prev0_used = (Xprev_noisy[0:1] if last_prev_train is None else last_prev_train)

            # (0) Builder forward to produce a predicted previous-state sequence for history-dropout.
            #     This MUST exist before pred_ok / mixing, otherwise Xprev_pred and Xmu_tf_builder are undefined.
            # (0) History Dropout & Builder Pass
            Xprev_noisy_feed = _clamp_feed(Xprev_noisy) if (clamp_x_rollout and clamp_x_rollout > 0) else Xprev_noisy
            clamp_val = float(clamp_x_rollout) if (clamp_x_rollout and clamp_x_rollout > 0) else None

            if stage1:
                # --- FAST PATH: STAGE 1 (No history dropout, no rollout) ---
                Xprev_mix_feed = Xprev_noisy_feed
                ss_p_eff = 0.0
                hist_p_eff = 0.0
            else:
                out_builder = student(
                    y_in,
                    x_prev=Xprev_noisy_feed,
                    b_prev=None,
                    sample=False,
                    innov_prev=None,
                )
                Xmu_tf_builder = out_builder[3]

                Xprev_pred_core = Xmu_tf_builder[:-1]
                if detach_hist_pred:
                    Xprev_pred_core = Xprev_pred_core.detach()

                prev0_used_for_pred = _clamp_feed(prev0_used) if clamp_val is not None else prev0_used
                Xprev_pred = torch.cat([prev0_used_for_pred, Xprev_pred_core], dim=0)
                Xprev_pred_feed = _clamp_feed(Xprev_pred) if clamp_val is not None else Xprev_pred

                # (1) Cleanliness of the *predicted* prev
                pred_ok = _xprev_rows_clean_mask(
                    x_unclamped=Xprev_pred.detach(), x_clamped=Xprev_pred_feed.detach(), clamp_val=clamp_val,
                    abs_mean_thr=1.5, abs_max_thr=(min(4.0, clamp_val) if clamp_val else 4.0), max_sat_frac=0.25,
                )

                # (2) CLOSED-LOOP cleanliness probe (PERFORMANCE FIX: Every 100 batches instead of 20)
                if 'pred_clean_frac' not in locals():
                    pred_clean_frac = 1.0  # Safe default initialization

                if (bi == 1) or (bi % 100 == 0):
                    pred_clean_frac_cl = _closedloop_predprev_clean_frac(
                        student=student, y_seq=y_in,
                        xprev0=prev0_used_for_pred if prev0_used_for_pred.dim() == 2 else prev0_used_for_pred.view(1,
                                                                                                                   -1),
                        ystd_seq=ystd, clamp_feed_fn=_clamp_feed, clamp_val=clamp_val,
                        model_yx=model_yx, cond_gain_for_teacher=cond_gain_for_teacher,
                        kappa_eff=kappa_eff, innov_on=innov_on,
                        t_indices_seq=t_indices
                    )
                    pred_clean_frac = float(pred_clean_frac_cl)

                if pred_clean_frac < 0.80:
                    pred_ok = torch.zeros_like(pred_ok, dtype=torch.bool)

                ss_p_eff = float(max(float(ss_p) * (0.25 + 0.75 * pred_clean_frac), 0.02))
                hist_p_eff = float(max(float(hist_p) * (0.25 + 0.75 * pred_clean_frac), 0.02))

                # (3) History dropout mixing
                keep_true = _bernoulli_mask((Bsz, 1), float(hist_p_eff), device=device)
                keep_true[0:1] = True
                keep_true = keep_true | (~pred_ok)
                Xprev_mix_feed = torch.where(keep_true, Xprev_noisy_feed, Xprev_pred_feed)

                # (4) Inject rollout-like noise
                noise_p = 0.25
                Xprev_mix_pre = Xprev_mix_feed
                if noise_p > 0.0:
                    with torch.no_grad():
                        noise_scale = (Xmu_tf_builder.detach() - Xb).abs().mean(dim=0, keepdim=True).clamp(min=1e-4,
                                                                                                           max=0.10)
                        inject = (torch.rand((Bsz, 1), device=device) < noise_p).float()
                        Xprev_mix_pre = Xprev_mix_feed + inject * (0.5 * noise_scale) * torch.randn_like(Xprev_mix_feed)
                Xprev_mix_feed = _clamp_feed(Xprev_mix_pre) if clamp_val is not None else Xprev_mix_pre

                # (5) Final clean mask fallback
                clean_mask = _xprev_rows_clean_mask(
                    x_unclamped=Xprev_mix_pre.detach(), x_clamped=Xprev_mix_feed.detach(), clamp_val=clamp_val,
                    abs_mean_thr=2.5, abs_max_thr=6.0, max_sat_frac=0.25,
                )
                Xprev_mix_feed = torch.where(clean_mask, Xprev_mix_feed, Xprev_noisy_feed)

                final_clean_frac = float(clean_mask.float().mean().item())
                ss_p_eff = float(max(float(ss_p) * (0.25 + 0.75 * final_clean_frac), 0.05))

            # ---------- end gating ----------

            # y-only proposal if needed
            need_yonly = (kappa_corr > 0.0) or (w_yonly > 0.0)
            X_mu_yonly = None
            if need_yonly:
                out_yonly = student(
                    y_in,
                    x_prev=None,
                    b_prev=None,
                    sample=False,
                    innov_prev=None,
                )

                X_mu_yonly = out_yonly[3]

            innovation_tf = None
            if innov_on:
                innovation_tf = _compute_innov_scalar(ystd, Xprev_mix_feed, t_indices)

            if DEBUG_PRINTS and innov_on and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                with torch.no_grad():
                    print(
                        f"[INNOV][TF] ep={ep:02d} b={bi:04d} mean|innovation_tf|={float(innovation_tf.abs().mean().item()):.5f}",
                        flush=True
                    )
            xprev0 = prev0_used
            xprev0 = _clamp_feed(xprev0) if (clamp_x_rollout and clamp_x_rollout > 0) else xprev0

            # Unpack xprop_tf and g_tf
            # == Closed-loop roll-in (Section 4.2 Phase 2 Stage 2) ==
            z, X_mu, X_logsig, b_tf, c_tf, prevfed_tf, ss_use_rate_tf, xprop_tf, g_tf = _rollin_forward_stateful_b(
                y_seq=y_in,
                xprev0=xprev0,
                x_true_seq=Xb,
                ystd_seq=ystd,
                kappa_eff=kappa_eff_detached,
                use_innov=innov_on,
                ss_p_eff=float(ss_p_eff),
                process_noise_std=float(rollout_noise_std),
                t_indices_seq=t_indices,
            )

            if (bi == 1) or (bi % 50 == 0):
                print(
                    f"[ROLLIN][CLOSED] ep={ep:02d} b={bi:04d} ss_p_eff={ss_p_eff:.3f} ss_use_rate={ss_use_rate_tf:.3f}",
                    flush=True)
                if (not stage1) and (float(ss_p_eff) >= 0.05) and (float(ss_use_rate_tf) <= 0.01):
                    print(
                        f"[WARN][SS_OFF] ep={ep:02d} b={bi:04d} "
                        f"ss_p_eff={float(ss_p_eff):.3f} but ss_use_rate_tf={float(ss_use_rate_tf):.3f}. "
                        f"SS is effectively off (forced teacher / gating / floor mismatch).",
                        flush=True
                    )

            if (bi == 1) or (bi % 50 == 0):
                kl_disp = loss_shift_kl.item() if 'loss_shift_kl' in locals() else 0.0
                print(
                    f"[ENC][DRIFT] ep={ep:02d} b={bi:04d} "
                    f"mean|c|={c_tf.abs().mean().item():.4f} "
                    f"mean|b|={b_tf.abs().mean().item():.4f}",
                    f"KL_shift={kl_disp:.6f}",  # <--- ADDED THIS
                    flush=True
                )
            if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                d = getattr(student, "last_diag", {}) or {}
                print(
                    f"[BIAS][TF] ep={ep:02d} b={bi:04d} "
                    f"gb_mean={d.get('gb_mean', float('nan')):.3f} "
                    f"gb_p10/p50/p90={d.get('gb_p10', float('nan')):.3f}/{d.get('gb_p50', float('nan')):.3f}/{d.get('gb_p90', float('nan')):.3f} "
                    f"bias_fraction={d.get('bias_fraction', float('nan')):.3f} "
                    f"dx_bias_abs_mean={d.get('dx_bias_abs_mean', float('nan')):.4f} "
                    f"dx_total_abs_mean={d.get('dx_total_abs_mean', float('nan')):.4f}",
                    flush=True
                )

                # --- DEBUG: gate stats ---
            # --- DEBUG: gate stats ---
            if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                with torch.no_grad():
                    # must match student forward
                    # <--- FIX 1: Use student instance and registered buffer --->
                    g_raw = torch.sigmoid(student.gate(z).float())  # [B, d_x]
                    g_eff = student.g_floor + (1.0 - student.g_floor) * g_raw

                    g_raw_m = g_raw.mean(dim=0).cpu().numpy()
                    g_eff_m = g_eff.mean(dim=0).cpu().numpy()

                    # small, readable summary per-dim
                    raw_s = ", ".join(f"{v:.3f}" for v in g_raw_m.tolist())
                    eff_s = ", ".join(f"{v:.3f}" for v in g_eff_m.tolist())

                    print(f"[gate][ep {ep:02d}][b{bi:04d}] g_raw_mean=[{raw_s}] g_eff_mean=[{eff_s}]")

            if stage_freeze_sigma:
                X_logsig = torch.zeros_like(X_mu) + float(fixed_logsig)

            w_batch = wb.clamp_min(1e-3)

            # base supervised losses (TF)
            # == Reconstruction losses (Section 4.3) ==
            d_hat = X_mu - prevfed_tf
            d_true = Xb - prevfed_tf
            loss_mu = _w_huber(X_mu, Xb, w_batch, beta=0.5)
            loss_delta = _w_mse(d_hat, d_true, w_batch)

            # === BUG FIX: DIRECT PROPOSAL SUPERVISION & GATE REGULARIZATION ===
            # 1. Direct Proposal Loss (Change 1 & 2): Supervise the increment pre-gate
            dx_prop = xprop_tf - prevfed_tf.detach()
            dx_true_prop = Xb - prevfed_tf.detach()
            # Direct proposal supervision (Section 4.3)
            per_prop = F.smooth_l1_loss(dx_prop, dx_true_prop, beta=0.2, reduction="none").sum(dim=1)
            loss_prop = (per_prop * w_batch).mean()

            # <--- FIX 4: Gate floor to 0.60 --->
            # Gate floor penalty: prevents persistence collapse
            g_target_min = 0.75
            loss_gate_floor = F.relu(g_target_min - g_tf).pow(2).mean()
            # ===================================================================

            delta_err = (d_hat - d_true)
            w_norm = w_batch.view(-1, 1)
            mean_err_dim = (delta_err * w_norm).sum(dim=0) / (w_norm.sum(dim=0).clamp_min(1e-6))
            # Drift penalty: per-dim systematic bias
            loss_delta_bias = (mean_err_dim.pow(2)).mean()

            # Stage 2b maintenance loss (Section 4.3)
            loss_yonly = torch.zeros((), device=device)
            if w_yonly > 0.0 and X_mu_yonly is not None:
                loss_yonly = _w_huber(X_mu_yonly, Xb, w_batch, beta=0.5)

                # --- Match pretrain loss formulation (prevents backbone corruption) ---
                # X_mu_yonly comes from x_prev=None path: pure yonly graph, zero WaveNet params
                eps_v = 1e-6
                # 1. Two-sided variance ratio penalty (weight 10.0, matching pretrain)
                var_pred_yo = X_mu_yonly.var(dim=0, unbiased=False)
                var_true_yo = Xb.var(dim=0, unbiased=False).detach()
                loss_var_yo = ((var_pred_yo / (var_true_yo + eps_v)) - 1.0).pow(2).mean()

                # 2. Per-dim correlation loss (weight 15.0, matching pretrain)
                pred_c_yo = X_mu_yonly - X_mu_yonly.mean(dim=0, keepdim=True)
                true_c_yo = Xb.detach() - Xb.detach().mean(dim=0, keepdim=True)
                corr_yo = (pred_c_yo * true_c_yo).sum(dim=0) / (
                        pred_c_yo.pow(2).sum(dim=0).sqrt() * true_c_yo.pow(2).sum(dim=0).sqrt() + 1e-6
                )
                loss_corr_yo = (1.0 - corr_yo).mean()

                # 3. Absolute variance MSE (weight 0.5, matching pretrain)
                loss_var_match_yo = F.mse_loss(var_pred_yo, var_true_yo)

                loss_yonly = (loss_yonly
                              + 10.0 * loss_var_yo
                              + 15.0 * loss_corr_yo
                              + 0.5 * loss_var_match_yo)
            # == Observation anchoring (Section 4.3) ==
            # Student state -> frozen teacher -> compare with y_obs
            # obs-y anchoring
            with torch.enable_grad():
                y_hat_std, *_ = model_yx(_teacher_input_from_pred(X_mu, t_indices),
                                         cond_gain_scale=cond_gain_for_teacher)
            per_obs = (y_hat_std - ystd).pow(2).view(-1)
            loss_obs_y = (per_obs * w_batch).mean()

            loss_x_nll = torch.zeros((), device=device)
            if w_x_eff > 0:
                loss_x_nll = (_gaussian_nll(Xb, X_mu, X_logsig) * w_batch).mean()

            # soft bound (objective)
            loss_bound = _soft_bound_penalty(X_mu, bound=float(bound_soft), power=float(bound_power))

            # --- TF amplitude preservation (two-sided variance match) ---
            eps_v = 1e-6
            var_true_tf = Xb.var(dim=0, unbiased=False).detach()
            var_pred_tf = X_mu.var(dim=0, unbiased=False)
            ratio_tf = var_pred_tf / (var_true_tf + eps_v)
            loss_var_tf = (ratio_tf - 1.0).pow(2).mean()
            # --- for Jacobian-on-rollout sampling (closed-loop state distribution) ---
            mu_roll_pool = None
            y_roll_pool = None
            ystd_roll_pool = None

            # rollout loss (closed-loop) + correction regularizers + shift KL + teacher rollout distill
            loss_roll = torch.zeros((), device=device)
            loss_shift = torch.zeros((), device=device)
            loss_corr_mag_roll = torch.zeros((), device=device)
            loss_corr_smooth_roll = torch.zeros((), device=device)
            loss_teacher_roll = torch.zeros((), device=device)
            loss_ct_dc = torch.zeros((), device=device)
            loss_b_state = torch.zeros((), device=device)
            loss_var_roll = torch.zeros((), device=device)

            loss_spectral = torch.zeros((), device=device)

            # == Rollout losses (Section 4.3) ==
            if (not stage1) and (w_roll_eff > 0) and (K >= 2):
                y_win = y_in[start:end]
                X_win = Xb[start:end]
                XprevT_win = XprevT[start:end]
                w_win = w_batch[start:end]

                D = Xb.size(1)

                mu_roll = torch.empty((K, D), device=device, dtype=X_mu.dtype)
                c_roll = torch.empty((K, D), device=device, dtype=X_mu.dtype)
                b_roll = torch.empty((K, D), device=device, dtype=X_mu.dtype)  # <-- KEEP

                # store prev stream for shift-KL and bias_help (no Python list)
                xprev_roll_for_kl = torch.empty((K, D), device=device, dtype=X_mu.dtype)
                prev_feed_seq = torch.empty((K, D), device=device, dtype=X_mu.dtype)  # for bias_help
                innov_last = None

                # seed prev
                if start == 0:
                    seed_prev = prev0_used
                else:
                    seed_prev = XprevT_win[0:1].detach()

                prev_used_feed = seed_prev
                prev_used_kl = seed_prev

                b_used_feed = torch.zeros((1, D), device=device, dtype=X_mu.dtype)

                # random draws in one shot (avoids per-step torch.rand/item overhead)
                if (rollout_sample_p is not None) and (float(rollout_sample_p) > 0.0):
                    do_sample_mask = (torch.rand((K, 1), device=device) < float(rollout_sample_p))
                    eps_hat = torch.randn((K, D), device=device, dtype=X_mu.dtype)
                else:
                    do_sample_mask = torch.zeros((K, 1), device=device, dtype=torch.bool)
                    eps_hat = None

                # Process noise: inject Gaussian on fed prev (Section 3.1)
                if (rollout_noise_std is not None) and (float(rollout_noise_std) > 0.0):
                    eps_proc = torch.randn((K, D), device=device, dtype=X_mu.dtype)
                    proc_std = float(rollout_noise_std)
                else:
                    eps_proc = None
                    proc_std = 0.0

                # scheduled sampling random draws (still respects per-step cleanliness forcing)
                if ss_p_eff > 0.0 and (K >= 2):
                    ss_mask = (torch.rand((K - 1, 1), device=device) < float(ss_p_eff)).float()
                else:
                    ss_mask = None

                # periodic reanchor mask (spike-trigger still computed per-step from innovation)
                if (reanchor_every is not None) and (int(reanchor_every) > 0):
                    t_idx = torch.arange(1, K + 1, device=device)
                    periodic_reanchor = ((t_idx % int(reanchor_every)) == 0).view(K, 1)
                else:
                    periodic_reanchor = torch.zeros((K, 1), device=device, dtype=torch.bool)

                # ---------- counters (preserved) ----------
                ss_eligible = 0
                ss_use = 0
                ss_forced_teacher = 0
                ss_off = 0
                tf_state_pred_b = 0  # teacher state but predicted b carried (your current behavior)
                innov_roll_sum = 0.0
                innov_roll_n = 0

                innov_tracker_train = InnovTracker()
                innov_buffer = torch.zeros((1, 8), device=device, dtype=X_mu.dtype)

                # --- K-step rollout unroll ---
                # ---------- rollout-unroll (still sequential; everything else vectorized) ----------
                for i in range(K):
                    yi = y_win[i:i + 1]
                    # store prev streams BEFORE step (matches your original xprev_roll_for_kl.append(prev_used_kl))
                    xprev_roll_for_kl[i:i + 1] = prev_used_kl
                    prev_feed_seq[i:i + 1] = prev_used_feed

                    # innovation for this step: y_obs - y_hat(prev)
                    innovation_i = None
                    if innov_on:
                        y_obs_i = ystd[start + i:start + i + 1]
                        innovation_i = _compute_innov_scalar(y_obs_i, prev_used_feed,
                                                             t_indices[start + i:start + i + 1])
                        innov_roll_sum += float(innovation_i.abs().mean().item())
                        innov_roll_n += 1
                        if i == (K - 1):
                            innov_last = innovation_i

                        innov_tracker_train.record(innovation_i)
                        # Shift 8-step innovation buffer
                        innov_buffer = torch.cat([innov_buffer[:, 1:], innovation_i.view(1, 1)], dim=1)

                    # y-only proposal (now comes from precomputed xy_win)
                    # y-only proposal (calculated per-step to allow correct lag slicing)
                    xy_i = None
                    if kappa_corr > 0.0:
                        out_yonly_step = student(
                            yi,
                            x_prev=None,
                            b_prev=None,
                            sample=False,
                            innov_prev=None,
                        )
                        xy_i = out_yonly_step[3]

                    # main step (keep correction path, keep b_prev stateful)
                    out_i = student(
                        yi,
                        x_prev=_clamp_feed(prev_used_feed) if (
                                    clamp_x_rollout and clamp_x_rollout > 0) else prev_used_feed,
                        b_prev=b_used_feed,
                        sample=False,  # we emulate sampling below (exact same distribution)
                        x_yonly_prop=xy_i,
                        kappa=kappa_eff if not torch.is_tensor(kappa_eff) else kappa_eff[:1],
                        innov_prev=innov_buffer,
                    )

                    xmu_i = out_i[3]
                    xlogsig_i = out_i[4]
                    b_next = out_i[6]
                    c_i = out_i[7]

                    # emulate your do_sample path without per-step student(sample=True)
                    if do_sample_mask[i].item():
                        x_hat_i = xmu_i + torch.exp(xlogsig_i) * eps_hat[i:i + 1]
                        x_next_raw = x_hat_i
                    else:
                        x_next_raw = xmu_i

                    # extra process noise (your rollout_noise_std)
                    if eps_proc is not None:
                        x_next_raw = x_next_raw + proc_std * eps_proc[i:i + 1]

                    # record sequences
                    mu_roll[i:i + 1] = xmu_i
                    c_roll[i:i + 1] = c_i
                    b_roll[i:i + 1] = b_next  # <-- KEEP

                    # advance bias state (detaching is preserved)
                    # Carry bias state forward (matches export)
                    b_used_feed = b_next.detach() if detach_rollout_prev else b_next

                    # clamp for feeding
                    x_next_feed = _clamp_feed(x_next_raw) if (clamp_x_rollout and clamp_x_rollout > 0) else x_next_raw

                    # --- Periodic / spike-triggered re-anchoring to y-only state estimate (preserved) ---
                    do_reanchor = bool(periodic_reanchor[i].item())
                    if (not do_reanchor) and (innovation_i is not None):
                        if float(innovation_i.abs().mean().item()) >= float(reanchor_innov_thr):
                            do_reanchor = True

                    # Re-anchor to y-only on spikes/periodic (Section 2.6)
                    if do_reanchor and (xy_i is not None):
                        a = float(max(0.0, min(1.0, reanchor_alpha)))
                        x_anchor = _clamp_feed(xy_i) if (clamp_x_rollout and clamp_x_rollout > 0) else xy_i
                        x_next_feed = (1.0 - a) * x_next_feed + a * x_anchor

                        r = float(max(0.0, min(1.0, reanchor_reset_b)))
                        b_used_feed = (1.0 - r) * b_used_feed

                    # --- DEBUG: rollout drift + clamp saturation (preserved) ---
                    if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1)) and (i in (0, K - 1)):
                        with torch.no_grad():
                            err_i = (xmu_i - X_win[i:i + 1]).abs().mean().item()
                            if clamp_x_rollout and clamp_x_rollout > 0:
                                thr = 0.95 * float(clamp_x_rollout)
                                sat_prev = (prev_used_feed.abs() > thr).float().mean().item()
                                sat_next = (x_next_feed.abs() > thr).float().mean().item()
                            else:
                                sat_prev = float("nan")
                                sat_next = float("nan")
                            prev_abs = prev_used_feed.abs().mean().item()
                            next_abs = x_next_feed.abs().mean().item()
                            print(
                                f"[roll][ep {ep:02d}][b{bi:04d}] i={i:02d}/{K - 1:02d} "
                                f"| mean|prev|={prev_abs:.3f} mean|next|={next_abs:.3f} "
                                f"| err_meanabs={err_i:.4f} | sat_prev={sat_prev:.3f} sat_next={sat_next:.3f}",
                                flush=True
                            )

                    # shift-KL stream chooses unclamped vs clamped next (preserved)
                    x_next_for_kl = x_next_raw if bool(shift_kl_use_unclamped) else x_next_feed

                    # scheduled sampling for prev streams (preserved semantics + counts)
                    if i < K - 1:
                        ss_eligible += 1

                        if ss_p_eff > 0.0:
                            prev_step_clean = True
                            if (clamp_x_rollout and clamp_x_rollout > 0):
                                prev_sat = float(_sat_frac(prev_used_feed.detach(), float(clamp_x_rollout)).item())
                                prev_step_clean = (prev_sat <= 0.50)
                                if not prev_step_clean:
                                    ss_forced_teacher += 1

                            if prev_step_clean:
                                use_pred = ss_mask[i:i + 1]  # [1,1]
                                use_pred_int = int((use_pred.item() > 0.5))
                                ss_use += use_pred_int
                                if use_pred_int == 0:
                                    tf_state_pred_b += 1
                            else:
                                use_pred = torch.zeros((1, 1), device=device, dtype=X_mu.dtype)  # force teacher
                                use_pred_int = 0
                                ss_use += 0
                                tf_state_pred_b += 1

                            x_pred = x_next_feed.detach() if detach_rollout_prev else x_next_feed
                            b_pred = b_next.detach() if detach_rollout_prev else b_next

                            prev_used_feed = use_pred * x_pred + (1.0 - use_pred) * X_win[i:i + 1].detach()
                            b_used_feed = b_pred  # keep your design: carry predicted bias even under teacher state

                            prev_used_kl = use_pred * x_next_for_kl + (1.0 - use_pred) * X_win[i:i + 1].detach()
                            b_used_kl = b_next

                        else:
                            ss_off += 1
                            prev_used_feed = X_win[i:i + 1].detach()
                            prev_used_kl = X_win[i:i + 1].detach()
                            b_used_feed = b_next.detach() if detach_rollout_prev else b_next
                            b_used_kl = b_used_feed

                # --- TRAINING: enforce that stateful bias should not make things worse (PRESERVED, still last-step only) ---
                # --- TRAINING: enforce that stateful bias should not make things worse ---
                loss_bias_help = torch.zeros((), device=device, dtype=X_mu.dtype)
                yi_last = y_win[K - 1:K]
                prev_last = prev_feed_seq[K - 1:K]

                # 1. Slice lags for the last step

                # 2. Recalculate xy_last locally (since xy_win was deleted)
                xy_last = None
                if kappa_corr > 0.0:
                    out_yonly_last = student(
                        yi_last,
                        x_prev=None,
                        b_prev=None,
                        sample=False,
                        innov_prev=None,
                    )
                    xy_last = out_yonly_last[3]

                # 3. Run the reset student
                out_reset = student(
                    yi_last,
                    x_prev=_clamp_feed(prev_last) if (clamp_x_rollout and clamp_x_rollout > 0) else prev_last,
                    b_prev=None,
                    sample=False,
                    x_yonly_prop=xy_last,
                    kappa=kappa_eff if not torch.is_tensor(kappa_eff) else kappa_eff[:1],
                    innov_prev=innov_last,
                )
                # Bias-help: penalise integrator when it hurts (Section 4.3)
                err_stateful = (mu_roll[K - 1:K] - X_win[K - 1:K]).abs().mean()
                err_reset = (out_reset[3] - X_win[K - 1:K]).abs().mean()
                loss_bias_help = F.relu(err_stateful - err_reset)

                # Cardiac variance: penalise under-variance (Section 4.3)
                # --- NEW: cardiac-aware local variance matching (anti-smoothing) ---
                loss_local_var = torch.zeros((), device=device)
                var_win_size = 12  # ~2 cardiac cycles at 6Hz
                if K > var_win_size + 1:
                    roll_unf = mu_roll.unfold(0, var_win_size, 1)  # [K-win+1, D, win]
                    true_unf = X_win.unfold(0, var_win_size, 1)  # [K-win+1, D, win]
                    roll_local_var = roll_unf.var(-1)  # [K-win+1, D]
                    true_local_var = true_unf.var(-1).detach()  # [K-win+1, D]
                    # Only penalize UNDER-variance (smoothing), not over-variance
                    ratio = roll_local_var / (true_local_var.clamp_min(1e-6))
                    loss_local_var = F.relu(1.0 - ratio).pow(2).mean()

                # Spectral power: penalise lost frequency power (Section 4.3)
                # --- NEW: roll-in variance match (targets std_ratio << 1) ---
                loss_spectral = torch.zeros((), device=device)
                if K >= 16:
                    fft_roll = torch.fft.rfft(mu_roll, dim=0)
                    fft_true = torch.fft.rfft(X_win.detach(), dim=0)
                    power_roll = fft_roll.abs().pow(2)
                    power_true = fft_true.abs().pow(2)
                    # Only penalize UNDER-power (smoothing)
                    loss_spectral = F.relu(power_true - power_roll).mean()

                var_roll_pred = mu_roll.var(dim=0)
                var_roll_true = X_win.var(dim=0).detach()
                # Rollout variance preservation (Section 4.3)
                loss_var_roll = F.relu(1.0 - var_roll_pred / (var_roll_true + 1e-6)).pow(2).mean()
                eps = 1e-6

                if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                    with torch.no_grad():
                        ss_use_rate = float(ss_use / max(1, ss_eligible))
                        ss_forced_rate = float(ss_forced_teacher / max(1, ss_eligible))
                        ss_off_rate = float(ss_off / max(1, ss_eligible))
                        warn_eff_off = (ss_p_eff > 0.10) and (ss_use_rate < 0.02)
                        print(
                            f"[SS][SUM] ep={ep:02d} b={bi:04d} K={K} "
                            f"ss_p={ss_p:.3f} ss_p_eff={ss_p_eff:.3f} "
                            f"eligible={ss_eligible} use={ss_use} use_rate={ss_use_rate:.3f} "
                            f"forced_teacher={ss_forced_teacher} forced_rate={ss_forced_rate:.3f} "
                            f"ss_off_steps={ss_off} ss_off_rate={ss_off_rate:.3f} "
                            f"tf_state_pred_b_steps={tf_state_pred_b}"
                            + ("  [WARN:SS_EFFECTIVELY_OFF]" if warn_eff_off else ""),
                            flush=True
                        )

                # INSERTION POINT B: rollout innovation magnitude summary
                if DEBUG_PRINTS and innov_on and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                    with torch.no_grad():
                        mean_innov_roll = float(innov_roll_sum / max(1, innov_roll_n))
                        print(
                            f"[INNOV][ROLL] ep={ep:02d} b={bi:04d} mean|innovation_roll|={mean_innov_roll:.5f} (n={innov_roll_n})",
                            flush=True
                        )
                    innov_tracker_train.print_summary(tag=f" [TRAIN_ROLL ep={ep:02d} b={bi:04d}]")

                # ---- stash rollout state pool for Jacobian penalty (closed-loop regime) ----
                mu_roll_pool = mu_roll.detach()
                y_roll_pool = y_win  # aligns 1:1 with mu_roll rows
                ystd_roll_pool = ystd[start:end]  # aligns 1:1 with mu_roll rows (for innovation)

                if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                    with torch.no_grad():
                        err_mean = (mu_roll - X_win).abs().mean().item()
                        err_last = (mu_roll[-1:] - X_win[-1:]).abs().mean().item()
                        print(
                            f"[roll][ep {ep:02d}][b{bi:04d}] K={K} | err_meanabs={err_mean:.4f} | err_last={err_last:.4f}")

                if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                    with torch.no_grad():
                        err_seq = (mu_roll - X_win).abs().mean(dim=1)  # [K]
                        err0 = float(err_seq[0].item())
                        errL = float(err_seq[-1].item())
                        growth = float((err_seq[-1] / err_seq[0].clamp_min(1e-6)).item())

                        t_idx = torch.arange(K, device=err_seq.device, dtype=err_seq.dtype)
                        y_log = torch.log(err_seq.clamp_min(1e-6))
                        t_c = t_idx - t_idx.mean()
                        y_c = y_log - y_log.mean()
                        log_slope = float((t_c * y_c).sum().item() / (t_c.pow(2).sum().clamp_min(1e-12)).item())

                        print(
                            f"[ROLLOUT][GROWTH] ep={ep:02d} b={bi:04d} K={K} "
                            f"err0={err0:.4f} errL={errL:.4f} growth={growth:.2f} log_slope={log_slope:+.3f}",
                            flush=True
                        )

                # discounted weights (late-step emphasis), normalized mean=1
                t = torch.arange(0, K, device=device, dtype=X_mu.dtype)
                w_t = (roll_gamma ** (K - 1 - t))
                w_t = (w_t / w_t.mean()).view(K)

                # Multi-scale L1: phase-sensitive at short scales, amplitude-sensitive at long scales
                ms_losses = []
                for scale in [1, 4, 16]:
                    if scale == 1:
                        r = mu_roll[:min(50, K)]
                        t = X_win[:min(50, K)].detach()
                        w = w_win[:min(50, K)]
                    else:
                        Ks = K // scale
                        if Ks < 2:
                            continue
                        r = F.avg_pool1d(mu_roll.t().unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0).t()
                        t = F.avg_pool1d(X_win.detach().t().unsqueeze(0), kernel_size=scale, stride=scale).squeeze(
                            0).t()
                        w = F.avg_pool1d(w_win.view(1, 1, -1), kernel_size=scale, stride=scale).squeeze()
                    per = F.smooth_l1_loss(r, t, beta=0.5, reduction="none").sum(dim=1)
                    ms_losses.append((per * w).mean())
                loss_x_roll = sum(ms_losses) / len(ms_losses)

                # === NEW: INNOVATION (dx) LOSSES (Beat Persistence) ===
                # dx loss: step increments must match truth
                dx_roll = mu_roll[1:K] - mu_roll[0:K - 1]
                dx_true = X_win[1:K] - X_win[0:K - 1]  # <--- FIX 1: Added dx_true
                loss_delta_roll = F.mse_loss(dx_roll, dx_true, reduction="mean")  # <--- FIX 2: Added loss_delta_roll

                loss_last = F.mse_loss(mu_roll[K - 1:K], X_win[K - 1:K], reduction="mean")
                err = (mu_roll - X_win)
                mean_err_dim_h = (err * w_win.view(K, 1)).sum(dim=0) / (w_win.view(K, 1).sum(dim=0).clamp_min(1e-6))
                loss_drift_h = (mean_err_dim_h.pow(2)).mean()
                loss_mean_abs = (err.abs() * w_win.view(K, 1)).mean()

                loss_roll = (
                        loss_x_roll
                        + 0.25 * loss_delta_roll
                        + 0.25 * loss_last
                        + 0.10 * loss_drift_h
                        + 0.10 * loss_mean_abs
                        + 0.50 * loss_bias_help
                        + 3.0 * loss_local_var  # <--- ADDED THIS LINE
                )

                # correction regularizers (rollout)
                # Correction magnitude: keep c(t) small
                loss_corr_mag_roll = c_roll.abs().mean()
                # Correction smoothness: no jerky corrections
                loss_corr_smooth_roll = (c_roll[1:] - c_roll[:-1]).abs().mean() if K >= 2 else torch.zeros((),
                                                                                                           device=device,
                                                                                                           dtype=X_mu.dtype)
                # --- NEW: prevent long-horizon drift by removing systematic bias ---
                # (a) DC penalty on correction increments (prevents small bias integrating into runaway offset)
                # DC penalty: prevent runaway offset
                loss_ct_dc = (c_roll.mean(dim=0).pow(2)).mean()

                # (b) Penalize large bias state (keeps integrator bounded even under weak supervision)
                # Bias magnitude: keep integrator bounded
                loss_b_state = (b_roll.pow(2)).mean()

                # shift KL (TF roll-in vs rollout roll-in)
                if (w_shift_eff > 0.0) and (K >= 4):
                    xprev_tf = Xprev_mix_feed[start:end].detach()  # TF roll-in (fed)
                    xprev_roll = xprev_roll_for_kl.detach()  # rollout-fed prev stream

                    m_tf = xprev_tf.mean(dim=0)
                    v_tf = xprev_tf.var(dim=0, unbiased=False).clamp_min(1e-6)

                    m_rl = xprev_roll.mean(dim=0)
                    v_rl = xprev_roll.var(dim=0, unbiased=False).clamp_min(1e-6)

                    # Shift-KL: limit TF vs rollout divergence (Section 4.3)
                    # KL term (diag-Gauss)
                    loss_shift_kl = diag_gauss_kl(m_tf, v_tf, m_rl, v_rl).mean()

                    # Mean match (targets mean_shift)
                    loss_shift_mean = (m_rl - m_tf).pow(2).mean()

                    # Log-std match (targets std_ratio collapse like 0.186)
                    logstd_tf = 0.5 * torch.log(v_tf)
                    logstd_rl = 0.5 * torch.log(v_rl)
                    loss_shift_logstd = (logstd_rl - logstd_tf).pow(2).mean()

                    # Combine (weights chosen to make log-std correction meaningful)
                    # Combined shift: KL + mean + log-std
                    loss_shift = loss_shift_kl + 0.5 * loss_shift_mean + 2.0 * loss_shift_logstd

                if (enable_aux_after_recon and (w_teacher_eff > 0.0) and (w_teacher_roll_mult > 0.0)):
                    with torch.no_grad():
                        y_teacher_true_seq, *_ = model_yx(_teacher_input_from_pred(X_win, t_indices[start:end]),
                                                          cond_gain_scale=cond_gain_for_teacher)
                    with torch.enable_grad():
                        y_teacher_hat_seq, *_ = model_yx(_teacher_input_from_pred(mu_roll, t_indices[start:end]),
                                                         cond_gain_scale=cond_gain_for_teacher)
                    per_t = (y_teacher_hat_seq - y_teacher_true_seq).pow(2).view(-1)
                    # Teacher distillation on rollout (Section 4.3)
                    loss_teacher_roll = (per_t * w_win.view(-1) * w_t).mean()

            # == Jacobian contractivity: ||J||_F <= 1.5 (Section 4.3) ==
            # Jacobian stability: penalize expansive Jacobians
            jac_loss = torch.zeros((), device=device)
            if (not stage1) and (w_jac_eff > 0) and (jac_samples > 0):
                if mu_roll_pool is not None:
                    m = min(int(jac_samples), int(mu_roll_pool.size(0)))
                    idx = torch.randint(0, int(mu_roll_pool.size(0)), (m,), device=device)
                    y_sub = y_roll_pool[idx]
                    xprev_sub = mu_roll_pool[idx].detach().requires_grad_(True)


                    innovation_sub = None
                    if innov_on:
                        innovation_sub = _compute_innov_scalar(ystd_roll_pool[idx], xprev_sub.detach(),
                                                               t_indices[start:end][idx])
                else:
                    m = min(int(jac_samples), Bsz)
                    idx = torch.randint(0, Bsz, (m,), device=device)
                    y_sub = y_in[idx]
                    xprev_sub = Xprev_mix_feed[idx].detach().requires_grad_(True)


                    innovation_sub = None
                    if innov_on:
                        innovation_sub = _compute_innov_scalar(ystd[idx], xprev_sub.detach(), t_indices[idx])

                xy_sub = None
                if kappa_corr > 0.0:
                    out_xy = student(y_sub, x_prev=None, b_prev=None, sample=False, innov_prev=None)
                    xy_sub = out_xy[3]

                out_sub = student(
                    y_sub,
                    x_prev=xprev_sub,
                    b_prev=None,
                    sample=False,
                    x_yonly_prop=xy_sub,
                    kappa=float(kappa_corr),
                    innov_prev=innovation_sub,
                )
                xmu_sub = out_sub[3]

                grads = []
                for j in range(xmu_sub.size(1)):
                    g = torch.autograd.grad(
                        outputs=xmu_sub[:, j].sum(),
                        inputs=xprev_sub,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    grads.append(g.unsqueeze(1))
                J = torch.cat(grads, dim=1)
                # Frobenius norm of AR Jacobian
                J_frob = torch.sqrt((J ** 2).sum(dim=(1, 2)) + 1e-12)
                target = 1.5
                jac_loss = F.relu(J_frob - target).pow(2).mean()
                if DEBUG_PRINTS and ((ep <= DEBUG_EARLY_EPS) or (bi == 1) or (bi % 50 == 0)):
                    with torch.no_grad():
                        J_mean = float(J_frob.mean().item())
                        J_p95 = float(
                            torch.quantile(J_frob.detach().float(), 0.95).item()) if J_frob.numel() > 0 else float(
                            "nan")
                        pct_over = float((J_frob > float(target)).float().mean().item())
                        print(
                            f"[JAC][STAT] ep={ep:02d} b={bi:04d} m={int(J_frob.numel())} "
                            f"target={float(target):.3f} J_mean={J_mean:.3f} J_p95={J_p95:.3f} "
                            f"pct_over={pct_over:.3f} jac_loss={float(jac_loss.item()):.6f}",
                            flush=True
                        )

            # == Auxiliary: GAN + teacher + dynamics (Section 4.3) ==
            # GAN / teacher / geom / dyn-var
            do_gan = (enable_aux_after_recon and (not stage1) and (w_gan_eff > 0.0))
            do_teacher = (enable_aux_after_recon and (not stage1) and (w_teacher_eff > 0.0))
            do_geom = (enable_aux_after_recon and (not stage1) and (w_geom_eff > 0.0))

            loss_geom = torch.zeros((), device=device)
            loss_teacher = torch.zeros((), device=device)
            loss_gan = torch.zeros((), device=device)
            loss_dyn = torch.zeros((), device=device)
            loss_var = torch.zeros((), device=device)

            if do_gan:
                X_samp = X_mu + torch.exp(X_logsig) * torch.randn_like(X_mu)

                logits_real = disc(Xb.detach(), y_cond.detach())
                logits_fake = disc(X_samp.detach(), y_cond.detach())

                smooth = float(max(0.0, min(0.5, gan_label_smooth)))
                noise = float(max(0.0, min(0.2, gan_label_noise)))

                labels_real = (1.0 - smooth) + smooth * torch.rand_like(logits_real)
                labels_fake = (0.0 + smooth) * torch.rand_like(logits_fake)

                if noise > 0:
                    labels_real = (labels_real + noise * torch.randn_like(labels_real)).clamp(0.0, 1.0)
                    labels_fake = (labels_fake + noise * torch.randn_like(labels_fake)).clamp(0.0, 1.0)

                # GAN disc: real vs fake + label smoothing
                loss_D_real = F.binary_cross_entropy_with_logits(logits_real, labels_real)
                loss_D_fake = F.binary_cross_entropy_with_logits(logits_fake, labels_fake)
                loss_D = loss_D_real + loss_D_fake

                if gan_r1_gamma and gan_r1_gamma > 0:
                    Xb_r1 = Xb.detach().requires_grad_(True)
                    logits_real_r1 = disc(Xb_r1, y_cond.detach())
                    grad_real = torch.autograd.grad(
                        outputs=logits_real_r1.sum(),
                        inputs=Xb_r1,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    # R1 gradient penalty
                    r1_pen = 0.5 * gan_r1_gamma * grad_real.pow(2).sum(dim=1).mean()
                    loss_D = loss_D + r1_pen
                else:
                    r1_pen = torch.zeros((), device=device)

                opt_disc.zero_grad(set_to_none=True)
                loss_D.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_disc.step()

                with torch.no_grad():
                    prob_real = torch.sigmoid(logits_real).mean().item()
                    prob_fake = torch.sigmoid(logits_fake).mean().item()

                disc_loss_sum += float(loss_D.item()) * Xb.size(0)
                disc_real_mean_sum += prob_real * Xb.size(0)
                disc_fake_mean_sum += prob_fake * Xb.size(0)
                n_gan += Xb.size(0)

                # GAN gen: fool discriminator
                logits_fake_for_G = disc(X_samp, y_cond)
                loss_gan = F.binary_cross_entropy_with_logits(logits_fake_for_G, torch.ones_like(logits_fake_for_G))
                gan_loss_sum += float(loss_gan.item()) * Xb.size(0)

            loss_geom = torch.zeros((), device=device)
            if do_teacher:
                with torch.no_grad():
                    y_teacher_true_b, *_ = model_yx(_teacher_input_from_pred(Xb, t_indices),
                                                    cond_gain_scale=cond_gain_for_teacher)
                with torch.enable_grad():
                    y_teacher_hat_b, *_ = model_yx(_teacher_input_from_pred(X_mu, t_indices),
                                                   cond_gain_scale=cond_gain_for_teacher)
                loss_teacher = F.mse_loss(y_teacher_hat_b, y_teacher_true_b)

            if enable_aux_after_recon and (not stage1):
                prev_basis = Xprev_mix_feed
                d0_hat = (X_mu[0:1] - prev_basis[0:1])
                d0_true = (Xb[0:1] - prev_basis[0:1])
                loss_dyn0 = F.mse_loss(d0_hat, d0_true)

                if X_mu.size(0) > 1:
                    # Dynamics: step-to-step increments
                    dX_hat = X_mu[1:] - X_mu[:-1]
                    dX_true = Xb[1:] - Xb[:-1]
                    loss_dyn = F.mse_loss(dX_hat, dX_true) + loss_dyn0
                else:
                    loss_dyn = loss_dyn0

                eps = 1e-6
                var_true = Xb.var(dim=0, unbiased=False).detach()
                var_hat = X_mu.var(dim=0, unbiased=False)
                # Variance: penalise under-variance
                var_ratio = var_hat / (var_true + eps)
                loss_var = F.relu(1.0 - var_ratio).pow(2).mean()

            # TF correction magnitude (c_tf)
            loss_corr_mag_tf = c_tf.abs().mean()

            # == Total Loss (Section 4.3) ==
            # TOTAL LOSS (clean, no undefined symbols)
            loss = (
                    (w_mu * loss_mu)
                    + (w_delta * loss_delta)
                    + (20.0 * loss_var_tf)
                    + (5.0 * loss_prop)  # <--- ADD DIRECT PROPOSAL LOSS (weight 5.0)
                    + (50.0 * loss_gate_floor)
                    + (w_drift * loss_delta_bias)
                    + (w_obs_y_eff * loss_obs_y)
                    + (w_yonly * loss_yonly)
                    + (w_roll_eff * rollout_scale * loss_roll)
                    + (w_shift_eff * rollout_scale * loss_shift)
                    + (w_jac_eff * jac_loss)
                    + (w_bound_eff * loss_bound)
                    + (w_x_eff * loss_x_nll)
                    + (w_dyn_eff * loss_dyn)
                    + (w_var_eff * loss_var)
                    + loss_geom
                    + (w_teacher_eff * loss_teacher)
                    + (w_teacher_eff * float(w_teacher_roll_mult) * rollout_scale * loss_teacher_roll)
                    + (w_gan_eff * loss_gan)
                    + (0.20 * loss_corr_mag_tf)
                    + (0.50 * loss_corr_mag_roll)
                    + (0.50 * loss_corr_smooth_roll)
                    + (0.50 * rollout_scale * loss_ct_dc)  # <-- ADD
                    + (w_b_state * rollout_scale * loss_b_state)  # <-- ADD
                    + (30 * rollout_scale * loss_var_roll)
                    + (5.0 * rollout_scale * loss_spectral)

            )

            if not torch.isfinite(loss):
                print(f"[mem][ep {ep:02d}][b{bi:04d}] Non-finite total loss; skipping step")
                opt.zero_grad(set_to_none=True)
                if do_gan:
                    opt_disc.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # <--- FIX: Save kappa gradients (using pn to avoid shadowing n) --->
            kappa_grads_saved = {}
            for pn, p in student.named_parameters():
                if 'kappa_logits' in pn and p.grad is not None:
                    kappa_grads_saved[pn] = p.grad.clone()

            if DEBUG_PRINTS and innov_on and hasattr(student, 'innov_embed'):
                print(f"[INNOV][EMBED] ep={ep:02d} b={bi:04d} "
                      f"||W||={student.innov_embed.weight.norm().item():.6f} "
                      f"||b||={student.innov_embed.bias.norm().item():.6f}")

            # === INNOVATION GRADIENT SIGNAL: is the embedding learning? ===
            if DEBUG_PRINTS and innov_on and hasattr(student, 'innov_embed') and ((ep <= DEBUG_EARLY_EPS) or (bi == 1)):
                with torch.no_grad():
                    W = student.innov_embed.weight  # [d_mem, 1]
                    b = student.innov_embed.bias  # [d_mem]

                    # Effective dx contribution: how much does unit innovation change head_dx_y output?
                    # Chain: innov -> innov_embed -> z_innov -> head_dx_y
                    # Approx: head_dx_y.weight @ innov_embed.weight  (ignoring tanh)
                    effective_gain = (student.head_dx_y.weight @ W).squeeze()  # [d_x]
                    print(
                        f"[INNOV][GAIN] ep={ep:02d} b={bi:04d} "
                        f"effective_gain_per_dim=[{', '.join(f'{v:+.4f}' for v in effective_gain.tolist())}] "
                        f"||W_embed||={W.norm().item():.6f} "
                        f"||b_embed||={b.norm().item():.6f}",
                        flush=True
                    )

            # Stage 2b clip: protect from rollout gradients (Section 3.6)
            # --- FIX: Separate yonly clip (protects backbone from rollout gradient dominance) ---
            nn.utils.clip_grad_norm_(yonly_sep_params, 5.0)

            # Tighter clip near fully closed-loop (Section 3.6)
            # --- FIX 1B: Global clip — tighter when rollout windows are long ---
            if ss_p_eff > 0.90:
                nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            else:
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)

            # <--- FIX: Restore kappa gradients (using pn to avoid shadowing n) --->
            for pn, p in student.named_parameters():
                if pn in kappa_grads_saved:
                    p.grad = kappa_grads_saved[pn]
                    # Gentle per-param clip: max 1.0 per element (plenty for 4 logits)
                    p.grad.clamp_(-1.0, 1.0)

            opt.step()

            with torch.no_grad():
                last_prev_train = _clamp_feed(X_mu[-1:]).detach() if (
                            clamp_x_rollout and clamp_x_rollout > 0) else X_mu[-1:].detach()

            loss_sum += float(loss.item()) * Xb.size(0)
            n += Xb.size(0)

            if (bi == 1) or (bi % 25 == 0) or (bi == batches_total):
                elapsed = time.time() - t_ep0
                pct = 100.0 * bi / max(1, batches_total)
                ex_per_s = n / max(1e-6, elapsed)  # examples/sec (approx)
                print(
                    f"[mem][ep {ep:02d}] batch {bi}/{batches_total} ({pct:5.1f}%) | elapsed {elapsed:6.1f}s | ex/s {ex_per_s:7.1f}")


        if n_gan > 0:
            gan_loss_mean = gan_loss_sum / n_gan
            disc_loss_mean = disc_loss_sum / n_gan
            disc_real_mean = disc_real_mean_sum / n_gan
            disc_fake_mean = disc_fake_mean_sum / n_gan
        else:
            gan_loss_mean = 0.0
            disc_loss_mean = 0.0
            disc_real_mean = 0.0
            disc_fake_mean = 0.0

        # Always print TF metrics (you already do this)
        student.eval()
        with torch.no_grad():
            tf_r2_now, tf_mae_now = _tf_metrics_small(kappa_corr=float(kappa_corr))
        tf_r2_prev = float(tf_r2_now)

        # == Epoch evaluation: TF/rollout metrics, drift probe ==
        # Decide whether to run heavy eval
        do_full_eval = ((eval_every is None) or (eval_every <= 1) or (ep % eval_every == 0) or (ep == epochs))

        if not do_full_eval:
            print(
                f"[mem][ep {ep:02d}] total={loss_sum / max(1, n):.6f} | X_TF_R2={tf_r2_now:.3f} X_TF_MAE={tf_mae_now:.4f} "
                f"| gan_loss={gan_loss_mean:.4f} D_loss={disc_loss_mean:.4f}")
            continue

        # --- FULL EVAL: compute TF + rollout on deterministic dl_eval ---
        X_true_np, X_tf_np, X_roll_np = _collect_tf_and_roll_for_plots(
            dl_eval,
            kappa_corr=float(kappa_corr),
            rollout_streams=1,  # CRITICAL FIX: Match export bias integration horizon
            use_amp=True,
            progress_every=5,
            max_batches=None,  # set e.g. 10 to speed up sanity checks
        )

        # === NEW: Y-ONLY DIAGNOSTIC CALL ===
        X_true_yo, X_yonly_diag = _collect_yonly_diagnostic(dl_eval)
        _print_yonly_diagnostic(X_true_yo, X_yonly_diag, ep)

        tf_r2_full, tf_mae_full = _r2_mae_np(X_true_np, X_tf_np)
        roll_r2_full, roll_mae_full = _r2_mae_np(X_true_np, X_roll_np)
        print(f"[ENC][ROLL_METRICS] ep={ep:02d} roll_r2={roll_r2_full:.4f} roll_mae={roll_mae_full:.4f}", flush=True)
        print(f"[ENC][TF_METRICS]   ep={ep:02d} tf_r2={tf_r2_full:.4f} tf_mae={tf_mae_full:.4f}", flush=True)

        # --- DEBUG: per-dim rollout variance ratio (smoothing diagnostic) ---
        with torch.no_grad():
            X_true_t = torch.from_numpy(X_true_np)
            X_roll_t = torch.from_numpy(X_roll_np)
            for j, nm in enumerate(["DBP", "SBP", "CO", "SV"]):
                vr = float((X_roll_t[:, j].var() / X_true_t[:, j].var().clamp(min=1e-12)).item())
                r2_j, _ = _r2_mae_np(X_true_np[:, j], X_roll_np[:, j])
                print(f"[ROLL_VAR] ep={ep:02d} {nm}: var_ratio={vr:.3f} R2={r2_j:+.4f}")

        last_full_roll_r2 = float(roll_r2_full)

        print(
            f"[mem][ep {ep:02d}] total={loss_sum / max(1, n):.6f} | "
            f"TF: R2={tf_r2_full:.3f} MAE={tf_mae_full:.4f} | "
            f"ROLL: R2={roll_r2_full:.3f} MAE={roll_mae_full:.4f} | "
            f"gan_loss={gan_loss_mean:.4f} D_loss={disc_loss_mean:.4f} "
            f"| D(real)≈{disc_real_mean:.3f} D(fake)≈{disc_fake_mean:.3f}"
        )

        # === INNOVATION ABLATION: does innovation actually help rollout? ===
        if innov_on and (ep >= recon_epochs + yonly_pretrain_epochs + 3):
            # Temporarily disable innovation for this quick 5-batch ablation
            _DISABLE_INNOV_DIAG_save = _DISABLE_INNOV_DIAG

            # Using global state modification safely to match your existing flag architecture
            globals()['_DISABLE_INNOV_DIAG'] = True

            _, _, X_roll_no_innov = _collect_tf_and_roll_for_plots(
                dl_eval,
                kappa_corr=float(kappa_corr),
                rollout_streams=1,
                use_amp=True,
                progress_every=None,
                max_batches=5,  # just 5 batches = fast
            )

            # Restore the flag
            globals()['_DISABLE_INNOV_DIAG'] = _DISABLE_INNOV_DIAG_save

            r2_with = roll_r2_full
            # Only compare against the first N samples that the 5-batch ablation actually produced
            r2_without, _ = _r2_mae_np(X_true_np[:X_roll_no_innov.shape[0]], X_roll_no_innov)
            delta_r2 = r2_with - r2_without

            print(
                f"[INNOV][ABLATION] ep={ep:02d} "
                f"roll_R2_with_innov={r2_with:.4f} "
                f"roll_R2_without={r2_without:.4f} "
                f"delta_R2={delta_r2:+.4f} "
                f"{'HELPING' if delta_r2 > 0.005 else 'NEUTRAL' if delta_r2 > -0.005 else 'HURTING'}",
                flush=True
            )
            # --- DRIFT PROBE (single-stream; horizon table) ---
        _drift_probe_report(
            ep=ep,
            kappa_corr=float(kappa_corr),
            horizons=(32, 128, 512, 2048),
            max_steps=3000,  # reduce if slow
        )

        # --- SAVE PLOTS (True vs TF vs Rollout) ---
        times_used = times_full[idx_center] if (USE_WAVENET_STUDENT and (idx_center is not None)) else times_full
        times_used = times_used[:len(X_true_np)]
        save_csv = (eval_every is not None) and (eval_every > 0) and (ep % (eval_every * 2) == 0)
        _save_x_diagnostics(
            out_dir=diag_dir,
            ep=ep,
            times=times_used,
            X_true_np=X_true_np,
            X_pred_np=X_roll_np,  # legacy slot; treat as rollout
            dim_names=dim_names,
            max_points_plot=diag_max_points_plot,
            X_pred_tf_np=X_tf_np,
            X_pred_roll_np=X_roll_np,
            save_csv=save_csv
        )

        # === SAVE Y-ONLY vs TRUTH plots to separate folder ===
        diag_dir_yonly = os.path.join(diag_dir, "yonly_only")
        _save_x_diagnostics(
            out_dir=diag_dir_yonly,
            ep=ep,
            times=times_used,
            X_true_np=X_true_yo,
            X_pred_np=X_yonly_diag,
            dim_names=dim_names,
            max_points_plot=diag_max_points_plot,
            X_pred_tf_np=None,
            X_pred_roll_np=None,
            save_csv=save_csv,
        )
        print(f"[mem][ep {ep:02d}] Saved Y-only vs Truth plots -> {diag_dir_yonly}")

        print(f"[mem][ep {ep:02d}] Saved X diagnostics (True/TF/Rollout) -> {diag_dir}")

        # Full eval (TF vs rollout) – existing evaluation block can remain,
        # but it should use _student_forward_consistent + innovation the same way as training.
        # For brevity, we keep your structure unchanged below this point except where necessary.

    payload = {
        "student_state": student.state_dict(),
        "d_mem": 512,
        "est_fs": float(est_fs),
        "y_only_mu": y_only_mu.squeeze().tolist(),
        "y_only_sigma": y_only_sd.squeeze().tolist(),
        "y_only_feat_names": yonly_names,
        "teacher_mean": T_mean.squeeze().tolist(),
        "teacher_std": T_std.squeeze().tolist(),
        "teacher_source": getattr(model_yx, "teacher_source_mode", "taps") if used_whitener else "projection",
        "student_type": "wavenet" if USE_WAVENET_STUDENT else "mlp",
        "wavenet_win": WAVENET_WIN if USE_WAVENET_STUDENT else None,
        "ar_xprev_priority": True,
        "delta_scale": float(_DELTA_SCALE),
        "ss_p_start": float(ss_p_start),
        "ss_p_end": float(ss_p_end),
        "ss_warmup_epochs": int(ss_warmup_epochs),
        "x_prev_noise_std": float(x_prev_noise_std),
        "geom_mult_start": float(geom_mult_start),
        "geom_mult_end": float(geom_mult_end),
        "geom_ramp_epochs": int(geom_ramp_epochs),
        "eval_every": int(eval_every),
        "diag_dir": str(diag_dir),
        "recon_epochs": int(recon_epochs),
        "freeze_sigma_epochs": int(freeze_sigma_epochs),
        "fixed_logsig": float(fixed_logsig),
        "w_mu": float(w_mu),
        "w_delta": float(w_delta),
        "w_obs_y": float(w_obs_y),
        "enable_aux_after_recon": bool(enable_aux_after_recon),
        "clamp_x_rollout": float(clamp_x_rollout),
        "stage1_min_tf_r2": float(stage1_min_tf_r2),
        "hist_p_start": float(hist_p_start),
        "hist_p_end": float(hist_p_end),
        "hist_warmup_epochs": int(hist_warmup_epochs),
        "w_yonly_start": float(w_yonly_start),
        "w_yonly_end": float(w_yonly_end),
        "yonly_ramp_epochs": int(yonly_ramp_epochs),
        "w_roll": float(w_roll),
        "roll_gamma": float(roll_gamma),
        "w_jac": float(w_jac),
        "jac_samples": int(jac_samples),
        "w_bound": float(w_bound),
        "bound_soft": float(bound_soft),
        "bound_power": float(bound_power),
        "detach_hist_pred": bool(detach_hist_pred),
        "shift_kl_use_unclamped": bool(shift_kl_use_unclamped),
        "corr_kappa_max": float(corr_kappa_max),
        "corr_ramp_epochs": int(corr_ramp_epochs),
        "use_innovation_feedback": bool(use_innovation_feedback),
        "w_teacher_roll_mult": float(w_teacher_roll_mult),
        "innov_disabled": bool(_DISABLE_INNOV_DIAG),
        "ar_disabled": bool(_DISABLE_AR_DIAG),
        "kappa_infer": float(corr_kappa_max),
        # --- Architecture params (for _build_student_from_payload) ---
        "hidden": 112,
        "layers": 6,
        "d_x": 4,

        "g_floor": float(student.g_floor.item()),
        "spectral_norm_ar": bool(_USE_SPECTRAL_NORM_AR),

        # --- Scalers for inference (so decoder pack is self-contained) ---
        "x_norm_mu": scaler_obj["x_norm_mu"],
        "x_norm_sigma": scaler_obj["x_norm_sigma"],
        "x_feat_mu_state": list(np.array(scaler_obj["x_feat_mu"], dtype=np.float32)[:4].tolist()),
        "x_feat_sigma_state": list(np.array(scaler_obj["x_feat_sigma"], dtype=np.float32)[:4].tolist()),
        "y_mu": scaler_obj["y_mu"],
        "y_sigma": scaler_obj["y_sigma"],

    }
    payload["student_module"] = "prior_student"
    payload["student_class_name"] = "YMemoryWaveNet"  # match what you used

    _ensure_dir(save_path)
    torch.save(payload, save_path)
    print(f"[mem] Saved geometry-aware Y→X encoder -> {save_path}")
    return payload


@torch.no_grad()
# --- Reconstruct student from saved payload ---
def _build_student_from_payload(payload: dict, d_yfeat: int, device: str):
    """
    Recreate the student (encoder prior) module from the saved payload.
    Loads state on CPU first (fast), then moves to device.
    """
    import time
    t0 = time.time()

    student_type = payload.get("student_type", "wavenet")
    d_mem = int(payload["d_mem"])
    win = payload.get("wavenet_win", None)

    # ---- IMPORTANT: build on CPU (do NOT .to(device) yet) ----
    if student_type == "wavenet":
        assert win is not None, "Payload missing wavenet_win"

        # Spectral norm check: must match training flag
        sn_flag = payload.get("spectral_norm_ar", None)
        if sn_flag is not None and bool(sn_flag) != _USE_SPECTRAL_NORM_AR:
            raise RuntimeError(
                f"[FATAL] spectral_norm_ar mismatch: payload={sn_flag} but module-level={_USE_SPECTRAL_NORM_AR}. "
                f"AR weights will be silently dropped. Set _USE_SPECTRAL_NORM_AR={sn_flag} before loading."
            )

        student = YMemoryWaveNet(
            d_yfeat=d_yfeat,
            d_mem=int(payload.get("d_mem", 512)),
            hidden=int(payload.get("hidden", 112)),
            layers=int(payload.get("layers", 6)),
            dilations=None,
            d_x=int(payload.get("d_x", 4)),
        )

    t1 = time.time()
    missing, unexpected = student.load_state_dict(payload["student_state"], strict=False)
    print(f"[time] load_state_dict (CPU): {time.time() - t1:.3f}s", flush=True)
    if missing:
        print(f"[WARN] student.load_state_dict MISSING keys ({len(missing)}): {missing[:10]}...", flush=True)
        print(f"[WARN] These weights are RANDOMLY INITIALIZED. This likely means architecture mismatch!", flush=True)
    if unexpected:
        print(f"[WARN] student.load_state_dict UNEXPECTED keys ({len(unexpected)}): {unexpected[:10]}...", flush=True)

    t2 = time.time()

    # <--- FIX 4: Warn if global diagnostic flags differ from payload --->
    if payload.get("innov_disabled") != _DISABLE_INNOV_DIAG:
        print(
            f"[WARN] innov_disabled mismatch: payload={payload.get('innov_disabled')} vs module={_DISABLE_INNOV_DIAG}")
    if payload.get("ar_disabled") != _DISABLE_AR_DIAG:
        print(f"[WARN] ar_disabled mismatch: payload={payload.get('ar_disabled')} vs module={_DISABLE_AR_DIAG}")

    student = student.to(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[time] to(device): {time.time() - t2:.3f}s", flush=True)
    print(f"[time] total build+load: {time.time() - t0:.3f}s", flush=True)

    student.eval()
    return student


@torch.no_grad()
# ============================================================================
# Export decoder pack: scalers, weights, streams (Section 5)
# ============================================================================
def export_decoder_pack(
        *,
        folder: str,
        ygx_ckpt_path: str,
        scaler_path: str,
        y_memory_encoder_path: str,
        out_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cond_gain_for_teacher: float = 20.0,
):
    """
    Exports EVERYTHING needed to train a decoder:
      - scalers/metadata
      - window alignment (idx_center)
      - encoder prior streams: mu/logsig/b/c/z (aligned to samples)
      - decoder inputs: y windows (or y_only_std)
      - decoder targets: X_true_std (if available from folder)
      - innovation stream (aligned)
    """
    import json
    import numpy as np
    import torch
    from contextlib import nullcontext

    # ----------------------------
    # Helpers (define BEFORE use)
    # ----------------------------

    @torch.no_grad()
    def _student_forward_batched(
            student,
            y_in_cpu: torch.Tensor,
            x_prev_cpu: torch.Tensor | None,
            innov_cpu: torch.Tensor | None,
            *,
            kappa_corr: float = 0.0,
            batch_size: int = 1024,
            use_amp: bool = True
    ):

        student.eval()

        outs = {"z": [], "mu": [], "logsig": [], "b": [], "c": []}
        Nloc = int(y_in_cpu.size(0))

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
            else nullcontext()
        )

        # Compute compound kappa matching training
        if hasattr(student, 'kappa_vec') and kappa_corr > 0.0:
            kappa_eff_export = float(kappa_corr) * student.kappa_vec().view(1, -1)
        else:
            kappa_eff_export = float(kappa_corr)

        for s in range(0, Nloc, batch_size):
            e = min(Nloc, s + batch_size)

            y_b = y_in_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
            xprev_b = None if x_prev_cpu is None else x_prev_cpu[s:e].to(device=device, dtype=torch.float32,
                                                                         non_blocking=True)
            innov_b = None if innov_cpu is None else innov_cpu[s:e].to(device=device, dtype=torch.float32,
                                                                       non_blocking=True)

            xy_b = None
            if (kappa_corr > 0.0) and (xprev_b is not None):
                with amp_ctx:
                    out_yonly = student(
                        y_b, x_prev=None, b_prev=None, sample=False, innov_prev=None,
                    )
                xy_b = out_yonly[3]

            with amp_ctx:
                out = student(
                    y_b,
                    x_prev=xprev_b,
                    b_prev=None,
                    sample=False,
                    x_yonly_prop=xy_b,
                    kappa=kappa_eff_export,
                    innov_prev=innov_b,
                )

            outs["z"].append(out[0].detach().cpu())
            outs["mu"].append(out[3].detach().cpu())
            outs["logsig"].append(out[4].detach().cpu())
            outs["b"].append(out[6].detach().cpu())
            outs["c"].append(out[7].detach().cpu())

            del y_b, xprev_b, innov_b, out

        return {k: torch.cat(v, dim=0) for k, v in outs.items()}

    @torch.no_grad()
    def _clamp_std(x: torch.Tensor, clamp_val: float | None):
        if clamp_val is None or clamp_val <= 0:
            return x
        return x.clamp(-float(clamp_val), float(clamp_val))

    def _tstats(name, t: torch.Tensor):
        t = t.detach().float().cpu()
        mu = t.mean(dim=0).numpy()
        sd = t.std(dim=0, unbiased=False).numpy()
        mn = t.min(dim=0).values.numpy()
        mx = t.max(dim=0).values.numpy()
        fmt = lambda a: "[" + ", ".join(f"{v:+.3f}" for v in a.tolist()) + "]"
        print(f"[EXPORT][STATS] {name} mean={fmt(mu)} std={fmt(sd)} min={fmt(mn)} max={fmt(mx)}", flush=True)

        # ----------------------------

    # Export diagnostics helpers
    # ----------------------------
    def _prior_metrics(tag: str, mu_cpu: torch.Tensor, x_true_cpu: torch.Tensor):
        """
        mu_cpu: [Nc,4] CPU tensor
        x_true_cpu: [Nc,4] CPU tensor
        Prints R2 + MAE + per-dim MAE in std-space.
        """
        assert mu_cpu.shape == x_true_cpu.shape, f"{tag}: mu{tuple(mu_cpu.shape)} != x_true{tuple(x_true_cpu.shape)}"
        mu_np = mu_cpu.detach().cpu().numpy()
        xt_np = x_true_cpu.detach().cpu().numpy()
        r2, mae = _r2_mae_np(xt_np, mu_np)
        mae_dim = np.mean(np.abs(mu_np - xt_np), axis=0)
        print(
            f"[EXPORT][PRIOR] {tag} | R2={r2:.4f} MAE={mae:.4f} | "
            f"MAE_dim=[{', '.join(f'{v:.4f}' for v in mae_dim.tolist())}]",
            flush=True
        )
        return float(r2), float(mae), mae_dim

    def _report_roll_windows(
            tag_prefix: str,
            mu_roll_cpu: torch.Tensor,  # [Nc,4] CPU
            prevfed_roll_cpu: torch.Tensor,  # [Nc,4] CPU (clamped feed stream)
            X_true_cent_cpu: torch.Tensor,  # [Nc,4] CPU
            X_prev_cent_feed_cpu: torch.Tensor,  # [Nc,4] CPU (TF roll-in, clamped)
            windows: list[tuple[str, int, int]],
    ):
        """
        Prints R2/MAE for rollout state and KL/MAE for rollout roll-in stream
        on specified index windows.
        """
        Nc = int(mu_roll_cpu.size(0))

        for name, a, b in windows:
            a = int(max(0, a))
            b = int(min(Nc, b))
            if b <= a + 1:
                print(f"[EXPORT][WINDOW] {tag_prefix}:{name} skipped (a={a}, b={b}, Nc={Nc})", flush=True)
                continue

            sl = slice(a, b)

            # 1) rollout prediction quality vs truth
            _prior_metrics(
                f"{tag_prefix}:{name} mu_roll vs X_true | idx=[{a}:{b}]",
                mu_roll_cpu[sl],
                X_true_cent_cpu[sl],
            )

            # 2) roll-in distribution shift (TF prev vs rollout-fed prev)
            _rollin_shift_metrics(
                f"{tag_prefix}:{name} prevfed_roll vs X_prev_tf_feed | idx=[{a}:{b}]",
                prevfed_roll_cpu[sl],
                X_prev_cent_feed_cpu[sl],
            )

    def _rollin_shift_metrics(tag: str, prevfed_cpu: torch.Tensor, xprev_tf_cpu: torch.Tensor):
        """
        prevfed_cpu: [Nc,4] CPU tensor (what rollout actually fed each step)
        xprev_tf_cpu: [Nc,4] CPU tensor (TF roll-in prev, i.e., X_prev_cent_cpu)
        Prints: KL(N_tf || N_roll) in diag-Gauss, plus mean shift + std ratio.
        """
        assert prevfed_cpu.shape == xprev_tf_cpu.shape, (
            f"{tag}: prevfed{tuple(prevfed_cpu.shape)} != xprev_tf{tuple(xprev_tf_cpu.shape)}"
        )

        m_tf = xprev_tf_cpu.mean(dim=0)
        v_tf = xprev_tf_cpu.var(dim=0, unbiased=False).clamp_min(1e-6)

        m_rl = prevfed_cpu.mean(dim=0)
        v_rl = prevfed_cpu.var(dim=0, unbiased=False).clamp_min(1e-6)

        kl = diag_gauss_kl(m_tf, v_tf, m_rl, v_rl).mean().item()
        mean_shift = (m_rl - m_tf).detach().cpu().numpy()
        std_ratio = torch.sqrt(v_rl / v_tf).detach().cpu().numpy()

        # also a direct MAE between roll-in streams (not a distribution metric)
        mae_rollin = (prevfed_cpu - xprev_tf_cpu).abs().mean().item()

        print(
            f"[EXPORT][SHIFT] {tag} | KL(tf||roll)={kl:.6f} | rollin_MAE={mae_rollin:.4f} | "
            f"mean_shift=[{', '.join(f'{v:+.4f}' for v in mean_shift.tolist())}] | "
            f"std_ratio=[{', '.join(f'{v:.3f}' for v in std_ratio.tolist())}]",
            flush=True
        )
        return float(kl), float(mae_rollin), mean_shift, std_ratio

    @torch.no_grad()
    def _compute_innovation_scalar_export(y_obs, x_prev_fed, x_cond_row, ygx_model, cond_gain):
        cond_in = x_cond_row.clone()
        cond_in[:, :D_STATE] = x_prev_fed
        y_hat = ygx_model(_pad_for_teacher(cond_in), cond_gain_scale=cond_gain)[0]
        return (y_obs - y_hat).detach()  # [B, 1] — NO Jacobian

    @torch.no_grad()
    def _student_rollout_single_stream(
            student,
            ygx_model,
            y_in_cpu: torch.Tensor,
            y_std_cpu: torch.Tensor,
            x0_cpu: torch.Tensor,
            *,
            X_cond_cent_cpu: torch.Tensor,  # Current T (for Innovation)
            X_prev_cond_cent_cpu: torch.Tensor,  # Previous T-1 (for State)
            d_state: int = 4,
            clamp_val: float | None = None,
            use_innovation: bool = True,
            expected_use_innovation: bool | None = None,
            kappa_corr: float = 0.0,
            use_amp: bool = True,
            chunk_size: int = 1024,
            progress_every: int = 2000,
            warmup_steps: int = 0,
            X_true_cpu: torch.Tensor | None = None,
    ):

        student.eval()
        ygx_model.eval()
        print(
            f"[rollout][DEBUG] ENTER _student_rollout_single_stream | device={device} | Nc={int(y_in_cpu.size(0))} | y_in_cpu.shape={tuple(y_in_cpu.shape)} | y_std_cpu.shape={tuple(y_std_cpu.shape)}")
        if expected_use_innovation is not None:
            assert use_innovation == expected_use_innovation, (
                f"[INNOV][ROLL][ASSERT] use_innovation={use_innovation} "
                f"!= expected_use_innovation={expected_use_innovation}"
            )

        Nc = int(y_in_cpu.size(0))
        d_x = int(x0_cpu.numel())

        # --- state lives on GPU ---
        prev = x0_cpu.view(1, -1).to(device=device, dtype=torch.float32)
        if clamp_val is not None and clamp_val > 0:
            prev = prev.clamp(-float(clamp_val), float(clamp_val))

        b_prev = torch.zeros((1, d_x), device=device, dtype=torch.float32)

        # --- preallocate outputs on GPU (fast) ---
        mu_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
        b_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
        c_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
        prevfed = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
        prev_unclamped = torch.empty((Nc, d_x), device=device, dtype=torch.float32)  # FIX 3

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
            else nullcontext()
        )
        # Compute compound kappa matching training

        innov_tracker = InnovTracker()

        # Compute compound kappa matching training
        if hasattr(student, 'kappa_vec') and kappa_corr > 0.0:
            kappa_eff_export = float(kappa_corr) * student.kappa_vec().view(1, -1)
        else:
            kappa_eff_export = float(kappa_corr)

        t0 = time.time()
        t_global0 = time.time()
        print(
            f"[rollout][DEBUG] starting chunks: chunk_size={chunk_size} progress_every={progress_every} use_innovation={use_innovation} use_amp={use_amp}")

        t0 = time.time()
        # 8-step innovation buffer (Section 2.6)
        innov_buffer_export = torch.zeros((1, 8), device=device, dtype=torch.float32)

        # --- iterate in CHUNKS to avoid 50k PCIe copies ---
        for s in range(0, Nc, chunk_size):
            e = min(Nc, s + chunk_size)
            # print(f"[rollout][DEBUG] chunk s={s} e={e} (len={e-s})")
            t_copy0 = time.time()

            # one CPU->GPU copy per chunk (not per timestep)
            y_chunk = y_in_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
            yobs_chunk = y_std_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)

            cond_chunk_curr = None
            if use_innovation:
                cond_chunk_curr = X_cond_cent_cpu[s:e].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )  # [chunk, D_COND] — current timestep T for innovation

            for i in range(e - s):
                t = s + i

                prev_unclamped[t] = prev.squeeze(0)

                # clamp prev for feeding
                prev_fed = prev
                if clamp_val is not None and clamp_val > 0:
                    prev_fed = prev_fed.clamp(-float(clamp_val), float(clamp_val))

                prevfed[t] = prev_fed.squeeze(0)

                innovation = None
                y_t = y_chunk[i:i + 1]

                if use_innovation:
                    innovation = _compute_innovation_scalar_export(
                        y_std_cpu[t:t + 1].to(device=device, dtype=torch.float32),
                        prev_fed,
                        X_cond_cent_cpu[s + i:s + i + 1].to(device=device, dtype=torch.float32),
                        ygx_model,
                        cond_gain_for_teacher
                    )
                    innov_tracker.record(innovation)
                    if innovation is not None:
                        innov_buffer_export = torch.cat([innov_buffer_export[:, 1:], innovation.view(1, 1)], dim=1)

                xy = None
                if kappa_corr > 0.0:
                    out_yonly = student(
                        y_t, x_prev=None, b_prev=None, sample=False, innov_prev=None
                    )
                    xy = out_yonly[3]

                with amp_ctx:
                    out = student(
                        y_t,
                        x_prev=prev_fed,
                        b_prev=b_prev,
                        sample=False,
                        x_yonly_prop=xy,
                        kappa=kappa_eff_export,
                        innov_prev=innov_buffer_export
                    )

                xmu = out[3]
                b_prev = out[6]
                c_t = out[7]

                mu_roll[t] = xmu.squeeze(0)
                b_roll[t] = b_prev.squeeze(0)
                c_roll[t] = c_t.squeeze(0)

                # === WARM-START ===
                if (warmup_steps > 0) and (t < warmup_steps) and (X_true_cpu is not None):
                    prev = X_true_cpu[t:t + 1].to(device=device, dtype=torch.float32)
                    if clamp_val is not None and clamp_val > 0:
                        prev = prev.clamp(-float(clamp_val), float(clamp_val))
                else:
                    prev = xmu

            if (progress_every is not None) and (progress_every > 0) and (e % progress_every == 0 or e == Nc):
                elapsed = time.time() - t0
                it_s = e / max(1e-9, elapsed)
                print(f"[rollout] t={e}/{Nc} | {it_s:.2f} it/s | elapsed={elapsed:.1f}s")

            # free chunk tensors
            del y_chunk, yobs_chunk

        innov_tracker.print_summary(tag=f" [SINGLE_STREAM]")

        # ONE transfer back to CPU (fast)
        return mu_roll.cpu(), b_roll.cpu(), c_roll.cpu(), prevfed.cpu(), prev_unclamped.cpu()

    # ----------------------------
    # Load data + scalers
    # ----------------------------
    df = load_folder_and_join(folder)

    # CRITICAL: match training detrending
    y_raw = df["ShortChannel"].to_numpy(np.float32)
    y_smooth = uniform_filter1d(y_raw, size=600)
    df["ShortChannel"] = y_raw - y_smooth

    with open(scaler_path, "r") as f:
        sc = json.load(f)

    d_teacher_export = len(sc["x_feat_mu"])

    def _pad_for_teacher(x4):
        if x4.shape[-1] == d_teacher_export:
            return x4
        pad = torch.zeros(*x4.shape[:-1], d_teacher_export - x4.shape[-1], device=x4.device, dtype=x4.dtype)
        return torch.cat([x4, pad], dim=-1)

    # ---- build X_true in x_feat_std space ----
    x_norm_mu = np.array(sc["x_norm_mu"], dtype=np.float32)
    x_norm_sd = np.array(sc["x_norm_sigma"], dtype=np.float32)
    _, x_feat, feat_names, _ = build_x_features_from_df(
        df, x_norm_mu=x_norm_mu, x_norm_sd=x_norm_sd, fit_slice=None, save_stats_path=None
    )
    # x_feat is [N,4] (base only). Apply scaling with FIRST 4 dims of scaler.
    x_feat_std = apply_scaling_np(
        x_feat,
        np.array(sc["x_feat_mu"], dtype=np.float32)[:x_feat.shape[1]],
        np.array(sc["x_feat_sigma"], dtype=np.float32)[:x_feat.shape[1]],
    ).astype(np.float32)

    # ----------------------------
    # Reconstruct full teacher-dim array (4 base + lagged features = 16)
    # This matches what make_loaders_from_folder builds during training.
    y_only_tmp, yonly_names_tmp = build_y_only_features(df)
    bp_names = ['Y_bp_energy_HR', 'Y_bp_envelope_HR', 'Y_bp_energy_RR',
                'Y_bp_envelope_RR', 'Y_bp_energy_Trend', 'Y_bp_cardiac_raw']
    bp_indices = [yonly_names_tmp.index(n) for n in bp_names if n in yonly_names_tmp]
    y_bp_raw = y_only_tmp[:, bp_indices]

    x_feat_full_raw = np.concatenate([x_feat, y_bp_raw], axis=1)

    x_feat_std_full = apply_scaling_np(
        x_feat_full_raw,
        np.array(sc["x_feat_mu"], dtype=np.float32),
        np.array(sc["x_feat_sigma"], dtype=np.float32)
    ).astype(np.float32)

    print(f"[export] x_feat_std shape: {x_feat_std.shape} (base 4-dim)")
    print(
        f"[export] x_feat_std_full shape: {x_feat_std_full.shape} (full teacher-dim, expected {len(sc['x_feat_mu'])})")

    # ----------------------------
    # STATE vs COND split (CRITICAL)
    # ----------------------------
    D_STATE = 4
    D_COND = int(x_feat_std_full.shape[1])  # 16 (what ygx expects)
    X_cond_std = x_feat_std_full.astype(np.float32)  # [N, D_COND=16]
    X_state_std = X_cond_std[:, :D_STATE]  # [N, 4]  (decoder targets / student state)
    X_true_std = X_state_std.astype(np.float32)

    # ----------------------------
    # x0 strategy (std-space)
    # ----------------------------

    # ---- build y_std ----
    y_scalar = df["ShortChannel"].to_numpy(np.float32).reshape(-1, 1)
    y_mu = np.array(sc["y_mu"], dtype=np.float32)
    y_sd = np.array(sc["y_sigma"], dtype=np.float32)
    y_std = apply_scaling_np(y_scalar, y_mu, y_sd).astype(np.float32)

    # ---- y_only_std using encoder payload stats ----
    y_only, yonly_names = build_y_only_features(df)
    payload = torch.load(y_memory_encoder_path, map_location="cpu")
    est_fs_export = float(payload.get("est_fs", 30.0))
    y_only, yonly_names = build_y_only_features(df, fs_hint=est_fs_export)
    # --- FIX: Define clamp_val early so it can be used for feed clamping ---
    clamp_val = float(payload.get("clamp_x_rollout", 0.0))
    clamp_val = None if (clamp_val is None or clamp_val <= 0) else clamp_val

    # >>> Authoritative export innovation flag (DO NOT use module globals here)
    innov_disabled_export = bool(payload.get("innov_disabled", False))
    innov_on_export = _innov_enabled(
        use_innovation_feedback=bool(payload.get("use_innovation_feedback", True)),
        innov_disabled=innov_disabled_export
    )
    print(
        f"[INNOV][EXPORT] payload.use_innovation_feedback={payload.get('use_innovation_feedback')} "
        f"payload.innov_disabled={payload.get('innov_disabled')} => innov_on_export={innov_on_export}",
        flush=True
    )

    y_only_mu = np.array(payload["y_only_mu"], dtype=np.float32)
    y_only_sd = np.array(payload["y_only_sigma"], dtype=np.float32)
    y_only_std = apply_scaling_np(y_only, y_only_mu, y_only_sd).astype(np.float32)
    d_yfeat = int(y_only_std.shape[1])

    # ----------------------------
    # Load teacher model (YGivenX) for innovation
    # ----------------------------
    # (warmup + load weights)
    x_feat_std_t = torch.from_numpy(x_feat_std_full).to(device=device, dtype=torch.float32)
    # Load frozen teacher for innovation
    ygx = YGivenXModel(d_x_cond=D_COND, cond_gain_y=cond_gain_for_teacher).to(device)
    _warmup_y_only(ygx, device)
    state = torch.load(ygx_ckpt_path, map_location=device)
    target_sd = ygx.state_dict()
    filtered = {k: v for k, v in state.items() if (k in target_sd and v.shape == target_sd[k].shape)}
    ygx.load_state_dict(filtered, strict=False)
    reset_actnorm_flags(ygx)
    ygx.eval()
    del x_feat_std_t  # free GPU memory early

    # ----------------------------
    # Recreate encoder student (the prior)
    # ----------------------------
    # Reconstruct student encoder
    student = _build_student_from_payload(payload, d_yfeat=d_yfeat, device=device)

    # ----------------------------
    # Windowing (KEEP y_in on CPU to avoid GPU OOM)
    # ----------------------------
    student_type = payload.get("student_type", "wavenet")

    if student_type == "wavenet":
        WIN = int(payload["wavenet_win"])
        # Grab lags early so we can filter by max_lag

        y_only_t = torch.from_numpy(y_only_std).to(dtype=torch.float32)  # CPU [N,F]
        N = int(y_only_t.size(0))
        if N < WIN:
            raise ValueError(f"Sequence shorter than WAVENET_WIN: N={N}, WIN={WIN}")

        # unfold gives [N-WIN+1, F, WIN] -> permute to [N-WIN+1, WIN, F]
        y_win = y_only_t.unfold(0, WIN, 1).permute(0, 2, 1).contiguous()  # CPU [Nc,WIN,F]
        idx_center = torch.arange(WIN - 1, N, dtype=torch.long)  # CPU [Nc]
        y_in_cpu = y_win  # CPU

        # aligned centers on CPU
        X_true_cent_cpu = torch.from_numpy(X_true_std).to(dtype=torch.float32)[idx_center]  # CPU [Nc,4]
        y_std_cent_cpu = torch.from_numpy(y_std).to(dtype=torch.float32)[idx_center]  # CPU [Nc,1]

    X_cond_full_cpu = torch.from_numpy(X_cond_std).to(dtype=torch.float32)  # [N, D_COND]

    # 1. Conditioning for the CURRENT timestep (Time T)
    X_cond_cent_cpu = X_cond_full_cpu[idx_center]

    X_prev_cond_full_cpu = torch.zeros_like(X_cond_full_cpu)
    X_prev_cond_full_cpu[0] = X_cond_full_cpu[0]
    X_prev_cond_full_cpu[1:] = X_cond_full_cpu[:-1]

    X_prev_cond_cent_cpu = X_prev_cond_full_cpu[idx_center]  # [Nc, D_COND]

    # Clamp only STATE dims in the feed stream (NOT exogenous dims)
    X_prev_cond_cent_feed_cpu = X_prev_cond_cent_cpu.clone()
    if clamp_val is not None:
        X_prev_cond_cent_feed_cpu[:, :D_STATE] = X_prev_cond_cent_feed_cpu[:, :D_STATE].clamp(-clamp_val, clamp_val)

    # Student/state streams are just the first 4 dims, derived from the same source of truth
    X_prev_cent_cpu = X_prev_cond_cent_cpu[:, :D_STATE].contiguous()  # [Nc, 4] (unclamped)
    X_prev_cent_feed_cpu = X_prev_cond_cent_feed_cpu[:, :D_STATE].contiguous()  # [Nc, 4] (clamped feed)

    # --- FIX: x0 MUST be aligned to the first exported center (especially for WaveNet) ---
    x0_first_std = X_prev_cent_cpu[0].clone()  # prev state for first window center
    x0_mean_std = X_prev_cent_cpu.mean(dim=0).clone()  # mean over aligned prev distribution
    x0_cov_std = torch.from_numpy(np.cov(X_prev_cent_cpu.numpy().T)).float()

    # ----------------------------
    # (0) TRUE sequential rollout prior (stateful)
    # ----------------------------
    print(f"[export] starting TRUE sequential rollouts (this is slow). Nc={int(y_in_cpu.size(0))}", flush=True)

    clamp_val = float(payload.get("clamp_x_rollout", 0.0))  # 0 => no clamp
    clamp_val = None if (clamp_val is None or clamp_val <= 0) else clamp_val
    # ----------------------------
    # TF feed stream must match training feed-space (clamped)  [FIX 2]
    # ----------------------------
    X_prev_cent_feed_cpu = _clamp_std(X_prev_cent_cpu, clamp_val)  # CPU [Nc,4]

    use_innov_roll = innov_on_export
    if bool(payload.get("innov_disabled", False)):
        assert use_innov_roll is False, (
            f"[INNOV][EXPORT][ASSERT] payload.innov_disabled=True but use_innov_roll={use_innov_roll}"
        )

    # Hard assertion: if payload says innovation disabled, use_innov_roll must be False
    if bool(payload.get("innov_disabled", False)):
        assert use_innov_roll is False, (
            f"[INNOV][EXPORT][ASSERT] payload.innov_disabled=True but use_innov_roll={use_innov_roll}"
        )

    print(
        f"[export] use_innov_roll={use_innov_roll} | payload.use_innovation_feedback={payload.get('use_innovation_feedback')} | _DISABLE_INNOV_DIAG={_DISABLE_INNOV_DIAG}",
        flush=True)
    kappa_export = float(payload.get("kappa_infer", 0.35))
    print(f"[export] kappa_export={kappa_export}", flush=True)

    mu_roll_mean, b_roll_mean, c_roll_mean, prevfed_roll_mean, prevraw_roll_mean = _student_rollout_single_stream(
        student=student,
        ygx_model=ygx,
        y_in_cpu=y_in_cpu,
        y_std_cpu=y_std_cent_cpu,
        x0_cpu=x0_mean_std,
        X_cond_cent_cpu=X_cond_cent_cpu,
        X_prev_cond_cent_cpu=X_prev_cond_cent_feed_cpu,
        d_state=D_STATE,
        clamp_val=clamp_val,
        use_innovation=use_innov_roll,
        expected_use_innovation=use_innov_roll,
        kappa_corr=kappa_export,
        use_amp=True,
        warmup_steps=32,
        X_true_cpu=X_true_cent_cpu
    )

    mu_roll_true0, b_roll_true0, c_roll_true0, prevfed_roll_true0, prevraw_roll_true0 = _student_rollout_single_stream(
        student=student,
        ygx_model=ygx,
        y_in_cpu=y_in_cpu,
        y_std_cpu=y_std_cent_cpu,
        x0_cpu=x0_mean_std,
        X_cond_cent_cpu=X_cond_cent_cpu,
        X_prev_cond_cent_cpu=X_prev_cond_cent_feed_cpu,
        d_state=D_STATE,
        clamp_val=clamp_val,
        use_innovation=use_innov_roll,
        expected_use_innovation=use_innov_roll,
        kappa_corr=kappa_export,
        use_amp=True,
        warmup_steps=32,
        X_true_cpu=X_true_cent_cpu
    )

    Nc = int(X_true_cent_cpu.size(0))
    windows = [
        ("head_0_500", 0, 500),
        ("head_500_2000", 500, 2000),
        ("mid_10k_12k", 10000, 12000),
        ("mid_20k_22k", 20000, 22000),
        ("tail_last2000", max(0, Nc - 2000), Nc),
    ]

    print("\n[EXPORT][WINDOWED] === Rollout seeded by MEAN (mu_roll_mean / prevfed_roll_mean) ===", flush=True)
    _report_roll_windows(
        tag_prefix="seed_mean",
        mu_roll_cpu=mu_roll_mean,
        prevfed_roll_cpu=prevfed_roll_mean,
        X_true_cent_cpu=X_true_cent_cpu,
        X_prev_cent_feed_cpu=X_prev_cent_feed_cpu,
        windows=windows,
    )

    print("\n[EXPORT][WINDOWED] === Rollout seeded by TRUE first-prev (mu_roll_true0 / prevfed_roll_true0) ===",
          flush=True)
    _report_roll_windows(
        tag_prefix="seed_true0",
        mu_roll_cpu=mu_roll_true0,
        prevfed_roll_cpu=prevfed_roll_true0,
        X_true_cent_cpu=X_true_cent_cpu,
        X_prev_cent_feed_cpu=X_prev_cent_feed_cpu,
        windows=windows,
    )

    _tstats("mu_roll_mean", mu_roll_mean)
    _tstats("X_true_cent_cpu", X_true_cent_cpu)
    _tstats("X_prev_cent_cpu", X_prev_cent_cpu)
    _tstats("X_prev_cent_feed_cpu", X_prev_cent_feed_cpu)

    _prior_metrics("roll_seed_mean vs X_true_center", mu_roll_mean, X_true_cent_cpu)
    _prior_metrics("roll_seed_true0 vs X_true_center", mu_roll_true0, X_true_cent_cpu)

    # ----------------------------
    # (1) y-only prior (batched, GPU in chunks, results returned to CPU)
    # ----------------------------
    out_yonly = _student_forward_batched(
        student,
        y_in_cpu=y_in_cpu,
        x_prev_cpu=None,
        innov_cpu=None,
        batch_size=1024,
        use_amp=True
    )
    z_yonly = out_yonly["z"]
    mu_yonly = out_yonly["mu"]
    logsig_yonly = out_yonly["logsig"]
    b_yonly = out_yonly["b"]
    c_yonly = out_yonly["c"]

    # ----------------------------
    # (2) TF prior + innovation (build X_prev on CPU; compute innovation via ygx in batches)
    # ----------------------------
    X_full_cpu = torch.from_numpy(X_true_std).to(dtype=torch.float32)  # CPU [N,4]
    X_prev_full_cpu = torch.zeros_like(X_full_cpu)
    X_prev_full_cpu[0] = X_full_cpu[0]
    X_prev_cent_cpu = X_prev_full_cpu[idx_center]  # CPU [Nc,4]

    _tstats("prevfed_roll_mean (clamped)", prevfed_roll_mean)
    _tstats("prevraw_roll_mean (unclamped)", prevraw_roll_mean)
    _tstats("prevfed_roll_true0 (clamped)", prevfed_roll_true0)
    _tstats("prevraw_roll_true0 (unclamped)", prevraw_roll_true0)

    _rollin_shift_metrics("prevfed_roll_seed_mean vs X_prev_cent_feed_cpu", prevfed_roll_mean, X_prev_cent_feed_cpu)
    _rollin_shift_metrics("prevfed_roll_seed_true0 vs X_prev_cent_feed_cpu", prevfed_roll_true0, X_prev_cent_feed_cpu)

    _rollin_shift_metrics("prevraw_roll_seed_mean vs X_prev_cent_cpu", prevraw_roll_mean, X_prev_cent_cpu)
    _rollin_shift_metrics("prevraw_roll_seed_true0 vs X_prev_cent_cpu", prevraw_roll_true0, X_prev_cent_cpu)

    # Innovation must use FULL cond vector (ygx's d_x_cond) and output [Nc, 4]
    print(f"[export] Computing 4D TF innovation for {Nc} samples...", flush=True)
    innov_list = []
    chunk_sz = 1024
    for s_idx in range(0, Nc, chunk_sz):
        e_idx = min(Nc, s_idx + chunk_sz)

        y_obs_chunk = y_std_cent_cpu[s_idx:e_idx].to(device)
        x_prev_feed_chunk = X_prev_cent_feed_cpu[s_idx:e_idx].to(device)
        x_cond_chunk = X_prev_cond_cent_feed_cpu[s_idx:e_idx].to(device)

        # Use the export 4D helper you added previously
        innov_4d = _compute_innovation_scalar_export(
            y_obs=y_obs_chunk,
            x_prev_fed=x_prev_feed_chunk,
            x_cond_row=x_cond_chunk,
            ygx_model=ygx,
            cond_gain=cond_gain_for_teacher
        )
        innov_list.append(innov_4d.cpu())

    # Innovation stream for all centres
    innovation_cent_cpu = torch.cat(innov_list, dim=0)

    out_tf = _student_forward_batched(
        student,
        y_in_cpu=y_in_cpu,
        x_prev_cpu=X_prev_cent_feed_cpu,  # clamped TF roll-in
        innov_cpu=innovation_cent_cpu,  # [Nc,1]
        kappa_corr=float(kappa_export),  # keep consistent with rollout
        batch_size=1024,
        use_amp=True
    )

    z_tf = out_tf["z"]
    mu_tf = out_tf["mu"]
    logsig_tf = out_tf["logsig"]
    b_tf = out_tf["b"]
    c_tf = out_tf["c"]

    # ----------------------------
    # Pack + save (everything already CPU)
    # ----------------------------
    # Assemble decoder pack
    pack = {
        # --- Metadata & Config ---
        "folder": folder,
        "feat_names": sc.get("feat_names", feat_names),
        "y_only_feat_names": payload.get("y_only_feat_names", yonly_names),

        # --- Scalers ---
        "x_norm_mu": sc["x_norm_mu"],
        "x_norm_sigma": sc["x_norm_sigma"],
        "x_feat_mu": sc["x_feat_mu"],
        "x_feat_sigma": sc["x_feat_sigma"],
        "y_mu": sc["y_mu"],
        "y_sigma": sc["y_sigma"],
        "y_only_mu": payload["y_only_mu"],
        "y_only_sigma": payload["y_only_sigma"],

        # --- Architecture Re-construction ---
        "student_state": student.state_dict(),  # <--- BEST PRACTICE: Embed weights here
        "student_type": student_type,
        "wavenet_win": payload.get("wavenet_win", None),
        "d_mem": int(payload["d_mem"]),
        "d_x": 4,

        # --- Settings ---
        "cond_gain_for_teacher": float(cond_gain_for_teacher),
        "clamp_x_rollout": float(payload.get("clamp_x_rollout", 0.0)),
        "use_innovation_feedback": bool(payload.get("use_innovation_feedback", True)),

        # --- Data Streams (Aligned) ---
        "idx_center": idx_center,
        "y_in": y_in_cpu,
        "y_std_center": y_std_cent_cpu,
        "X_true_std_center": X_true_cent_cpu,
        "innovation_center": innovation_cent_cpu,

        # --- Encoder Outputs (Batched/Chunked) ---
        "mu_enc_yonly": mu_yonly,
        "logsig_enc_yonly": logsig_yonly,
        "b_enc_yonly": b_yonly,
        "c_enc_yonly": c_yonly,
        "z_enc_yonly": z_yonly,

        "mu_enc_tf": mu_tf,
        "logsig_enc_tf": logsig_tf,
        "b_enc_tf": b_tf,
        "c_enc_tf": c_tf,
        "z_enc_tf": z_tf,

        # --- Rollout Priors (The Gold Standard) ---
        "mu_enc_roll_seed_mean": mu_roll_mean,
        "b_enc_roll_seed_mean": b_roll_mean,
        "c_enc_roll_seed_mean": c_roll_mean,
        "prevfed_enc_roll_seed_mean": prevfed_roll_mean,
        "prevraw_enc_roll_seed_mean": prevraw_roll_mean,

        "mu_enc_roll_seed_true0": mu_roll_true0,
        "b_enc_roll_seed_true0": b_roll_true0,
        "c_enc_roll_seed_true0": c_roll_true0,
        "prevfed_enc_roll_seed_true0": prevfed_roll_true0,
        "prevraw_enc_roll_seed_true0": prevraw_roll_true0,

        # --- Reference ---
        "x0_first_std": x0_first_std,
        "x0_mean_std": x0_mean_std,
        "x0_cov_std": x0_cov_std,

        "X_prev_std_center": X_prev_cent_cpu,
        "X_prev_std_center_feed": X_prev_cent_feed_cpu,

        # --- Architecture Re-construction (MUST match training) ---
        "hidden": 112,
        "layers": 6,
        "g_floor": float(student.g_floor.item()),
        "spectral_norm_ar": bool(payload.get("spectral_norm_ar", _USE_SPECTRAL_NORM_AR)),

        # --- Inference parameters ---
        "kappa_infer": float(payload.get("kappa_infer", 0.35)),
        "innov_disabled": bool(payload.get("innov_disabled", False)),
        "ar_disabled": bool(payload.get("ar_disabled", False)),

        # --- 4-dim state scalers (for raw-unit conversion at inference) ---
        "x_feat_mu_state": list(np.array(sc["x_feat_mu"], dtype=np.float32)[:4].tolist()),
        "x_feat_sigma_state": list(np.array(sc["x_feat_sigma"], dtype=np.float32)[:4].tolist()),

        "ygx_state_dict": ygx.state_dict(),
        "ygx_d_x_cond": D_COND,
        "ygx_cond_gain": float(cond_gain_for_teacher),
    }

    _ensure_dir(out_path)
    torch.save(pack, out_path)
    print(f"[export] Saved decoder pack -> {out_path}")

# ---------------------- Inference ----------------------

# --- Choose raw vs calibrated by R2/MAE/variance ---
def _choose_between_raw_and_cal(y_true, y_pred, y_cal, var_ratio_cal, tol_r2=1e-6):
    r2_raw, mae_raw = _r2_mae_np(y_true, y_pred)
    r2_cal, mae_cal = _r2_mae_np(y_true, y_cal)
    if (r2_cal >= r2_raw - tol_r2) and (mae_cal <= mae_raw + 1e-9) and (var_ratio_cal >= 0.85):
        choice = "calibrated";
        r2, mae = r2_cal, mae_cal
    else:
        choice = "raw";
        r2, mae = r2_raw, mae_raw
    return choice, r2_raw, mae_raw, r2_cal, mae_cal, r2, mae


@torch.no_grad()
# --- Stage 1 inference: flow inverse + residual (Section 2.3) ---
def predict_y_given_x(model, x_feats_std, cond_gain_scale: float = 1.0, map_refine_steps: int = 0):
    z0 = torch.zeros(x_feats_std.size(0), 1, device=x_feats_std.device)
    y_base, _ = model.flow_y_given_x.inverse(z0, cond=x_feats_std * cond_gain_scale, extra=None)
    if map_refine_steps and map_refine_steps > 0:
        y_base = _map_decode_y_base(model.flow_y_given_x, x_feats_std * cond_gain_scale,
                                    steps=map_refine_steps, lr=0.02, z_clip=0.3)
    delta_core, logsig_y, nu_y = model.res_y(x_feats_std)
    delta_y = delta_core + model.delta_skip(x_feats_std)
    return y_base + delta_y


# --- Affine correction on last 10%% ---
def _apply_tail_affine_if_useful(y_true, y_pred, tail_frac=0.10, min_gain=1e-6, var_ratio_floor=0.85):
    n = len(y_pred);
    k = max(1, int(tail_frac * n))
    idx_tail = np.arange(n - k, n)
    y_hat_t = y_pred[idx_tail];
    y_true_t = y_true[idx_tail]
    a, b = _affine_from_xy(y_hat_t, y_true_t)
    y_tail_cal = a * y_hat_t + b
    r2_raw, _ = _r2_mae_np(y_true_t, y_hat_t)
    r2_cal, _ = _r2_mae_np(y_true_t, y_tail_cal)
    var_ratio = float(np.var(y_tail_cal) / (np.var(y_hat_t) + 1e-12))
    if (r2_cal >= r2_raw + min_gain) and (var_ratio >= var_ratio_floor):
        y_out = y_pred.copy();
        y_out[idx_tail] = y_tail_cal;
        used = True
    else:
        y_out = y_pred;
        used = False
    return y_out, used, dict(a=a, b=b, r2_raw=r2_raw, r2_cal=r2_cal, var_ratio=var_ratio)


# --- Affine correction on first 10%% ---
def _apply_head_affine_if_useful(y_true, y_pred, head_frac=0.10, min_gain=1e-6, var_ratio_floor=0.85):
    n = len(y_pred);
    k = max(1, int(head_frac * n))
    idx_head = np.arange(0, k)
    y_hat_h = y_pred[idx_head];
    y_true_h = y_true[idx_head]
    a, b = _affine_from_xy(y_hat_h, y_true_h)
    y_head_cal = a * y_hat_h + b
    r2_raw, _ = _r2_mae_np(y_true_h, y_hat_h)
    r2_cal, _ = _r2_mae_np(y_true_h, y_head_cal)
    var_ratio = float(np.var(y_head_cal) / (np.var(y_hat_h) + 1e-12))
    if (r2_cal >= r2_raw + min_gain) and (var_ratio >= var_ratio_floor):
        y_out = y_pred.copy();
        y_out[idx_head] = y_head_cal;
        used = True
    else:
        y_out = y_pred;
        used = False
    return y_out, used, dict(a=a, b=b, r2_raw=r2_raw, r2_cal=r2_cal, var_ratio=var_ratio)


# ============================================================================
# End-to-end inference (Section 6)
# ============================================================================
def run_inference(folder="./output",
                  ckpt_path="./output/ygx_ckpt.pt",
                  scaler_path="./output/scaler.json",
                  out_csv="./output/ygx_predictions.csv",
                  device=None, head_rows=10,
                  cond_gain_infer=12.0, map_refine_steps: int = 0,
                  guard_var_ratio=0.85,
                  ckpt_paths=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = load_folder_and_join(folder)

    with open(scaler_path, "r") as f:
        sc = json.load(f)
    x_norm_mu = np.array(sc.get("x_norm_mu", [0, 0, 0, 0]), dtype=np.float32)
    x_norm_sd = np.array(sc.get("x_norm_sigma", [1, 1, 1, 1]), dtype=np.float32)

    x_raw_raw, x_feat, feat_names, _ = build_x_features_from_df(
        df, x_norm_mu=x_norm_mu, x_norm_sd=x_norm_sd, fit_slice=None, save_stats_path=None
    )
    y_true = df[["ShortChannel"]].to_numpy(np.float32)

    x_feat_mu, x_feat_sd = sc["x_feat_mu"], sc["x_feat_sigma"]
    y_mu, y_sd = sc["y_mu"], sc["y_sigma"]
    calib = sc.get("calibration", {"type": "affine", "a": 1.0, "b": 0.0})

    x_feat_std = torch.from_numpy(apply_scaling_np(
        x_feat,
        np.array(x_feat_mu, dtype=np.float32)[:x_feat.shape[1]],
        np.array(x_feat_sd, dtype=np.float32)[:x_feat.shape[1]]
    )).to(device)

    d_teacher_infer = len(x_feat_mu) if isinstance(x_feat_mu, list) else len(x_feat_mu.tolist())

    def _load_one(ckpt):
        m = YGivenXModel(d_x_cond=d_teacher_infer, cond_gain_y=cond_gain_infer).to(device)
        _warmup_y_only(m, device)
        state = torch.load(ckpt, map_location=device)
        target_sd = m.state_dict()
        filtered = {k: v for k, v in state.items() if (k in target_sd and v.shape == target_sd[k].shape)}
        m.load_state_dict(filtered, strict=False)
        reset_actnorm_flags(m);
        m.eval()
        return m

    models = []
    if ckpt_paths and isinstance(ckpt_paths, (list, tuple)) and len(ckpt_paths) > 0:
        for p in ckpt_paths: models.append(_load_one(p))
    else:
        models.append(_load_one(ckpt_path))

    x_feat_std_padded = torch.cat(
        [x_feat_std, torch.zeros(x_feat_std.size(0), d_teacher_infer - x_feat_std.size(1), device=device)], dim=1)

    preds_std = []
    for m in models:
        preds_std.append(predict_y_given_x(m, x_feat_std_padded, cond_gain_scale=cond_gain_infer,
                                           map_refine_steps=map_refine_steps).cpu().numpy())
    y_pred_std = np.mean(np.stack(preds_std, axis=0), axis=0)
    y_pred = invert_scaling_np(y_pred_std, y_mu, y_sd)

    y_pred_aff = y_pred.copy()
    if calib.get("type") in {"blend_iso", "isotonic"}:
        a = float(calib.get("a", 1.0)) if "a" in calib else 1.0
        b = float(calib.get("b", 0.0)) if "b" in calib else 0.0
        y_pred_aff = a * y_pred + b

    y_pred_cal_candidate = _apply_calib_blob(y_pred, calib)
    var_ratio_inf = float(np.var(y_pred_cal_candidate) / (np.var(y_pred) + 1e-12))

    choice, r2_raw, mae_raw, r2_calCand, mae_calCand, r2_used, mae_used = _choose_between_raw_and_cal(
        y_true=y_true, y_pred=y_pred, y_cal=y_pred_cal_candidate, var_ratio_cal=var_ratio_inf, tol_r2=1e-6
    )

    if (choice == "calibrated") and (var_ratio_inf >= guard_var_ratio):
        y_pred_final = y_pred_cal_candidate;
        used_calib = calib.get("type", "affine")
    else:
        y_pred_final = y_pred if calib.get("type") not in {"blend_iso", "isotonic",
                                                           "affine"} else y_pred_aff if choice == "calibrated" else y_pred
        used_calib = "raw" if choice == "raw" else (
            "affine_fallback" if calib.get("type") in {"blend_iso", "isotonic"} else "affine")

    y_pred_head_fixed, used_head_fix, head_info = _apply_head_affine_if_useful(
        y_true[:, 0], y_pred_final[:, 0], head_frac=0.10, min_gain=1e-6, var_ratio_floor=guard_var_ratio
    )
    if used_head_fix:
        y_pred_final[:, 0] = y_pred_head_fixed
        used_calib = used_calib + "+head_affine"

    y_pred_tail_fixed, used_tail_fix, tail_info = _apply_tail_affine_if_useful(
        y_true[:, 0], y_pred_final[:, 0], tail_frac=0.10, min_gain=1e-6, var_ratio_floor=guard_var_ratio
    )
    if used_tail_fix:
        y_pred_final[:, 0] = y_pred_tail_fixed
        used_calib = used_calib + "+tail_affine"

    out = df.copy()
    out["y_true_ShortChannel"] = y_true[:, 0]
    out["y_pred_raw_ShortChannel"] = y_pred[:, 0]
    out["y_pred_ShortChannel"] = y_pred_final[:, 0]
    out["y_res_ShortChannel"] = out["y_pred_ShortChannel"] - out["y_true_ShortChannel"]

    _ensure_dir(out_csv);
    out.to_csv(out_csv, index=False)
    print(f"Saved predictions -> {out_csv}")

    slope_raw = _slope(out["y_true_ShortChannel"].to_numpy(), out["y_pred_raw_ShortChannel"].to_numpy())
    slope_cal = _slope(out["y_true_ShortChannel"].to_numpy(), out["y_pred_ShortChannel"].to_numpy())
    var_ratio_report = float(
        np.var(out["y_pred_ShortChannel"].to_numpy()) / (np.var(out["y_pred_raw_ShortChannel"].to_numpy()) + 1e-12))

    print(f"Y (raw) | MAE: {mae_raw:.4f} | R^2: {r2_raw:.4f}")
    if used_calib.startswith("raw"):
        print(f"[inference] calibrated underperformed or failed guardrails -> using RAW outputs.")
    print(
        f"Y ({used_calib}) | MAE: {mae_used:.4f} | R^2: {r2_used:.4f} | var_ratio={var_ratio_report:.3f} | slope_raw={slope_raw:.3f} | slope_cal={slope_cal:.3f}")
    if used_head_fix:
        print(
            f"[head-fix] applied affine on first 10%: a={head_info['a']:.3f}, b={head_info['b']:.3f}, R^2_head_raw={head_info['r2_raw']:.4f}->{head_info['r2_cal']:.4f}, var_ratio_head={head_info['var_ratio']:.3f}")
    if used_tail_fix:
        print(
            f"[tail-fix] applied affine on last 10%: a={tail_info['a']:.3f}, b={tail_info['b']:.3f}, R^2_tail_raw={tail_info['r2_raw']:.4f}->{tail_info['r2_cal']:.4f}, var_ratio_tail={tail_info['var_ratio']:.3f}")

    X_raw = out[["DiastolicBP", "SystolicBP", "CardiacOutput", "StrokeVolume"]].to_numpy(np.float64)
    X_mu = X_raw.mean(axis=0, keepdims=True)
    X_sd = X_raw.std(axis=0, keepdims=True)
    X_sd_safe = np.where(X_sd < 1e-6, 1.0, X_sd)
    X_z = (X_raw - X_mu) / X_sd_safe

    y_true_vec = out["y_true_ShortChannel"].to_numpy(np.float64)
    y_pred_vec = out["y_pred_ShortChannel"].to_numpy(np.float64)

    def _ridge_beta(X, y, lam=1e-6):
        y_c = y - y.mean()
        XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
        XtY = X.T @ y_c
        beta = np.linalg.solve(XtX, XtY)
        return beta

    # Ridge sensitivity: how each CV dim influences optical signal
    beta_true_z = _ridge_beta(X_z, y_true_vec)
    beta_model_z = _ridge_beta(X_z, y_pred_vec)

    beta_true_raw = beta_true_z / X_sd_safe.squeeze()
    beta_model_raw = beta_model_z / X_sd_safe.squeeze()

    print("\n[probe] ShortChannel ~ z-scored [DBP, SBP, CO, SV] (GROUND TRUTH fit):")
    print(
        f"        DBP={beta_true_z[0]:.3f}, SBP={beta_true_z[1]:.3f}, CO={beta_true_z[2]:.3f}, SV={beta_true_z[3]:.3f}")
    print("[probe] ShortChannel ~ z-scored [DBP, SBP, CO, SV] (MODEL fit):")
    print(
        f"        DBP={beta_model_z[0]:.3f}, SBP={beta_model_z[1]:.3f}, CO={beta_model_z[2]:.3f}, SV={beta_model_z[3]:.3f}")
    print("[probe] Approx raw-unit weights (GROUND TRUTH):")
    print(
        f"        DBP={beta_true_raw[0]:.3f}, SBP={beta_true_raw[1]:.3f}, CO={beta_true_raw[2]:.3f}, SV={beta_true_raw[3]:.3f}")
    print("[probe] Approx raw-unit weights (MODEL):")
    print(
        f"        DBP={beta_model_raw[0]:.3f}, SBP={beta_model_raw[1]:.3f}, CO={beta_model_raw[2]:.3f}, SV={beta_model_raw[3]:.3f}\n")

    if head_rows > 0:
        k = max(1, int(0.10 * len(out)))
        r2_head_after, mae_head_after = _r2_mae_np(out["y_true_ShortChannel"].values[:k],
                                                   out["y_pred_ShortChannel"].values[:k])
        r2_tail_after, mae_tail_after = _r2_mae_np(out["y_true_ShortChannel"].values[-k:],
                                                   out["y_pred_ShortChannel"].values[-k:])
        print(f"[head metrics] first10% AFTER fixes: MAE={mae_head_after:.4f} R^2={r2_head_after:.4f}")
        print(f"[tail metrics] last10% AFTER fixes:  MAE={mae_tail_after:.4f} R^2={r2_tail_after:.4f}")

        print("\n=== Y (ShortChannel): true vs predicted (HEAD) ===")
        print(
            out[["Time", "y_true_ShortChannel", "y_pred_ShortChannel", "y_res_ShortChannel"]].head(head_rows).to_string(
                index=False))
        print("\n=== Y (ShortChannel): true vs predicted (TAIL) ===")
        print(
            out[["Time", "y_true_ShortChannel", "y_pred_ShortChannel", "y_res_ShortChannel"]].tail(head_rows).to_string(
                index=False))


# ---------------------- Main ----------------------

if __name__ == "__main__":
    FOLDER = "./output"
    CKPT = "./output/ygx_ckpt.pt"
    SCALER = "./output/scaler.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler = train_y_given_x(
        folder=FOLDER,
        ckpt_out=CKPT,
        scaler_out=SCALER,
        epochs=18,
        lr_flow=6e-5,
        lr_res=2e-5,
        batch=128,
        val_frac=0.2,
        device=device,
        max_grad_norm=2.0,
        lambda_sens_start=4e-2,
        lambda_sens_end=4e-2,
        lambda_sigma_prior=2e-3,
        lambda_sigma_cov=2e-3,
        lambda_l1=0,
        gate_floor_start=0.58,
        gate_floor_peak=0.58,
        gate_floor_tail=0.58,
        cond_gain_start=20.0,
        cond_gain_end=20.0,
        es_patience=8,
        use_isotonic=True,
        iso_alpha=0.25,
        iso_knots=64,
        guard_var_ratio=0.98,
        guard_bins=10
    )

    TRAIN_Y_MEMORY_ENCODER = True
    if TRAIN_Y_MEMORY_ENCODER:
        print("\n[mem] ===== Starting geometry-aware Y->X encoder training (with GAN) =====")
        df_full = load_folder_and_join(FOLDER)
        _ = train_y_memory_encoder(
            model_yx=model,
            df=df_full,
            scaler_obj=scaler,
            epochs=5,
            lr=1e-3,
            batch=320,
            device=device,
            save_path="./output/y_memory_encoder.pt",
            USE_WAVENET_STUDENT=True,
            WAVENET_WIN=128,
            eval_every=5,
            w_geom=0.0,
            w_teacher=1.0,
            w_gan=0.10,
            gan_lr=5e-4,
            recon_epochs=0,
            freeze_sigma_epochs=5,
            fixed_logsig=-2.0,
            w_mu=30.0,
            w_delta=5.0,
            clamp_x_rollout=5.0,
            stage1_min_tf_r2=0.30,
            hist_p_start=0.0,
            hist_p_end=0.7,
            hist_warmup_epochs=1,
            w_yonly_start=5,
            w_yonly_end=15.0,
            yonly_ramp_epochs=1,
            ss_p_start=0.3,
            ss_p_end=1,
            ss_warmup_epochs=0,
            w_roll=50.0,
            roll_gamma=1.0,
            w_shift=0.05,
            w_obs_y=15.0,
            w_jac=0.5,
            jac_samples=16,
            w_bound=2.0,
            bound_soft=4.5,
            bound_power=2.0,
            corr_kappa_max=0.35,
            corr_ramp_epochs=6,
            yonly_pretrain_epochs=7,
            detach_hist_pred=True,
        )

    export_decoder_pack(
        folder=FOLDER,
        ygx_ckpt_path=CKPT,
        scaler_path=SCALER,
        y_memory_encoder_path="./output/y_memory_encoder.pt",
        out_path="./output/decoder_pack.pt",
        device=device,
        cond_gain_for_teacher=20.0,
    )

    run_inference(
        folder=FOLDER,
        ckpt_path=CKPT,
        scaler_path=SCALER,
        out_csv="./output/ygx_predictions.csv",
        device=device,
        cond_gain_infer=20.0,
        map_refine_steps=0,
        guard_var_ratio=0.85,
        ckpt_paths=None
    )
