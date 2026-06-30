# ============================================================================
# core_flow.py — Normalizing Flow Components for fNIRS Pipeline
# ============================================================================
#
# CONTENTS:
#   Math helpers: EMA, causal lag, Kalman filter, scaling
#   Safe autograd: gradient computation with NaN protection
#   Masks: alternating binary masks for coupling layers
#   ActNorm1d: data-dependent activation normalisation
#   Inv1x1Conv: learnable permutation with spectral norm cap
#   FiLM: feature-wise linear modulation
#   Conditioner: shared MLP mapping conditioning -> hidden h
#   RQS: rational quadratic spline (flexible monotonic transform)
#   RQSCouplingHead: parameterises spline knots from conditioner
#   CondRQSBlock: one coupling layer (ActNorm + 1x1Conv + RQS)
#   ConditionalFlow: K stacked blocks = the full flow P(y|x)
#   ResidualHeadStudentT: heavy-tailed correction on flow output
#   student_t_nll: Student-t negative log-likelihood loss
#
# FLOW ARCHITECTURE (Section 2.3):
#   K=8 CondRQSBlocks, each: ActNorm -> 1x1Conv -> split -> RQS -> merge
#   Gated residual: y = x + s*(RQS(x)-x), s in [gate_floor, 1]
#   Change-of-variables: log p(y) = log p_Z(f(y)) + log|det df/dy|
# ============================================================================
# core_flow.py
# Stable core: math helpers, flow blocks/coupling, conditioners, residual heads, and safe grad utilities.
# These are your original definitions, moved here verbatim so you don't need to touch them when
# you iterate on choosers/struct losses/training in experiment_train.py.

import os, json, math, time, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== Small constants & simple helpers ===============
# --- Numerical stability constant ---
SMALL = 1e-6

# --- Causal EMA: y[t] = alpha*x[t] + (1-alpha)*y[t-1] ---
# Used in build_y_only_features for multi-scale temporal smoothing
def ema_causal(x, alpha=0.2):
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    # Recursive EMA: each output depends only on current input + previous output
    y[0] = x[0]
    for i in range(1, len(x)): y[i] = alpha * x[i] + (1.0 - alpha) * y[i-1]
    return y

# --- Causal lag: shift x forward by k steps, repeat x[0] to fill gap ---
# Used in build_y_only_features (dy, ddy) and _make_hard_weights_from_y
def lag_np_causal(x, k):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        import traceback
        traceback.print_stack()
        raise ValueError(f"[lag_np_causal] received empty array! shape={x.shape}, k={k}")
    if k <= 0: return x.copy()
    # Prepend k copies of x[0], drop last k elements -> causal shift
    return np.r_[np.repeat(x[0], k), x[:-k]].astype(np.float32)

# ================= Kalman filter helper (verbatim) =================
# --- Local-trend Kalman filter with adaptive process noise ---
# State = [position, velocity]. Transition: x[t] = F*x[t-1] + process_noise
# Returns: smoothed signal, innovations, innovation std, Kalman gain.
# Used in build_y_only_features for Y observation feature bank.
def kf_local_trend(z, dt=1.0, q=1e-3, r=None, adapt=True, adapt_beta=0.10, eps=1e-9):
    z = np.asarray(z, dtype=np.float64)
    T = z.shape[0]
    if r is None:
        mad = np.median(np.abs(np.diff(z))) if T > 2 else 0.0
        r = max((1.4826 * mad)**2, 1e-6)
    xhat = np.zeros(T); v = np.zeros(T); S = np.zeros(T); K1 = np.zeros(T)
    s = np.array([z[0], 0.0]); P = np.diag([1.0, 1.0]) * 1.0
    # F = state transition: position += velocity*dt, velocity persists
    # G = process noise input matrix: acceleration drives both state dims
    # H = observation: we only observe position (not velocity)
    Fm = np.array([[1.0, dt],[0.0, 1.0]]); G = np.array([[0.5*dt*dt],[dt]]); H = np.array([[1.0, 0.0]])
    S_ew = None
    for t in range(T):
        s = Fm @ s; Q = (q * (G @ G.T)); P = Fm @ P @ Fm.T + Q
        # Innovation: y = observation - predicted observation
        # S = innovation variance: H*P*H' + R
        # K = Kalman gain: P*H' / S (how much to trust observation)
        y = z[t] - (H @ s)[0]; S_t = (H @ P @ H.T)[0, 0] + r; K = (P @ H.T)[:, 0] / max(S_t, eps)
        # State update: correct prediction by gain * innovation
        # Covariance update: P -= K * H * P (reduce uncertainty)
        s = s + K * y; P = P - np.outer(K, (H @ P)[0])
        xhat[t] = s[0]; v[t] = y; S[t] = S_t; K1[t] = K[0]
        # Adaptive Q: track exponentially-weighted innovation variance
        # If innovations are larger than expected, increase process noise q
        if adapt:
            S_ew = S_t if S_ew is None else (1-adapt_beta)*S_ew + adapt_beta*S_t
            ratio = S_ew / max(r, eps)
            q = np.clip(q * (0.9 + 0.1 * ratio**0.25), 1e-8, 1e2)
    return {
        "xhat": xhat.astype(np.float32),
        "innov": v.astype(np.float32),
        "innov_std": np.sqrt(np.maximum(S, eps)).astype(np.float32),
        "gain": K1.astype(np.float32),
        "R": float(r),
        "q_final": float(q),
    }

# ================= Scaling helpers (verbatim) =================
# --- Compute (mean, std) from training data for standardisation ---
# std floored at 1e-12 to prevent division by zero
def standardize_train_stats(arr):
    mu = arr.mean(axis=0, keepdims=True).astype(np.float32)
    sd = arr.std(axis=0, keepdims=True).astype(np.float32) + 1e-12
    return mu, sd

# --- Standardise: (x - mu) / sigma ---
def apply_scaling_np(arr, mu, sigma):
    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(1, -1)
    return ((arr - mu) / (sigma + 1e-12)).astype(np.float32)

# --- Invert standardisation: x * sigma + mu ---
def invert_scaling_np(arr, mu, sigma):
    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(1, -1)
    return (arr * (sigma + 1e-12) + mu).astype(np.float32)

# ================= Safe autograd/backward (verbatim) =================
# --- Zero tensors matching shapes (fallback for failed grads) ---
def _zeros_like_list(xs):
    if isinstance(xs, (list, tuple)):
        return tuple(torch.zeros_like(x, requires_grad=False) for x in xs)
    return torch.zeros_like(xs, requires_grad=False)

# --- torch.autograd.grad with NaN/error protection ---
# Returns zero gradients on failure instead of crashing.
# Used in Stage 1 Jacobian sensitivity computation (Section 2.3).
def safe_autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=False,
    create_graph=False,
    allow_unused=True,
):
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    # Fast exit: if no input requires grad, return zeros immediately
    if not any(getattr(p, "requires_grad", False) for p in inputs):
        return _zeros_like_list(inputs)
    if isinstance(outputs, (list, tuple)):
        outputs = sum([o if isinstance(o, torch.Tensor) else torch.as_tensor(o, dtype=torch.float32, device=inputs[0].device) for o in outputs])
    try:
        grads = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
        # Replace None grads (from allow_unused) with zeros
        grads = tuple(torch.zeros_like(p) if (g is None) else g for g, p in zip(grads, inputs))
        return grads
    except Exception as e:
        return _zeros_like_list(inputs)


# ================= Mask helpers (verbatim) =================
# --- Create K alternating binary masks for coupling layers ---
# Even blocks: first half=1 (kept), second=0 (transformed)
# Odd blocks: reversed. d=1: all zeros (everything transformed).
def make_alternating_masks(d, K):
    if d == 1:
        masks = [torch.zeros(1, dtype=torch.float32).clone() for _ in range(K)]
    else:
        a = d // 2; b = d - a
        m1 = torch.cat([torch.ones(a), torch.zeros(b)])
        m2 = 1.0 - m1
        masks = [m1.clone() if k % 2 == 0 else m2.clone() for k in range(K)]
    return nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in masks])

# --- Split tensor: xa (mask=1, kept unchanged), xb (mask=0, transformed by RQS) ---
def split_by_mask(x, mask):
    keep_idx = (mask == 1).nonzero(as_tuple=False).squeeze(-1)
    trans_idx = (mask == 0).nonzero(as_tuple=False).squeeze(-1)
    # index_select gathers dims where mask=1 into xa, mask=0 into xb
    xa = x.index_select(-1, keep_idx) if keep_idx.numel() > 0 else x.new_zeros(x.size(0), 0)
    xb = x.index_select(-1, trans_idx) if trans_idx.numel() > 0 else x.new_zeros(x.size(0), 0)
    return xa, xb, keep_idx, trans_idx

# --- Reassemble: scatter xa and xb back to their original positions ---
def merge_by_mask(xa, xb, d, keep_idx, trans_idx):
    B = xa.size(0)
    out = torch.empty(B, d, device=xa.device, dtype=xa.dtype)
    # scatter_ writes xa into positions where mask was 1, xb where mask was 0
    if keep_idx.numel() > 0: out.scatter_(dim=-1, index=keep_idx.view(1, -1).expand(B, -1), src=xa)
    if trans_idx.numel() > 0: out.scatter_(dim=-1, index=trans_idx.view(1, -1).expand(B, -1), src=xb)
    return out

# ============== NEW: tiny context to collect taps per pass ==============
# --- Collects intermediate activations during a flow pass ---
# Each coupling block appends its internal features (cond_h, head_feat, etc.)
# Used by extract_conditioner_taps for teacher embedding (whitener).
class _TapCtx:
    def __init__(self):
        self.extras = []  # list of dicts; each dict holds tensors to tap for one block

# ================= Core layers/blocks (verbatim + tap plumbing) =================
# ============================================================================
# ActNorm1d — Activation Normalisation (Section 2.3)
# ============================================================================
# Data-dependent init: first forward batch sets loc and scale.
# Forward: z = x * exp(log_scale) + loc, log_det = sum(log_scale)
# Inverse: x = (z - loc) * exp(-log_scale)
# log_scale bounded by tanh to prevent explosion.
class ActNorm1d(nn.Module):
    def __init__(self, d, eps=1e-6, log_scale_bound=np.log(5.0)):
        super().__init__()
        # loc: learned translation (init zeros, set by data_init)
        # log_scale: learned scaling in log-space (bounded by tanh)
        self.loc = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        # initialized: False until first batch triggers data_init
        self.initialized = False
        self.eps = eps
        self.log_scale_bound = float(log_scale_bound)
    @torch.no_grad()
    def data_init(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        small = std < 1e-4
        # loc = -mean (center at zero), scale = 1/std clamped [0.2, 5.0]
        # Small-std dims get identity (no scaling) to prevent amplifying noise
        loc = -mean; scale = (1.0 / std).clamp(0.2, 5.0)
        self.loc.copy_(torch.where(small, torch.zeros_like(loc), loc))
        self.log_scale.copy_(torch.log(torch.where(small, torch.ones_like(scale), scale)))
        self.initialized = True
    # --- Effective log-scale: tanh bounds prevent divergence during training ---
    # tanh(x/B)*B: smoothly saturates at +/-B, prevents log_scale explosion
    def _eff_log_scale(self): return self.log_scale_bound * torch.tanh(self.log_scale / self.log_scale_bound)
    def forward(self, x, logdet=None):
    # --- ActNorm forward: z = x * exp(ls) + loc, accumulate log_det ---
        if not self.initialized: self.data_init(x)
        # Apply: multiply by scale (exp of bounded log_scale), then shift by loc
        # Log-det contribution: sum of log_scale (same for every sample)
        ls = self._eff_log_scale(); z = x.mul(ls.exp()) + self.loc
        ld = ls.sum()
        return z, (ld.expand(x.size(0)) if logdet is None else logdet + ld)
    def inverse(self, z, logdet=None):
        ls = self._eff_log_scale()
        # ActNorm inverse: x = (z - loc) * exp(-log_scale)
        x = (z - self.loc).mul(torch.exp(-ls)); ld = ls.sum()
        return x, (-ld.expand(z.size(0)) if logdet is None else logdet - ld)

# ============================================================================
# Inv1x1Conv — Invertible 1x1 Convolution (Section 2.3)
# ============================================================================
# Learnable permutation matrix, QR-initialised (orthogonal start).
# Forward: z = x @ W^T, log_det = log|det(W)|
# Inverse: x = solve(W^T, z^T)^T
# Spectral norm capped to prevent ill-conditioning.
class Inv1x1Conv(nn.Module):
    def __init__(self, d, cap=10.0):
        super().__init__()
        # QR init: start with a random orthogonal matrix (det = +/-1)
        W = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32))[0]
        self.W = nn.Parameter(W)
        self.cap = float(cap)
    # --- Spectral norm cap via 3-step power iteration ---
    def _eff_W(self):
        with torch.no_grad():
            u = torch.randn(self.W.shape[0], 1, device=self.W.device, dtype=self.W.dtype)
            # Power iteration: estimate largest singular value sigma
            # Then scale W so sigma <= cap (prevents ill-conditioning)
            # Newton iteration: solve f(s) = yk + hk*num(s)/den(s) - y = 0
            # Update: s <- s - f(s)/f'(s), clamped to [eps, 1-eps]
            for _ in range(3):
                v = F.normalize(self.W.T @ u, dim=0, eps=1e-8)
                u = F.normalize(self.W @ v, dim=0, eps=1e-8)
            sigma = (u.T @ self.W @ v).abs()
            # If sigma > cap, scale W down. If sigma <= cap, keep W as-is.
            scale = torch.minimum(torch.tensor(1.0, device=sigma.device, dtype=sigma.dtype),
                                  torch.tensor(self.cap, device=sigma.device, dtype=sigma.dtype) / (sigma + 1e-12))
        return self.W * scale
    def forward(self, x, logdet=None):
        W = self._eff_W()
        # 1x1Conv forward: z = x @ W^T (learnable linear transform)
        # log_det = log|det(W)| — same for every sample in batch
        # Matrix multiply: each sample x[i] is linearly transformed by W
        # This is a learnable permutation + scaling of dimensions
        z = x @ W.t()
        _, ldet = torch.slogdet(W)
        return z, (ldet.expand(x.size(0)) if logdet is None else logdet + ldet)
    def inverse(self, z, logdet=None):
        W = self._eff_W()
        # 1x1Conv inverse: solve(W^T, z^T)^T — more stable than W^{-1}
        # Solve W^T * x^T = z^T for x — more stable than inverting W explicitly
        x = torch.linalg.solve(W.T, z.T).T
        _, ldet = torch.slogdet(W)
        return x, (-ldet.expand(z.size(0)) if logdet is None else logdet - ldet)

# ============================================================================
# FiLM — Feature-wise Linear Modulation
# ============================================================================
# Modulates hidden features using conditioner output h:
#   output = feat * (1 + gamma(h)) + beta(h)
# Zero-initialised: starts as identity (no modulation), learns to modulate.
class FiLM(nn.Module):
    def __init__(self, d_h, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(d_h, hidden_dim, dtype=torch.float32)
        self.beta  = nn.Linear(d_h, hidden_dim, dtype=torch.float32)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta .weight); nn.init.zeros_(self.beta .bias)
    # Affine transform: scale by (1+gamma) and shift by beta, both functions of h
    def forward(self, h, feat): return feat * (1 + self.gamma(h)) + self.beta(h)

# ============================================================================
# Conditioner — Maps conditioning vector to hidden representation h
# ============================================================================
# MLP: d_cond -> 256 -> 256 -> d_h with SiLU activations + LayerNorm.
# Learnable gain parameter scales output magnitude.
# h is computed ONCE per flow pass and shared across all K coupling blocks.
# This is the "brain" that tells each RQS block how to transform.
class Conditioner(nn.Module):
    def __init__(self, d_in, d_rel_extra=0, d_h=160, gain_init=1.5):
        super().__init__()
        d_total = d_in + d_rel_extra
        self.net = nn.Sequential(
            nn.Linear(d_total, 256, dtype=torch.float32), nn.SiLU(),
            nn.Linear(256, 256, dtype=torch.float32),    nn.SiLU(),
            nn.Linear(256, d_h, dtype=torch.float32),    nn.LayerNorm(d_h),
        )
        # Learnable gain: scales conditioner output. Starts at gain_init (~1.5).
        # Allows the network to control the magnitude of conditioning signal.
        self.gain = nn.Parameter(torch.tensor(float(gain_init), dtype=torch.float32))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)): m.float()
    # Forward: MLP(conditioning) * gain -> h [B, d_h]
    def forward(self, c): return self.net(c) * self.gain

# ============================================================================
# Rational Quadratic Spline (RQS) Transform (Section 2.3)
# ============================================================================
# The CORE bijective transform inside each coupling layer.
# Defines a monotonic piecewise-rational function with K knots.
#
# Inside [-bound, bound]: K-piece rational quadratic interpolation
#   y = yk + hk * (a*s^2 + dk0*s*(1-s)) / (1 + c*s*(1-s))
#   where s = (x - xk) / wk is the local coordinate within bin k
#   Analytic Jacobian: dy/dx computable in closed form
#
# Outside [-bound, bound]: linear tails with slope = boundary derivative
#   Ensures the transform is defined everywhere, not just inside bounds.
#
# Parameters:
#   widths [B, d, K]: softmax-normalised bin widths (sum to 1)
#   heights [B, d, K]: softmax-normalised bin heights (sum to 1)
#   derivatives [B, d, K+1]: positive slopes at knot boundaries
#
# Returns: (y, log_det_sum, log_det_per_dim)
def rqs_with_tails(x, widths, heights, derivatives, bound=4.0, inverse=False):
    eps = 1e-6
    # Normalise widths and heights to sum to 1 (valid probability simplex)
    # Clamp derivatives to [0.05, 6.0] for numerical stability
    widths  = (widths.clamp_min(1e-3)); widths  = widths  / (widths .sum(-1, keepdim=True) + eps)
    heights = (heights.clamp_min(1e-3)); heights = heights / (heights.sum(-1, keepdim=True) + eps)
    derivatives = derivatives.clamp_min(5e-2).clamp_max(6.0)
    # Cumulative sums -> knot positions in [-bound, bound]
    # xk[i] = -bound + 2*bound*cumsum(widths)[i] (bin boundaries on x-axis)
    # yk[i] = -bound + 2*bound*cumsum(heights)[i] (bin boundaries on y-axis)
    cumw = torch.cumsum(widths, dim=-1);  cumh = torch.cumsum(heights, dim=-1)
    cumw = F.pad(cumw, (1, 0), value=0.0); cumh = F.pad(cumh, (1, 0), value=0.0)
    xk = 2.0 * bound * cumw - bound; yk = 2.0 * bound * cumh - bound; dk = derivatives
    # === RQS FORWARD: x -> y ===
    # Three regions: left tail (linear), inner (spline), right tail (linear)
    if not inverse:
        left = x < -bound; right = x > bound; inner = ~(left | right)
        y = torch.empty_like(x); lad = torch.zeros_like(x)
        # LEFT TAIL: y = y0 + d0*(x - x0) — linear with slope d0 (first derivative)
        if left.any():
            x0 = -bound; y0 = -bound; d0 = dk[..., 0]
            yl = y0 + d0 * (x - x0); y[left] = yl[left]; lad[left] = torch.log(d0[left].clamp_min(1e-6))
        # RIGHT TAIL: y = yK + dK*(x - xK) — linear with slope dK (last derivative)
        if right.any():
            xK = bound; yK = bound; dK = dk[..., -1]
            yr = yK + dK * (x - xK); y[right] = yr[right]; lad[right] = torch.log(dK[right].clamp_min(1e-6))
        # INNER: find which bin x falls in, compute RQS interpolation
        if inner.any():
            xi = x.clamp(-bound + 1e-5, bound - 1e-5)
            # Bin index: which knot interval does each x[i] belong to?
            # Binary search via cumulative comparison with knot positions
            idx = torch.sum(xi.unsqueeze(-1) >= xk, dim=-1) - 1; idx = idx.clamp(0, widths.size(-1) - 1)
            def gath(arr):
                if arr.size(-1) == widths.size(-1) + 1:
                    left_g = arr[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                    right_g = arr[..., 1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1); return left_g, right_g
                return arr.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            xk0, xk1 = gath(xk); yk0, yk1 = gath(yk)
            wk = (xk1 - xk0).clamp_min(1e-4); hk = (yk1 - yk0).clamp_min(1e-4)
            dk0 = dk[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1); dk1 = dk[..., 1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            # a = height/width ratio of this bin
            # s = normalised position within bin [0,1]
            # c = curvature parameter from boundary derivatives
            a = hk / wk; s = ((xi - xk0) / wk).clamp(1e-5, 1 - 1e-5); c = dk0 + dk1 - 2.0 * a
            # RQS formula: y = yk + hk * num/den
            # num = quadratic in s, den = quadratic in s
            # This is the rational quadratic — monotonic by construction
            num = a * s ** 2 + dk0 * s * (1 - s); den = 1 + c * s * (1 - s)
            yi = yk0 + hk * (num / den).clamp(-20.0, 20.0)
            den_safe = den ** 2 + 1e-6
            # Analytic Jacobian: dy/ds = hk * (dnum*den - num*dden) / den^2
            # Then dy/dx = dy/ds / wk (chain rule with bin width)
            dnum = 2 * a * s + dk0 * (1 - 2 * s) + (dk1 - dk0) * s; dden = c * (1 - 2 * s)
            dyds = hk * ((dnum * den - num * dden) / den_safe); dydx = (dyds / wk).clamp_min(1e-4)
            y[inner] = yi[inner]; lad[inner] = torch.log(dydx[inner])
        return y, lad.sum(dim=-1), lad
    # === RQS INVERSE: y -> x via Newton iteration ===
    # Same three regions. Inner uses 3-step Newton to invert the spline.
    y = x; left = y < -bound; right = y > bound; inner = ~(left | right)
    x_out = torch.empty_like(y); lad = torch.zeros_like(y)
    if left.any():
        x0 = -bound; y0 = -bound; d0 = dk[..., 0].clamp_min(1e-6)
        xl = x0 + (y - y0) / d0; x_out[left] = xl[left]; lad[left] = -torch.log(d0[left])
    if right.any():
        xK = bound; yK = bound; dK = dk[..., -1].clamp_min(1e-6)
        xr = xK + (y - yK) / dK; x_out[right] = xr[right]; lad[right] = -torch.log(dK[right])
    if inner.any():
        yi = y.clamp(-bound + 1e-5, bound - 1e-5)
        idx = torch.sum(yi.unsqueeze(-1) >= yk, dim=-1) - 1; idx = idx.clamp(0, widths.size(-1) - 1)
        def gath(arr):
            if arr.size(-1) == widths.size(-1) + 1:
                left_g = arr[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                right_g = arr[..., 1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1); return left_g, right_g
            return arr.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        xk0, xk1 = gath(xk); yk0, yk1 = gath(yk)
        wk = (xk1 - xk0).clamp_min(1e-4); hk = (yk1 - yk0).clamp_min(1e-4)
        dk0 = dk[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1); dk1 = dk[..., 1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        a = hk / wk; s = ((yi - yk0) / hk).clamp(1e-5, 1 - 1e-5); c = dk0 + dk1 - 2.0 * a
        for _ in range(3):
            num = a * s ** 2 + dk0 * s * (1 - s); den = 1 + c * s * (1 - s)
            f = yk0 + hk * (num / den) - yi; den_safe = den ** 2 + 1e-6
            dnum = 2 * a * s + dk0 * (1 - 2 * s) + (dk1 - dk0) * s; dden = c * (1 - 2 * s)
            df = hk * ((dnum * den - num * dden) / den_safe); s = (s - f / (df + 1e-6)).clamp(1e-5, 1 - 1e-5)
        xi = xk0 + wk * s
        num = a * s ** 2 + dk0 * s * (1 - s); den = 1 + c * s * (1 - s)
        den_safe = den ** 2 + 1e-6
        dnum = 2 * a * s + dk0 * (1 - 2 * s) + (dk1 - dk0) * s; dden = c * (1 - 2 * s)
        dyds = hk * ((dnum * den - num * dden) / den_safe); dydx = (dyds / wk).clamp_min(1e-4)
        x_out[inner] = xi[inner]; lad[inner] = -torch.log(dydx[inner])
    return x_out, lad.sum(dim=-1), lad

# ============================================================================
# RQSCouplingHead — Parameterises spline from conditioner output
# ============================================================================
# Input: xa (kept dims) + projected conditioning -> hidden
# FiLM modulation by conditioner hidden h -> post-LN -> dropout -> MLP
# Output: widths logits [B,d_out,K], heights logits, slopes logits
# slope_bias_target=2.0: derivatives initialised near 2.0 (close to identity).
class RQSCouplingHead(nn.Module):
    def __init__(self, d_in, d_cond_raw, d_out, K=8, d_h=160, hidden_dim=320, proj_dim=48, slope_bias_target=2.0, dropout_p=0.10):
        super().__init__()
        self.K = K
        # Project raw conditioning to compact representation before concat with xa
        self.proj_cond = nn.Linear(d_cond_raw, proj_dim, dtype=torch.float32)
        # Merge kept dimensions xa + projected conditioning -> hidden_dim
        self.hidden_in = nn.Linear(d_in + proj_dim, hidden_dim, dtype=torch.float32)
        self.hidden = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.SiLU(),
        )
        # FiLM: modulate hidden features using conditioner output h
        # This is how the shared conditioner h influences each block's RQS params
        self.film = FiLM(d_h, hidden_dim)
        self.post_film_ln = nn.LayerNorm(hidden_dim)
        self.post_film_dropout = nn.Dropout(dropout_p)
        # Three output heads: widths [d_out*K], heights [d_out*K], slopes [d_out*(K+1)]
        # These become the RQS knot parameters after softmax/softplus
        self.out_widths  = nn.Linear(hidden_dim, d_out * K, dtype=torch.float32)
        self.out_heights = nn.Linear(hidden_dim, d_out * K, dtype=torch.float32)
        self.out_slopes  = nn.Linear(hidden_dim, d_out * (K + 1), dtype=torch.float32)
        nn.init.kaiming_uniform_(self.out_widths.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_heights.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_slopes.weight, a=math.sqrt(5))
        nn.init.normal_(self.out_widths.bias, mean=0.0, std=0.02)
        nn.init.normal_(self.out_heights.bias, mean=0.0, std=0.02)
        # Slope bias init: softplus^{-1}(target) so initial derivatives ~ target
        # target=2.0 -> slopes start near 2.0 -> transform starts close to identity
        slope_bias = math.log(math.exp(slope_bias_target) - 1.0)
        with torch.no_grad():
            self.out_slopes.bias.fill_(slope_bias)
            self.out_slopes.bias.add_(0.02 * torch.randn_like(self.out_slopes.bias))
    # --- Head forward: xa + cond -> FiLM(h) -> MLP -> (widths, heights, slopes) ---
    def forward(self, xa, h, cond_raw, return_feats: bool = False):
        pr = self.proj_cond(cond_raw)
        # Concatenate kept dims xa with projected conditioning, map to hidden
        feat = self.hidden_in(torch.cat([xa, pr], dim=-1))
        feat = self.film(h, feat)
        feat = self.post_film_ln(feat)
        self.post_film_dropout.train(self.training)
        feat = self.post_film_dropout(feat)
        feat = self.hidden(feat)
        wL = self.out_widths(feat); hL = self.out_heights(feat); sL = self.out_slopes(feat)
        if return_feats:
            return (wL, hL, sL), {
                "head_feat": feat,       # post hidden, pre-out
                "head_w_pre": wL,        # pre-softmax widths logits
                "head_h_pre": hL,        # pre-softmax heights logits
                "head_s_pre": sL,        # pre-softplus slopes logits
            }
        return wL, hL, sL

# ============================================================================
# CondRQSBlock — One Complete Coupling Layer (Section 2.3)
# ============================================================================
# This is where the actual flow transformation happens.
# Forward path:
#   1. ActNorm: data-dependent normalisation
#   2. Inv1x1Conv: learnable permutation
#   3. Split by mask: xa (kept unchanged), xb (to be transformed)
#   4. RQSCouplingHead(xa, h, cond) -> widths, heights, slopes
#   5. RQS transform on xb using those parameters
#   6. Gated residual: yb = xb + s*(RQS(xb) - xb)
#   7. Merge xa + yb back to full vector
# Each step contributes to the log-det-Jacobian.
class CondRQSBlock(nn.Module):
    def __init__(self, d, d_cond_raw, d_h, K=8, bound=6.0, gate_init=0.9, gate_floor=0.6, temp=1.0, slope_bias_target=2.0):
        super().__init__()
        # Step 1+2: ActNorm normalises, 1x1Conv permutes dimensions
        self.norm = ActNorm1d(d); self.mix = Inv1x1Conv(d, cap=10.0)
        self.d = d; self.d_h = d_h; self.K = K; self.bound = bound
        self.heads = nn.ModuleDict()
        # Gate parameter: controls residual strength. Initialised via logit(gate_init).
        # s = gate_floor + (1-gate_floor)*sigmoid(gate_param)
        self.gate_param = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32))
        self.d_cond_raw = d_cond_raw; self.gate_floor = float(gate_floor); self.temp = float(temp); self.slope_bias_target = float(slope_bias_target)
    # --- Lazy head creation: build RQSCouplingHead on first use for given split sizes ---
    def _get_head(self, d_xa, d_xb):
        key = f"{d_xa}->{d_xb}"
        if key not in self.heads:
            self.heads[key] = RQSCouplingHead(d_in=d_xa, d_cond_raw=self.d_cond_raw, d_out=d_xb, K=self.K, d_h=self.d_h, slope_bias_target=self.slope_bias_target, dropout_p=0.10)
        return self.heads[key]
    # --- Coupling forward: ActNorm -> 1x1Conv -> split -> RQS -> gate -> merge ---
    def forward(self, x, h, cond_raw, mask, logdet=None, collect_feats: bool = False, tap_ctx: _TapCtx = None):
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype) if logdet is None else logdet
        # Steps 1-2: normalise then permute, accumulating log-det
        z, ld = self.norm(x, None); logdet = logdet + ld
        z, ld = self.mix(z, None); logdet = logdet + ld
        if self.d == 1:
            B = z.size(0); xa = z.new_zeros(B, 0); xb = z; keep_idx = torch.empty(0, dtype=torch.long, device=z.device); trans_idx = torch.arange(self.d, dtype=torch.long, device=z.device)
        else:
        # Step 3: split — xa stays unchanged, xb gets transformed by RQS
            xa, xb, keep_idx, trans_idx = split_by_mask(z, mask)
        jac_abs_mean = torch.zeros(1, device=x.device, dtype=x.dtype)
        if xb.numel() == 0:
            return z, logdet, jac_abs_mean
        head = self._get_head(xa.size(-1), xb.size(-1))
        B = x.size(0); d_b = xb.size(-1); K = self.K

        if collect_feats:
            (wL, hL, sL), feats = head(xa, h, cond_raw, return_feats=True)
        else:
            wL, hL, sL = head(xa, h, cond_raw, return_feats=False)

        # Convert logits to valid RQS parameters:
        #   widths/heights: softmax -> positive, sum to 1
        #   slopes: softplus -> positive, clamped to [~0, 6.0]
        widths  = (wL.view(B, d_b, K) / self.temp).softmax(dim=-1)
        heights = (hL.view(B, d_b, K) / self.temp).softmax(dim=-1)
        slopes  = F.softplus(sL.view(B, d_b, K + 1)) + 1e-3
        widths = widths.clamp_min(1e-3); widths = widths / (widths.sum(-1, keepdim=True) + 1e-6)
        heights = heights.clamp_min(1e-3); heights = heights / (heights.sum(-1, keepdim=True) + 1e-6)
        slopes = slopes.clamp_max(6.0)
        # Step 5: apply RQS transform to xb — the actual nonlinear bijection
        # lad = per-element log|dy/dx| from the spline
        yb_rqs, _, lad = rqs_with_tails(xb, widths, heights, slopes, bound=self.bound, inverse=False)
        # Step 6a: compute gate s in [gate_floor, 1.0]
        # s close to 0 -> nearly identity, s close to 1 -> full RQS
        s_raw = torch.sigmoid(self.gate_param)
        s = self.gate_floor + (1.0 - self.gate_floor) * s_raw
        # Step 6b: Jacobian of gated residual
        # d/dx[x + s*(f(x)-x)] = (1-s) + s*f'(x)
        # log of this is the log-det contribution
        lad_res = torch.log(((1 - s) + s * torch.exp(lad)).clamp_min(1e-6))
        # Step 6c: gated output = interpolate identity <-> RQS
        # Sum per-dim log-det to get per-sample log-det
        yb = xb + s * (yb_rqs - xb); ld_res = lad_res.sum(dim=-1)
        jac_abs_mean = lad.abs().mean()
        if torch.isnan(yb).any() or torch.isinf(yb).any(): raise RuntimeError("NaN/Inf in RQS forward output.")
        # Step 7: merge xa + yb back into full vector, add log-det
        z_out = merge_by_mask(xa, yb, self.d, keep_idx, trans_idx); logdet = logdet + ld_res

        if collect_feats and (tap_ctx is not None):
            tap_ctx.extras.append({
                "cond_h": h,                # conditioner hidden
                "cond_raw": cond_raw,       # raw conditioner input
                "head_feat": feats.get("head_feat", None) if isinstance(feats, dict) else None,
                "head_w_pre": feats.get("head_w_pre", None) if isinstance(feats, dict) else None,
                "head_h_pre": feats.get("head_h_pre", None) if isinstance(feats, dict) else None,
                "head_s_pre": feats.get("head_s_pre", None) if isinstance(feats, dict) else None,
            })
        return z_out, logdet, jac_abs_mean
    # --- Coupling inverse: split -> inverse RQS -> un-permute -> un-normalise ---
    def inverse(self, z, h, cond_raw, mask, logdet=None, collect_feats: bool = False, tap_ctx: _TapCtx = None):
        logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype) if logdet is None else logdet
        if self.d == 1:
            B = z.size(0); xa = z.new_zeros(B, 0); xb = z; keep_idx = torch.empty(0, dtype=torch.long, device=z.device); trans_idx = torch.arange(self.d, dtype=torch.long, device=z.device)
        else:
            xa, xb, keep_idx, trans_idx = split_by_mask(z, mask)
        jac_abs_mean = torch.zeros(1, device=z.device, dtype=z.dtype)
        if xb.numel() > 0:
            head = self._get_head(xa.size(-1), xb.size(-1))
            B = z.size(0); d_b = xb.size(-1); K = self.K

            if collect_feats:
                (wL, hL, sL), feats = head(xa, h, cond_raw, return_feats=True)
            else:
                wL, hL, sL = head(xa, h, cond_raw, return_feats=False)

            widths  = (wL.view(B, d_b, K) / self.temp).softmax(dim=-1)
            heights = (hL.view(B, d_b, K) / self.temp).softmax(dim=-1)
            slopes  = F.softplus(sL.view(B, d_b, K + 1)) + 1e-3
            widths = widths.clamp_min(1e-3); widths = widths / (widths.sum(-1, keepdim=True) + 1e-6)
            heights = heights.clamp_min(1e-3); heights = heights / (heights.sum(-1, keepdim=True) + 1e-6)
            slopes = slopes.clamp_max(6.0)
            # Inverse RQS: given yb, find xb such that RQS(xb) = yb
            # Uses Newton iteration inside rqs_with_tails
            xb0, _, lad_inv_dim = rqs_with_tails(xb, widths, heights, slopes, bound=self.bound, inverse=True)
            y_target = xb
            # Re-evaluate forward RQS at inverse solution to compute gated inverse
            rqs_y, _, lad_fwd_dim = rqs_with_tails(xb0, widths, heights, slopes, bound=self.bound, inverse=False)
            s_raw = torch.sigmoid(self.gate_param); s = self.gate_floor + (1.0 - self.gate_floor) * s_raw
            # Newton for gated residual inverse:
            # Solve g(x) = (1-s)*x + s*RQS(x) = y for x
            # One step: x <- x - g(x)/g'(x)
            # g'(x) = (1-s) + s*RQS'(x)
            f = (1 - s) * xb0 + s * rqs_y - y_target
            lad_fwd_dim_clamped = lad_fwd_dim.clamp(-20.0, 20.0)
            safe_res = ((1 - s) + s * torch.exp(lad_fwd_dim_clamped)).clamp_min(1e-6)
            # Newton update: subtract residual / derivative
            # safe_res = (1-s) + s*exp(log|RQS'|) clamped for stability
            xb_in = xb0 - f / safe_res
            lad_res = -torch.log(safe_res); ld_res = lad_res.sum(dim=-1)
            jac_abs_mean = lad_inv_dim.abs().mean()
            if torch.isnan(xb_in).any() or torch.isinf(xb_in).any(): raise RuntimeError("NaN/Inf in residual inverse output.")
            x_merge = merge_by_mask(xa, xb_in, self.d, keep_idx, trans_idx); logdet = logdet + ld_res
        else:
            x_merge = merge_by_mask(xa, xb, self.d, keep_idx, trans_idx)
        # Undo steps 2,1: inverse permutation then inverse normalisation
        x, ld = self.mix.inverse(x_merge, None); logdet = logdet + ld
        x, ld = self.norm.inverse(x, None);  logdet = logdet + ld

        if collect_feats and (tap_ctx is not None):
            tap_ctx.extras.append({
                "cond_h": h,
                "cond_raw": cond_raw,
                "head_feat": feats.get("head_feat", None) if (xb.numel() > 0 and isinstance(feats, dict)) else None,
                "head_w_pre": feats.get("head_w_pre", None) if (xb.numel() > 0 and isinstance(feats, dict)) else None,
                "head_h_pre": feats.get("head_h_pre", None) if (xb.numel() > 0 and isinstance(feats, dict)) else None,
                "head_s_pre": feats.get("head_s_pre", None) if (xb.numel() > 0 and isinstance(feats, dict)) else None,
            })
        return x, logdet, jac_abs_mean

# ============================================================================
# ConditionalFlow — Full Normalizing Flow P(y|x) (Section 2.3)
# ============================================================================
# K_blocks CondRQSBlocks stacked sequentially with alternating masks.
# Single shared conditioner maps x -> h, then h conditions all blocks.
#
# Forward (training): data y -> latent z, accumulate log|det J|
#   Used to compute log p(y|x) = log p_Z(z) + log|det J|
#
# Inverse (inference): latent z -> data y
#   Set z=0 (mode of standard normal) -> y_base = most likely y given x
#
# Also provides extract_conditioner_taps for teacher embedding.
class ConditionalFlow(nn.Module):
    def __init__(self, d_var, d_cond, d_h=128, K_blocks=4, K_bins=8, bound=6.0, temp=1.0, slope_bias_target=2.0, cond_gain_init=1.5, d_rel_extra=0):
        super().__init__()
        self.d_var = d_var
        # Shared conditioner: computed ONCE, shared across all K blocks
        self.cond_net = Conditioner(d_cond, d_rel_extra=d_rel_extra, d_h=d_h, gain_init=cond_gain_init)
        self.blocks = nn.ModuleList([
            CondRQSBlock(d_var, d_cond + d_rel_extra, d_h, K=K_bins, bound=bound, temp=temp, slope_bias_target=slope_bias_target)
            for _ in range(K_blocks)
        ])
        # Alternating masks: even blocks transform one half, odd blocks the other
        self.masks = make_alternating_masks(d_var, K_blocks)
        self.d_cond = d_cond
        self.d_rel_extra = d_rel_extra
        self.d_h = d_h
        self._fix_d1_masks_if_needed()
        self._prebuild_heads()
        # NEW: cache for last concatenated taps
        # Cache for extracted taps (teacher embedding features)
        self._last_taps = None
    @torch.no_grad()
    def _fix_d1_masks_if_needed(self):
        if self.d_var == 1:
            bad = any([(m.data.abs().sum().item() > 0) for m in self.masks])
            if bad:
                for m in self.masks: m.data.zero_()
    def _prebuild_heads(self):
        with torch.no_grad():
            for blk, m in zip(self.blocks, self.masks):
                d_xa = int((m.data == 1).sum().item()); d_xb = int(self.d_var - d_xa)
                if d_xb > 0: _ = blk._get_head(d_xa, d_xb)
    # --- Forward: y -> z through all blocks, accumulate total log|det J| ---
    def forward(self, x, cond, extra=None, collect_stats=False, collect_feats: bool = False):
        cond_in = torch.cat([cond, extra], dim=-1) if extra is not None else cond
        # Compute conditioner hidden h ONCE, then pass to each block
        # z starts as x (data), gets transformed by each block sequentially
        h = self.cond_net(cond_in); z = x
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        jac_means = []
        tap_ctx = _TapCtx() if collect_feats else None
        # record conditioner hidden/cond_in as a "block 0" tap
        if collect_feats and tap_ctx is not None:
            tap_ctx.extras.append({"cond_h": h, "cond_raw": cond_in})
        # Sequential pass: z flows through block 1 -> 2 -> ... -> K
        # Each block adds its log-det contribution
        for blk, m in zip(self.blocks, self.masks):
            z, logdet, jmean = blk(z, h, cond_in, m, logdet, collect_feats=collect_feats, tap_ctx=tap_ctx); jac_means.append(jmean.item())
        if collect_feats:
            self._last_taps = self._concat_taps(tap_ctx.extras)
        return (z, logdet, jac_means) if collect_stats else (z, logdet)
    # --- Inverse: z -> y through blocks in REVERSE order ---
    def inverse(self, z, cond, extra=None, collect_stats=False, collect_feats: bool = False):
        cond_in = torch.cat([cond, extra], dim=-1) if extra is not None else cond
        h = self.cond_net(cond_in); x = z
        logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        jac_means = []
        tap_ctx = _TapCtx() if collect_feats else None
        if collect_feats and tap_ctx is not None:
            tap_ctx.extras.append({"cond_h": h, "cond_raw": cond_in})
        # Blocks applied in REVERSE order (inverse of forward composition)
        for blk, m in zip(reversed(self.blocks), reversed(self.masks)):
            x, logdet, jmean = blk.inverse(x, h, cond_in, m, logdet, collect_feats=collect_feats, tap_ctx=tap_ctx); jac_means.append(jmean.item())
        jac_means = list(reversed(jac_means))
        if collect_feats:
            self._last_taps = self._concat_taps(tap_ctx.extras)
        return (x, logdet, jac_means) if collect_stats else (x, logdet)

    # Expose conditioner h and processed cond_in for external use
    def conditioner_hidden(self, cond, extra=None):
        cond_in = torch.cat([cond, extra], dim=-1) if extra is not None else cond
        h = self.cond_net(cond_in)
        return h, cond_in

    @staticmethod
    # Flatten all intermediate activations into one [B, D_raw] tensor
    # Each block contributes cond_h, head_feat, head_w/h/s_pre
    def _concat_taps(extras_list):
        # Flatten each dict’s tensors to [B, *] and concat along feature dim
        feat_chunks = []
        for d in extras_list:
            for k, v in d.items():
                if v is None:
                    continue
                feat_chunks.append(v.flatten(1))
        if not feat_chunks:
            return None
        return torch.cat(feat_chunks, dim=1)  # [B, D_raw]

    @torch.no_grad()
    # --- Extract internal activations as teacher embedding ---
    # Runs inverse pass with collect_feats=True
    # Returns concatenated [B, D_raw] tensor of all block internals
    # Used by fit_teacher_whitener in exp_base_static
    def extract_conditioner_taps(self, cond, cond_gain_scale=1.0, z_perturb: float = 0.0):
        """
        Run inverse(z, cond, collect_feats=True) and return concatenated taps [N, D_raw].
        Supports averaging across cond_gain_scale values and small z jitters for stability.
        """
        # Multi-scale: average taps across different gain values for stability
        if isinstance(cond_gain_scale, (list, tuple)):
            taps = []
            for g in cond_gain_scale:
                taps.append(self.extract_conditioner_taps(cond, g, z_perturb))
            return torch.stack(taps, dim=0).mean(dim=0)

        B = cond.size(0)
        if z_perturb and z_perturb > 0:
            z = torch.empty(B, self.d_var, device=cond.device).normal_(0.0, z_perturb)
        else:
            z = torch.zeros(B, self.d_var, device=cond.device)
        _ = self.inverse(z, cond=cond * float(cond_gain_scale), extra=None, collect_feats=True)
        return self._last_taps  # [B, D_raw] or None

    @torch.no_grad()
    # --- Per-dim importance: std of taps across batch (Fisher-ish proxy) ---
    # Dims with higher variance across samples carry more information
    def tap_importance(self, cond, cond_gain_scale=12.0, eps: float = 0.0):
        """
        Optional: Fisher-ish importance proxy for taps. Here we use a simple, stable proxy:
        the per-dimension std of taps across the batch, normalized to mean 1.0.
        """
        T = self.extract_conditioner_taps(cond, cond_gain_scale=cond_gain_scale, z_perturb=0.0)
        if T is None:
            return None
        w = T.std(dim=0).clamp_min(1e-6)
        w = w / (w.mean() + 1e-8)
        return w


# ============================================================================
# ResidualHeadStudentT — Heavy-tailed residual correction (Section 2.3)
# ============================================================================
# Operates ON TOP of the flow base prediction y_base.
# Outputs: delta (mean correction), log_sigma (scale), nu (df).
# Final prediction: y = y_base + delta
# Loss computed with Student-t NLL using (y_obs - y_base, delta, log_sigma, nu).
#
# nu (degrees of freedom) in [nu_min=2, nu_max=30] via sigmoid.
# Small nu -> heavy tails (robust to outliers).
# delta zero-init: flow prediction is the initial output.
class ResidualHeadStudentT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=192, dropout_p=0.1, nu_min=2.0, nu_max=30.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # mean_corr: additive correction delta. Zero-init -> start at flow prediction.
        # log_sigma: observation noise scale. Init bias=-1 -> sigma starts small.
        # nu_raw: degrees of freedom logits -> sigmoid -> [nu_min, nu_max].
        self.mean_corr = nn.Linear(hidden, out_dim)
        self.log_sigma = nn.Linear(hidden, out_dim)
        self.nu_raw    = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.mean_corr.weight); nn.init.zeros_(self.mean_corr.bias)
        nn.init.constant_(self.log_sigma.bias, -1.0)
        self.nu_min = float(nu_min); self.nu_max = float(nu_max)
    # --- Forward: x_feat -> (delta, log_sigma, nu) ---
    def forward(self, z):
        h = self.net(z)
        delta = self.mean_corr(h)
        log_sig = self.log_sigma(h).clamp(-4.0, 2.0)
        # nu via sigmoid: maps to [nu_min=2, nu_max=30]
        # Low nu (~2): very heavy tails. High nu (~30): nearly Gaussian.
        nu = self.nu_min + (self.nu_max - self.nu_min) * torch.sigmoid(self.nu_raw(h))
        return delta, log_sig, nu

# ================= Loss helpers (verbatim) =================

# --- Student-t NLL: main loss for Stage 1 teacher training (Section 2.3) ---
# NLL = 0.5*log(nu*pi) + log_sigma + ((nu+1)/2)*log(1 + z^2/nu)
# where z = (target - mean) / exp(log_sigma)
# Heavier tails than Gaussian -> more robust to outlier observations.
# As nu -> inf, approaches Gaussian NLL.
def student_t_nll(target, mean, log_sigma, nu):
    e = (target - mean)
    # Standardised residual z = (target - mean) / sigma
    # NLL per sample: 0.5*log(nu*pi) + log(sigma) + ((nu+1)/2)*log(1 + z^2/nu)
    z = e * torch.exp(-log_sigma)
    nll = 0.5 * (torch.log(nu) + math.log(math.pi)) + log_sigma + ((nu + 1.0) / 2.0) * torch.log1p((z ** 2) / nu)
    return nll.mean()

# ================= Warmups & misc (verbatim) =================

@torch.no_grad()
# --- Reset all ActNorm init flags -> re-triggers data_init on next forward ---
# Call after loading weights to ensure ActNorm re-initialises from new data.
def reset_actnorm_flags(model: nn.Module):
    # Walk all submodules, set initialized=False for every ActNorm1d
    for m in model.modules():
        if isinstance(m, ActNorm1d): m.initialized = False


# --- Count trainable parameters ---
def _count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
