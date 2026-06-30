# prior_student.py
# Shared "prior student" module for BOTH encoder training and decoder inference.
# Architecture and forward logic MUST match exp_base_dynamic.py EXACTLY.
#
# This file is the SINGLE SOURCE OF TRUTH for inference.
# Any architecture change in training must be reflected here.

# ===========================================================================
# prior_student.py — SINGLE SOURCE OF TRUTH for inference (Section 2.4-2.7)
# ===========================================================================
# This file defines the WaveNet student encoder architecture and rollout logic
# used at inference time. Every class and function MUST match the training code
# in exp_base_dynamic.py exactly. Any architecture change in training must be
# reflected here, or state_dict loading will silently drop weights.
#
# Architecture overview:
#   Stage 2a (WaveNet backbone, Section 2.4):
#     Input Conv1d(d_yfeat->112) -> 6 causal dilated blocks -> Conv1d(112->512) -> z
#     5 zero-init heads: dmu_y, log_sigma, bias estimate, bias gate, state gate
#     AR sub-net: 2-layer (4->64->8), spectral norm, dropout p=0.4
#   Stage 2b (Y-only backbone, Section 2.5):
#     Separate MLP (d_yfeat->256->128->64) + temporal Conv1d (d=[1,4,16])
#     No shared parameters with WaveNet
#   Drift correction (Section 2.6):
#     Gated EMA bias integrator + Kalman-inspired innovation feedback
#   State update (Section 2.7):
#     x_prop = x_hat(t-1) + dmu_y + dmu_AR + b(t) + dx_innov
#     x_prop = (1-kappa)*x_prop + kappa*x_hat_yonly
#     x_hat(t) = x_hat(t-1) + g*(x_prop - x_hat(t-1))
#
# Key inference functions:
#   build_student_from_payload() — reconstruct student from saved checkpoint
#   student_forward_consistent() — single-step forward matching training
#   rollout_from_y() — full closed-loop rollout over observation sequence
# ===========================================================================
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Global knobs — MUST match exp_base_dynamic.py at module level
# ===========================================================================

# --- Global knobs: MUST match exp_base_dynamic.py at module level ---
# These control the student's output scaling, gating initialisation,
# and diagnostic flags. Mismatches cause silent prediction errors.
_DELTA_SCALE = 2.0
_LOGSIG_SCALE = 0.5
_GATE_BIAS_INIT = 1.5
_USE_SPECTRAL_NORM_AR = True          # Training uses True => spectral_norm on AR

# --- Diagnostic flags: control which correction pathways are active ---
# When False (default), both AR and innovation pathways are enabled.
# These can be toggled via set_diag_flags() or from saved payload.
_DISABLE_AR_DIAG = False              # Training: False (AR is ENABLED)
_DISABLE_INNOV_DIAG = False           # Training: False (Innovation is ENABLED)

# --- Gate override: when True, gate g is forced to 1.0 (full update) ---
# _SAT_THRESH: pre-tanh magnitude threshold for saturation diagnostics
_FORCE_GATE_OPEN = False
_SAT_THRESH = 2.0


# --- Toggle diagnostic flags from caller (e.g. from saved payload) ---
# Called by build_student_from_payload() to match training configuration.
def set_diag_flags(*, disable_ar: bool | None = None, disable_innov: bool | None = None) -> None:
    """Allow toggling diag flags from caller (e.g. from payload)."""
    global _DISABLE_AR_DIAG, _DISABLE_INNOV_DIAG
    if disable_ar is not None:
        _DISABLE_AR_DIAG = bool(disable_ar)
    if disable_innov is not None:
        _DISABLE_INNOV_DIAG = bool(disable_innov)


# --- Set spectral norm flag BEFORE constructing student ---
# Spectral norm changes weight key names in state_dict.
# Must be called before YMemoryWaveNet() or load will silently fail.
def set_spectral_norm_flag(enabled: bool) -> None:
    """Must be called BEFORE building student if payload flag differs."""
    global _USE_SPECTRAL_NORM_AR
    _USE_SPECTRAL_NORM_AR = bool(enabled)


# --- Single source of truth for whether innovation feedback is active ---
# Returns True only if BOTH conditions met:
#   1. use_innovation_feedback=True (training config)
#   2. innov_disabled=False (diagnostic override)
def _innov_enabled(use_innovation_feedback: bool, innov_disabled: bool) -> bool:
    return bool(use_innovation_feedback) and (not bool(innov_disabled))


# --- Spectral normalisation wrapper for AR sub-network (Section 2.4) ---
# Constrains the largest singular value of the AR weight matrix.
# Applied to both layers of the 2-layer AR feedforward.
def _sn(linear: nn.Linear) -> nn.Module:
    return nn.utils.parametrizations.spectral_norm(linear) if _USE_SPECTRAL_NORM_AR else linear


# --- Percentile helper for diagnostic logging ---
def _pct(t: torch.Tensor, q: float) -> float:
    """q in [0,100]. Returns percentile of flattened tensor."""
    if t is None:
        return float("nan")
    tt = t.detach()
    if tt.numel() == 0:
        return float("nan")
    tt = tt.float().reshape(-1)
    qq = max(0.0, min(100.0, float(q))) / 100.0
    return float(torch.quantile(tt, qq).item())


# ===========================================================================
# Bias update — gated EMA (matches training exactly)
# ===========================================================================

# ===========================================================================
# Gated EMA bias integrator (Section 2.6, Stage E Section 3.5)
# ===========================================================================
# Tracks and corrects systematic offset (drift) in state estimates.
# b(t) = (1 - g_b) * b(t-1) + g_b * b_hat
# where g_b = g_min + (g_max - g_min) * sigmoid(logits), typically ~0.09
# Slow integration rate smooths per-step noise while tracking persistent offset.
# Output clamped to [-b_max, b_max] to prevent unbounded accumulation.
#
# INFERENCE NOTE: g_min/g_max/b_max here (0.05/0.20/0.3) match training.
# Mismatched bounds will cause drift divergence at inference.
def gated_ema_bias_update(b_prev, b_hat, g_logits, *, g_min=0.05, g_max=0.20, b_max=4.0):
    g01 = torch.sigmoid(g_logits.float())
    g = g_min + (g_max - g_min) * g01
    g = g.to(dtype=b_prev.dtype)
    b_next = (1.0 - g) * b_prev + g * b_hat
    b_next = torch.clamp(b_next, -b_max, b_max)
    return b_next, g


# ===========================================================================
# Kappa sanitizer (matches training exactly)
# ===========================================================================

# ===========================================================================
# Kappa shape normaliser (Section 2.7)
# ===========================================================================
# Ensures kappa is a consistent shape for the observation-AR mixing step:
#   x_prop = (1 - kappa) * x_prop + kappa * x_hat_yonly
# Accepts: None (=0), scalar, (D,), (1,D), (B,D), or (N,D) collapsed to (D,)
# Falls back to zeros (no mixing) for None/empty inputs.
def _ensure_kappa_vec_or_batch(
    k: torch.Tensor | float | None,
    *,
    x_prop: torch.Tensor,
    D: int,
) -> torch.Tensor:
    device = x_prop.device
    dtype = x_prop.dtype
    B = x_prop.size(0)

    if k is None:
        return torch.zeros(D, device=device, dtype=dtype)
    if not torch.is_tensor(k):
        return torch.full((D,), float(k), device=device, dtype=dtype)

    k = k.to(device=device, dtype=dtype)
    if k.numel() == 0:
        return torch.zeros(D, device=device, dtype=dtype)

    if k.dim() == 0:
        return k.expand(D)
    if k.dim() == 1:
        if k.numel() == D:
            return k
        if k.numel() == 1:
            return k.expand(D)
        raise RuntimeError(f"[kappa] expected (D,) or scalar; got {tuple(k.shape)} (D={D})")
    if k.dim() == 2:
        if k.shape == (B, D):
            return k
        if k.shape[0] == 1 and k.shape[1] == D:
            return k
        if k.shape[1] == D and k.shape[0] > 1:
            return k.mean(dim=0)
        raise RuntimeError(f"[kappa] bad 2D shape {tuple(k.shape)}; expected (B,D), (1,D), or (N,D) with D={D}")
    raise RuntimeError(f"[kappa] bad rank {k.dim()} for shape {tuple(k.shape)}")


# ===========================================================================
# WaveNetBlock — with lag conditioning (matches training WaveNetBlock)
# ===========================================================================

# ===========================================================================
# WaveNetBlock — causal dilated residual block (Stage 2a, Section 2.4)
# ===========================================================================
# Implements tanh(f) * sigmoid(g) gating with strictly causal left-padding.
# Kernel size 3 with dilation d gives receptive field increment of 2*d.
# 6 blocks with d=[1,2,4,8,16,32] give ~42s receptive field at 6 Hz.
# Residual connection scaled by 1/sqrt(2) for gradient stability.
#
# Optional conditioning via cond_proj: projects lag features [B,C_cond,1]
# into 2*ch channels that additively modulate f_in and g_in.
# When cond_channels=0, conditioning is disabled (no lag conditioning).
class WaveNetBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, cond_channels: int = 0):
        super().__init__()
        self.dilation = int(dilation)
        self.conv_f = nn.Conv1d(ch, ch, kernel_size=3, dilation=self.dilation, padding=0)
        self.conv_g = nn.Conv1d(ch, ch, kernel_size=3, dilation=self.dilation, padding=0)
        self.proj = nn.Conv1d(ch, ch, kernel_size=1)

        self.cond_proj = None
        if cond_channels and int(cond_channels) > 0:
            self.cond_proj = nn.Conv1d(int(cond_channels), 2 * ch, kernel_size=1)

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad_left = 2 * self.dilation
        x = F.pad(x, (pad_left, 0))
        return conv(x)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        f_in = self._causal_conv(self.conv_f, x)
        g_in = self._causal_conv(self.conv_g, x)

        if (self.cond_proj is not None) and (cond is not None):
            cg = self.cond_proj(cond)
            cf, cg2 = cg.chunk(2, dim=1)
            f_in = f_in + cf
            g_in = g_in + cg2

        f = torch.tanh(f_in)
        g = torch.sigmoid(g_in)
        h = f * g
        h = self.proj(h)
        return (x + h) * (1.0 / math.sqrt(2.0))


# ===========================================================================
# LagConditioner (matches training exactly)
# ===========================================================================

# ===========================================================================
# LagConditioner — projects lagged state estimates into conditioning signal
# ===========================================================================
# Converts (B, num_lags, d_feat) lag values + optional mask into a
# (B, cond_channels, 1) conditioning vector for WaveNet blocks.
# Uses Linear -> SiLU -> LayerNorm projection.
#
# At inference, lag_values are filled with the student's own past
# predictions (mu_roll[t - lag_dist]) rather than ground truth.
class LagConditioner(nn.Module):
    def __init__(self, d_feat: int, num_lags: int, cond_channels: int, include_mask: bool = True):
        super().__init__()
        self.d_feat = int(d_feat)
        self.num_lags = int(num_lags)
        self.include_mask = bool(include_mask)

        in_dim = self.num_lags * (self.d_feat + (1 if self.include_mask else 0))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, int(cond_channels)),
            nn.SiLU(),
            nn.LayerNorm(int(cond_channels)),
        )

    def forward(self, lag_values: torch.Tensor, lag_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, K, F = lag_values.shape
        if K != self.num_lags or F != self.d_feat:
            raise ValueError(f"Expected lag_values (B,{self.num_lags},{self.d_feat}), got {tuple(lag_values.shape)}")

        if self.include_mask:
            if lag_mask is None:
                lag_mask = lag_values.new_ones((B, K))
            m = lag_mask.unsqueeze(-1)
            x = torch.cat([lag_values, m], dim=-1)
        else:
            x = lag_values

        x = x.reshape(B, -1)
        c = self.proj(x)
        return c.unsqueeze(-1)


# ===========================================================================
# YMemoryMLP (matches training exactly — gated EMA bias, per-dim kappa, etc.)
# ===========================================================================

# ===========================================================================
# YMemoryMLP — MLP student fallback (used when student_type != 'wavenet')
# ===========================================================================
# Simpler alternative to YMemoryWaveNet: processes a single-frame y_feat
# through a 6-layer MLP backbone (d_yfeat -> 1536 -> ... -> d_mem).
# Same output interface as YMemoryWaveNet:
#   (z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, x_prop, g)
#
# Contains all the same correction mechanisms:
#   - Gated EMA bias integrator (Section 2.6)
#   - Innovation feedback via innov_embed + innov_direct (Section 2.6)
#   - AR sub-network with spectral norm (Section 2.4)
#   - Kappa mixing with y-only proposal (Section 2.7)
#   - Gate with floor (Section 5.8)
#
# NOTE: Does NOT use temporal concatenation in the y-only path.
# The y-only backbone here is a separate 3-layer MLP (256->256->128).
class YMemoryMLP(nn.Module):
    def __init__(self, d_yfeat, d_mem=2040, width=1536, depth=6, drop_p=0.10, d_x=4):
        super().__init__()
        layers = []
        in_dim = d_yfeat
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(drop_p)]
            in_dim = width
        self.backbone = nn.Sequential(*layers)
        self.proj_out = nn.Linear(width, d_mem)

                # --- Stage 2b Y-only backbone (Section 2.5) ---
                # Fully separate from WaveNet — no shared parameters.
                # MLP processes last-frame features, temporal Conv1d processes full window.
                # Combined output feeds head_dx_y_separate for initial state estimate.
        self.yonly_backbone = nn.Sequential(
            nn.Linear(d_yfeat, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.head_dx_y_separate = nn.Linear(128, d_x)
        self.register_buffer('g_floor', torch.tensor(0.65))

        self.delta1 = nn.Linear(d_mem, d_mem)
        self.delta4 = nn.Linear(d_mem, d_mem)
        nn.init.zeros_(self.delta1.weight); nn.init.zeros_(self.delta1.bias)
        nn.init.zeros_(self.delta4.weight); nn.init.zeros_(self.delta4.bias)

        self.head_corr = nn.Linear(d_mem, d_x)
        self.head_corr_gate = nn.Linear(d_mem, d_x)
        nn.init.normal_(self.head_corr_gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_corr_gate.bias, -0.224)
        nn.init.zeros_(self.head_corr.weight); nn.init.zeros_(self.head_corr.bias)

                # --- Output heads from WaveNet memory vector z (all zero-init) ---
                # head_dx_y: observation increment dmu_y (pre-tanh, scaled by _DELTA_SCALE)
                # head_logsig: log-sigma for state uncertainty (pre-tanh, scaled by _LOGSIG_SCALE)
                # head_corr: raw bias estimate b_hat (gradient-scaled to 10%)
                # head_corr_gate: EMA gate logits for bias update rate
                # gate: state update gate logits (init bias=1.5 -> sigmoid ~0.82)
        self.head_dx_y = nn.Linear(d_mem, d_x)
        self.head_logsig = nn.Linear(d_mem, d_x)
        nn.init.zeros_(self.head_dx_y.weight); nn.init.zeros_(self.head_dx_y.bias)
        nn.init.zeros_(self.head_logsig.weight); nn.init.zeros_(self.head_logsig.bias)

                # --- Innovation feedback layers (Section 2.6) ---
                # innov_embed: scalar innovation -> d_mem additive modulation on z
                # innov_direct: (8-step innovation buffer + x_prev) -> dx_innov correction
                # Both zero-init to start as identity (no correction initially)
        self.innov_embed = nn.Linear(1, d_mem)
        nn.init.zeros_(self.innov_embed.weight)
        nn.init.zeros_(self.innov_embed.bias)
        self.innov_direct = nn.Sequential(
            nn.Linear(8 + d_x, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, d_x),
        )
        for m in self.innov_direct.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight);
                nn.init.zeros_(m.bias)

        self.ar = nn.Sequential(
            _sn(nn.Linear(d_x, 64)), nn.GELU(), nn.LayerNorm(64),
            _sn(nn.Linear(64, 2 * d_x))
        )
        for m in self.ar.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        self.gate = nn.Linear(d_mem, d_x)
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate.bias, float(_GATE_BIAS_INIT))

        self.d_x = d_x
                # --- Kappa mixing parameter (Section 2.7) ---
                # Per-dimension learnable logits -> sigmoid -> clamp(min=0.85)
                # Controls blending between AR proposal and y-only anchor
                # Init at 1.7 -> sigmoid ~0.85 (strong y-only anchoring initially)
        self.kappa_logits = nn.Parameter(torch.full((d_x,), 1.7))

        self.last_diag = {}

    def kappa_vec(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_logits).clamp(min=0.85)


    def forward(self, y_feat_only, x_prev=None, b_prev=None, sample=False,
                x_yonly_prop=None, kappa=None, innov_prev=None, return_parts=False,
                lag_values=None, lag_mask=None):

        y_last = y_feat_only

        h = self.backbone(y_feat_only)
        z = self.proj_out(h)
        d1, d4 = None, None

                # --- Innovation feedback (Section 2.6) ---
                # Modulate z BEFORE observation head (dmu_y) to allow innovation
                # to influence the state increment. Also compute direct dx_innov
                # correction from (innovation_buffer, x_prev).
        # --- Innovation embedding ---
        z_innov = z
        dx_innov = torch.zeros((z.size(0), self.d_x), device=z.device, dtype=z.dtype)
        if (innov_prev is not None) and (not _DISABLE_INNOV_DIAG):
            if innov_prev.dim() == 1:
                innov_prev = innov_prev.unsqueeze(0)
            innov_scalar = innov_prev[:, -1:]
            z_innov = z + self.innov_embed(innov_scalar)
            if x_prev is not None:
                dx_innov = self.innov_direct(
                    torch.cat([innov_prev[:, :8], x_prev], dim=-1)
                )

        dmu_y_raw = self.head_dx_y(z_innov)
        dls_y_raw = self.head_logsig(z)
        with torch.no_grad():
            sat_y = (dmu_y_raw.abs() > _SAT_THRESH).float().mean().item()

                # Scale through tanh to bound increments: dmu in [-2, +2] sigma
                # dls_y (log-sigma adjustment) bounded to [-0.5, +0.5]
        dmu_y = _DELTA_SCALE * torch.tanh(dmu_y_raw)
        dls_y = _LOGSIG_SCALE * torch.tanh(dls_y_raw)

        _z_bias = z.detach() + 0.10 * (z - z.detach())
        b_hat_raw = self.head_corr(_z_bias)
        g_logits = self.head_corr_gate(_z_bias)


        if b_prev is None:
            b_prev = torch.zeros((z.size(0), self.d_x), device=z.device, dtype=z.dtype)
        b_prev0 = b_prev
        b_max = 0.3
        b_hat = b_hat_raw.clamp(-b_max, b_max)
        b_t, g_b = gated_ema_bias_update(b_prev0, b_hat, g_logits, g_min=0.05, g_max=0.20, b_max=b_max)
        c_t = b_t - b_prev0

        sat_ar = 0.0
        g = None

                # === Y-ONLY PATH (Stage 2b, Section 2.5) ===
                # When x_prev is None, use separate y-only backbone for initial estimate.
                # MLP processes last frame, temporal Conv1d processes full window.
                # Combined features -> head_dx_y_separate -> dmu_y_sep
        if x_prev is None:
            z_yonly_sep = self.yonly_backbone(y_last)
            # MLP DOES NOT use temporal concatenation in training
            dmu_y_sep = _DELTA_SCALE * torch.tanh(self.head_dx_y_separate(z_yonly_sep))

            x_prop = dmu_y_sep
            x_logsig = dls_y
            x_mu = x_prop
            g = torch.ones_like(x_mu)
        else:
            if _DISABLE_AR_DIAG:
                dmu_ar = torch.zeros_like(dmu_y)
                dls_ar = torch.zeros_like(dls_y)
            else:
                ar_params = self.ar(x_prev)
                with torch.no_grad():
                    sat_ar = (ar_params[:, :self.d_x].abs() > _SAT_THRESH).float().mean().item()
                dmu_ar = _DELTA_SCALE * torch.tanh(ar_params[:, :self.d_x])
                dls_ar = _LOGSIG_SCALE * torch.tanh(ar_params[:, self.d_x:])

                        # --- Build AR proposal (Section 2.7) ---
                        # x_prop = x_hat(t-1) + dmu_y + dmu_AR + b(t) + dx_innov
                        # Four additive correction sources:
                        #   dmu_y: WaveNet observation increment
                        #   dmu_ar: AR sub-network increment from x_prev
                        #   b_t: slow drift correction from EMA integrator
                        #   dx_innov: fast correction from innovation feedback
            x_prop = x_prev + (dmu_y + dmu_ar) + b_t + dx_innov
            x_logsig = dls_y + dls_ar

            if x_yonly_prop is not None:
                k = self.kappa_vec() if (kappa is None) else kappa
                k = _ensure_kappa_vec_or_batch(k, x_prop=x_prop, D=self.d_x)
                                # --- Kappa mixing (Section 2.7) ---
                                # Blend AR proposal with detached y-only estimate.
                                # x_yonly_prop is detached to prevent kappa gradients from
                                # distorting the y-only backbone's learning.
                x_prop = (1.0 - k) * x_prop + k * x_yonly_prop.detach()
                        # --- State gate (Section 2.7, Section 5.8) ---
                        # g = g_floor + (1 - g_floor) * sigmoid(gate_logits)
                        # WHY gate floor 0.65: at 6 Hz, lag-1 autocorrelation > 0.99
                        # Without the floor, gate learns to shut (g->0) and model
                        # ignores all observations/corrections (persistence collapse).
                        # The floor forces minimum 65% update at every step.
            g_raw = torch.sigmoid(self.gate(z).float())
            g = self.g_floor + (1.0 - self.g_floor) * g_raw
            if _FORCE_GATE_OPEN:
                g = torch.ones_like(g)
                        # Gated state update: x_hat(t) = x_hat(t-1) + g*(x_prop - x_hat(t-1))
            x_mu = x_prev + g * (x_prop - x_prev)

        x_mu = torch.clamp(x_mu, -10.0, 10.0)
        x_logsig = x_logsig.clamp(-6.0, 2.0)
        x_hat = (x_mu + torch.exp(x_logsig) * torch.randn_like(x_mu)) if sample else x_mu

        with torch.no_grad():
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
                "sat_y": float(sat_y), "sat_ar": float(sat_ar),
                "ct_abs_mean": float(c_t.detach().abs().mean().item()),
                "bt_abs_mean": float(b_t.detach().abs().mean().item()),
                "gb_mean": float(gb.mean().item()),
                "gb_p10": _pct(gb, 10), "gb_p50": _pct(gb, 50), "gb_p90": _pct(gb, 90),
                "dx_total_abs_mean": dx_total_abs_mean,
                "dx_bias_abs_mean": dx_bias_abs_mean,
                "bias_fraction": bias_frac,
            }
        if return_parts:
            dx = (x_mu - x_prev) if (x_prev is not None) else x_mu
            return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, dx
        return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, x_prop, g


# ===========================================================================
# YMemoryWaveNet (matches training exactly — lag cond, gated EMA, per-dim kappa, etc.)
# ===========================================================================

# ===========================================================================
# YMemoryWaveNet — main student encoder (Stage 2a+2b, Sections 2.4-2.7)
# ===========================================================================
# Estimates cardiovascular state x(t) from short-channel observations y(t)
# alone, without access to ground-truth physiology at inference time.
#
# Stage 2a (WaveNet backbone, Section 2.4):
#   Conv1d(d_yfeat->hidden) -> N causal dilated blocks -> Conv1d(hidden->d_mem) -> z
#   5 zero-init heads from z: dmu_y, log_sigma, bias estimate, bias gate, state gate
#   AR sub-net: 2-layer (d_x->64->2*d_x), spectral norm
#
# Stage 2b (Y-only backbone, Section 2.5):
#   Separate MLP (d_yfeat->256->128->64, GELU, LayerNorm, Dropout)
#   + dedicated temporal Conv1d encoder (d=[1,4,16])
#   Concatenated [MLP_out, temporal_out] -> head_dx_y_separate -> dmu_y_sep
#   No shared parameters with WaveNet backbone
#
# Two operating modes:
#   x_prev=None: Y-only path (Stage 2b) — observation-only state estimate
#   x_prev given: Full AR path combining all correction mechanisms
#
# State update equation (Section 2.7):
#   x_prop = x_hat(t-1) + dmu_y + dmu_AR + b(t) + dx_innov
#   x_prop = (1-kappa)*x_prop + kappa*x_hat_yonly  (observation anchoring)
#   x_hat(t) = x_hat(t-1) + g*(x_prop - x_hat(t-1))  (gated update)
#   where g in [g_floor, 1.0], g_floor=0.65 (prevents persistence collapse)
#
# Drift correction (Section 2.6):
#   Gated EMA bias: b(t) = (1-g_b)*b(t-1) + g_b*b_hat, g_b in [0.05,0.20]
#   Innovation: embedding(scalar) + direct MLP(8+d_x -> 128 -> 64 -> d_x)
#
# Outputs: (z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, x_prop, g)
#   z: WaveNet memory vector (d_mem-dimensional)
#   d1, d4: legacy (always None)
#   x_mu: predicted state mean after gating
#   x_logsig: predicted log-sigma per dimension
#   x_hat: x_mu (deterministic) or x_mu + noise (if sample=True)
#   b_t: updated bias state (carried to next step)
#   c_t: bias increment this step (b_t - b_prev)
#   x_prop: pre-gate proposal (diagnostic)
#   g: gate values per dimension (diagnostic)
class YMemoryWaveNet(nn.Module):
    def __init__(self, d_yfeat, d_mem=512, hidden=112, layers=6, dilations=None, d_x=4,
                 lags=None, cond_channels=None, lag_include_mask=True, lag_feat_dim=None):
        super().__init__()

        # Lag conditioning config
        self.lags = list(sorted(lags)) if (lags is not None and len(lags) > 0) else []
        self.num_lags = len(self.lags)
        self.use_lag_cond = self.num_lags > 0
        self.cond_channels = int(cond_channels) if (cond_channels is not None and int(cond_channels) > 0) else int(hidden)
        self.lag_feat_dim = int(lag_feat_dim) if (lag_feat_dim is not None and int(lag_feat_dim) > 0) else int(d_x)

                # Optional lag conditioning: projects past state predictions into
                # WaveNet block conditioning signal. Enabled when lags list is non-empty.
        self.lag_conditioner = None
        if self.use_lag_cond:
            self.lag_conditioner = LagConditioner(
                d_feat=self.lag_feat_dim, num_lags=self.num_lags,
                cond_channels=self.cond_channels, include_mask=bool(lag_include_mask),
            )

                # --- Stage 2a WaveNet backbone layers ---
                # inp: project d_yfeat observation features into hidden channels
        self.inp = nn.Conv1d(d_yfeat, hidden, kernel_size=1)
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128][:layers]

        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden, d, cond_channels=(self.cond_channels if self.use_lag_cond else 0))
            for d in dilations
        ])

        self.out = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1), nn.GELU(),
            nn.Conv1d(hidden, d_mem, kernel_size=1)
        )
        self.yonly_backbone = nn.Sequential(
            nn.Linear(d_yfeat, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
        )
        self.yonly_temporal_inp = nn.Conv1d(d_yfeat, 64, kernel_size=1)
        self.yonly_temporal_convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=3, dilation=d, padding=0)
            for d in [1, 4, 16]
        ])
        self.yonly_temporal_out = nn.Conv1d(64, 64, kernel_size=1)
        self.head_dx_y_separate = nn.Linear(64 + 64, d_x)
        self.register_buffer('g_floor', torch.tensor(0.65))

        self.head_dx_y = nn.Linear(d_mem, d_x)
        self.head_logsig = nn.Linear(d_mem, d_x)
        self.head_corr = nn.Linear(d_mem, d_x)
        self.head_corr_gate = nn.Linear(d_mem, d_x)
        nn.init.normal_(self.head_corr_gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_corr_gate.bias, -0.224)
        nn.init.zeros_(self.head_corr.weight); nn.init.zeros_(self.head_corr.bias)
        nn.init.zeros_(self.head_dx_y.weight); nn.init.zeros_(self.head_dx_y.bias)
        nn.init.zeros_(self.head_logsig.weight); nn.init.zeros_(self.head_logsig.bias)

        self.innov_embed = nn.Linear(1, d_mem)
        nn.init.zeros_(self.innov_embed.weight)
        nn.init.zeros_(self.innov_embed.bias)
        self.innov_direct = nn.Sequential(
            nn.Linear(8 + d_x, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, d_x),
        )
        for m in self.innov_direct.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight);
                nn.init.zeros_(m.bias)

        self.ar = nn.Sequential(
            _sn(nn.Linear(d_x, 64)), nn.GELU(), nn.LayerNorm(64),
            _sn(nn.Linear(64, 2 * d_x))
        )
        for m in self.ar.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        self.gate = nn.Linear(d_mem, d_x)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(_GATE_BIAS_INIT))

        self.d_x = d_x
        self.kappa_logits = nn.Parameter(torch.full((d_x,), 1.7))

        self.last_diag = {}

    def kappa_vec(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_logits).clamp(min=0.85)


    def forward(self, y_seq, x_prev=None, b_prev=None, sample=False,
                x_yonly_prop=None, kappa=None, innov_prev=None, return_parts=False,
                lag_values=None, lag_mask=None):

                # Extract last frame for y-only backbone (Stage 2b uses single-frame MLP)
        y_last = y_seq[:, -1, :]

                # --- Stage 2a WaveNet forward pass (Section 2.4) ---
                # Process 128-step observation window through causal backbone:
                # y_seq [B,T,F] -> transpose -> Conv1d blocks -> take last timestep -> z
        x = y_seq.transpose(1, 2)
        h = self.inp(x)

        cond = None
        if self.use_lag_cond:
            if lag_values is None:
                raise ValueError("Model configured with lags but lag_values=None was passed.")
            cond = self.lag_conditioner(lag_values, lag_mask)

        for blk in self.blocks:
            h = blk(h, cond=cond)

        last = h[:, :, -1:]
        z = self.out(last).squeeze(-1)

        d1, d4 = None, None

        # --- Innovation embedding ---
        z_innov = z
        dx_innov = torch.zeros((z.size(0), self.d_x), device=z.device, dtype=z.dtype)
        if (innov_prev is not None) and (not _DISABLE_INNOV_DIAG):
            if innov_prev.dim() == 1:
                innov_prev = innov_prev.unsqueeze(0)
            innov_scalar = innov_prev[:, -1:]
            z_innov = z + self.innov_embed(innov_scalar)
            if x_prev is not None:
                dx_innov = self.innov_direct(
                    torch.cat([innov_prev[:, :8], x_prev], dim=-1)
                )

        dmu_y_raw = self.head_dx_y(z_innov)
        dls_y_raw = self.head_logsig(z)
        with torch.no_grad():
            sat_y = (dmu_y_raw.abs() > _SAT_THRESH).float().mean().item()

        dmu_y = _DELTA_SCALE * torch.tanh(dmu_y_raw)
        dls_y = _LOGSIG_SCALE * torch.tanh(dls_y_raw)

        _z_bias = z.detach() + 0.10 * (z - z.detach())
        b_hat_raw = self.head_corr(_z_bias)
        g_logits = self.head_corr_gate(_z_bias)


        if b_prev is None:
            b_prev = torch.zeros((z.size(0), self.d_x), device=z.device, dtype=z.dtype)
        b_prev0 = b_prev
        b_max = 0.3
        b_hat = b_hat_raw.clamp(-b_max, b_max)
        b_t, g_b = gated_ema_bias_update(b_prev0, b_hat, g_logits, g_min=0.05, g_max=0.20, b_max=b_max)
        c_t = b_t - b_prev0

        sat_ar = 0.0
        g = None

        if x_prev is None:
            z_yonly_sep = self.yonly_backbone(y_last)

            # USE DEDICATED TEMPORAL ENCODER (not WaveNet z)
            if y_seq.dim() == 3:
                xt = y_seq.transpose(1, 2)
                ht = self.yonly_temporal_inp(xt)
                for conv in self.yonly_temporal_convs:
                    pad = 2 * conv.dilation[0]
                    ht_padded = F.pad(ht, (pad, 0))
                    ht = ht + torch.tanh(conv(ht_padded))
                ht = self.yonly_temporal_out(ht)
                z_yonly_temp = ht[:, :, -1]  # [B, 64]
            else:
                z_yonly_temp = torch.zeros(y_last.size(0), 64, device=y_last.device, dtype=y_last.dtype)

            z_combined = torch.cat([z_yonly_sep, z_yonly_temp], dim=-1)  # [B, 128]
            dmu_y_sep = _DELTA_SCALE * torch.tanh(self.head_dx_y_separate(z_combined))

            x_prop = dmu_y_sep
            x_logsig = dls_y
            x_mu = x_prop
            g = torch.ones_like(x_mu)
        else:
            if _DISABLE_AR_DIAG:
                dmu_ar = torch.zeros_like(dmu_y)
                dls_ar = torch.zeros_like(dls_y)
            else:
                ar_params = self.ar(x_prev)
                with torch.no_grad():
                    sat_ar = (ar_params[:, :self.d_x].abs() > _SAT_THRESH).float().mean().item()
                dmu_ar = _DELTA_SCALE * torch.tanh(ar_params[:, :self.d_x])
                dls_ar = _LOGSIG_SCALE * torch.tanh(ar_params[:, self.d_x:])

            x_prop = x_prev + (dmu_y + dmu_ar) + b_t + dx_innov
            x_logsig = dls_y + dls_ar

            if x_yonly_prop is not None:
                k = self.kappa_vec() if (kappa is None) else kappa
                k = _ensure_kappa_vec_or_batch(k, x_prop=x_prop, D=self.d_x)
                x_prop = (1.0 - k) * x_prop + k * x_yonly_prop.detach()
            g_raw = torch.sigmoid(self.gate(z).float())
            g = self.g_floor + (1.0 - self.g_floor) * g_raw
            if _FORCE_GATE_OPEN:
                g = torch.ones_like(g)
            x_mu = x_prev + g * (x_prop - x_prev)

        x_mu = torch.clamp(x_mu, -10.0, 10.0)
        x_logsig = x_logsig.clamp(-6.0, 2.0)
        x_hat = (x_mu + torch.exp(x_logsig) * torch.randn_like(x_mu)) if sample else x_mu

        with torch.no_grad():
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
                "sat_y": float(sat_y), "sat_ar": float(sat_ar),
                "ct_abs_mean": float(c_t.detach().abs().mean().item()),
                "bt_abs_mean": float(b_t.detach().abs().mean().item()),
                "gb_mean": float(gb.mean().item()),
                "gb_p10": _pct(gb, 10), "gb_p50": _pct(gb, 50), "gb_p90": _pct(gb, 90),
                "dx_total_abs_mean": dx_total_abs_mean,
                "dx_bias_abs_mean": dx_bias_abs_mean,
                "bias_fraction": bias_frac,
            }
        if return_parts:
            dx = (x_mu - x_prev) if (x_prev is not None) else x_mu
            return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, dx
        return z, d1, d4, x_mu, x_logsig, x_hat, b_t, c_t, x_prop, g


# ===========================================================================
# Build student from payload (matches _build_student_from_payload in training)
# ===========================================================================

@torch.no_grad()
# ===========================================================================
# Reconstruct student from saved payload (matches training builder exactly)
# ===========================================================================
# Reads ALL architecture params from payload dict (d_mem, hidden, layers, etc.)
# Sets spectral norm and diagnostic flags BEFORE construction to ensure
# state_dict keys match. Warns on missing/unexpected keys.
#
# Called by:
#   - export_decoder_pack() in exp_base_dynamic.py
#   - Any inference script that loads a saved encoder checkpoint
def build_student_from_payload(payload: dict, d_yfeat: int, device: str):
    """Recreate the student from saved payload. Reads ALL arch params from payload."""
    import time as _time
    t0 = _time.time()

    student_type = payload.get("student_type", "wavenet")
    d_mem = int(payload["d_mem"])
    win = payload.get("wavenet_win", None)
    lags_saved = payload.get("lags", [])

    # Set spectral norm flag BEFORE constructing (affects weight key names)
    sn_flag = payload.get("spectral_norm_ar", None)
    if sn_flag is not None:
        if bool(sn_flag) != _USE_SPECTRAL_NORM_AR:
            print(f"[WARN] spectral_norm_ar mismatch: payload={sn_flag}, module={_USE_SPECTRAL_NORM_AR}. "
                  f"Updating module flag to match payload.")
            set_spectral_norm_flag(bool(sn_flag))

    # Set AR/innov diag flags from payload
    set_diag_flags(
        disable_ar=payload.get("ar_disabled", False),
        disable_innov=payload.get("innov_disabled", False),
    )

    if student_type == "wavenet":
        assert win is not None, "Payload missing wavenet_win"
        student = YMemoryWaveNet(
            d_yfeat=d_yfeat,
            d_mem=int(payload.get("d_mem", 512)),
            hidden=int(payload.get("hidden", 112)),
            layers=int(payload.get("layers", 6)),
            dilations=None,
            d_x=int(payload.get("d_x", 4)),
            lags=lags_saved,
            cond_channels=int(payload.get("cond_channels", 112)),
            lag_include_mask=bool(payload.get("lag_include_mask", True)),
            lag_feat_dim=int(payload.get("lag_feat_dim", 4)),
        )
    else:
        student = YMemoryMLP(d_yfeat=d_yfeat, d_mem=d_mem)

    missing, unexpected = student.load_state_dict(payload["student_state"], strict=False)
    if missing:
        print(f"[WARN] MISSING keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        print(f"[WARN] UNEXPECTED keys ({len(unexpected)}): {unexpected[:10]}...")

    student = student.to(device)
    student.eval()
    print(f"[build_student] type={student_type} | device={device} | "
          f"lags={lags_saved} | total time={_time.time()-t0:.3f}s")
    return student


# ===========================================================================
# Consistent forward call (matches training _student_forward_consistent)
# ===========================================================================

@torch.no_grad()
# ===========================================================================
# Consistent forward pass matching training convention exactly
# ===========================================================================
# Matches _student_forward_consistent in exp_base_dynamic.py:
#   1. Compute compound kappa = kappa_corr * student.kappa_vec()
#   2. If kappa > 0: compute y-only proposal WITHOUT innovation
#   3. Main forward with innovation, compound kappa, lags
#
# CRITICAL: y-only proposal must NOT use innovation (prevents circular
# dependency where innovation depends on y-only which depends on innovation).
def student_forward_consistent(
    student: nn.Module,
    y_in: torch.Tensor,
    x_prev: torch.Tensor | None,
    *,
    kappa_corr: float = 0.0,
    innovation: torch.Tensor | None = None,
    lag_values: torch.Tensor | None = None,
    lag_mask: torch.Tensor | None = None,
):
    """
    Matches training convention exactly:
      1. Compute compound kappa = kappa_corr * student.kappa_vec()
      2. If kappa > 0: compute y-only proposal WITHOUT innovation
      3. Main forward with innovation, compound kappa, lags
    """
    # Compound kappa (Bug 4 fix)
    if hasattr(student, "kappa_vec") and kappa_corr > 0.0:
        kappa_eff = float(kappa_corr) * student.kappa_vec().view(1, -1)
    else:
        kappa_eff = float(kappa_corr)

    xy = None
    if float(kappa_corr) > 0.0:
        out_yonly = student(
            y_in, x_prev=None, b_prev=None, sample=False, innov_prev=None,
            lag_values=lag_values, lag_mask=lag_mask,
        )
        xy = out_yonly[3]

    return student(
        y_in, x_prev=x_prev, b_prev=None, sample=False,
        x_yonly_prop=xy, kappa=kappa_eff, innov_prev=innovation,
        lag_values=lag_values, lag_mask=lag_mask,
    )


# ===========================================================================
# Standalone rollout (y-only ground truth, optional innovation via teacher)
# Matches _student_rollout_single_stream in export_decoder_pack exactly.
# ===========================================================================

@torch.no_grad()
# ===========================================================================
# Full closed-loop rollout from Y observations (inference entry point)
# ===========================================================================
# Matches _student_rollout_single_stream in export_decoder_pack exactly.
# Runs the student autoregressively for Nc steps:
#
# Per step:
#   1. Store prev_unclamped (diagnostic)
#   2. Clamp prev for feeding (safety bound, +/-clamp_val sigma)
#   3. If use_innovation: compute innovation via frozen teacher
#      innovation(t) = y_obs(t) - y_hat(x_hat(t-1)) via Jacobian projection
#   4. If kappa > 0 and not in warmup: compute y-only proposal
#   5. Main student forward with prev, b_prev, innovation, kappa
#   6. If warmup: use true state as prev (bias integrator still learns)
#      Else: use predicted state as prev (closed-loop)
#
# WHY warmup (32 steps): the bias integrator needs ~50 steps to converge.
# Without warmup, early errors compound before drift correction locks on.
#
# WHY lag_values substitution: at inference, ground-truth past states are
# unavailable. The student's own past predictions are used instead.
#
# Returns: mu_roll, b_roll, c_roll, prevfed, prev_unclamped (all CPU)
#   mu_roll: predicted state trajectory [Nc, d_x]
#   b_roll: bias state trajectory [Nc, d_x]
#   c_roll: bias increment trajectory [Nc, d_x]
#   prevfed: clamped previous state that was fed [Nc, d_x]
#   prev_unclamped: raw previous state before clamping [Nc, d_x]
def rollout_from_y(
        *,
        student: nn.Module,
        y_in_cpu: torch.Tensor,
        y_std_cpu: torch.Tensor,
        x0_cpu: torch.Tensor,
        clamp_val: float | None = None,
        kappa_corr: float = 0.0,
        use_innovation: bool = True,
        ygx_model: nn.Module | None = None,
        cond_gain_for_teacher: float = 20.0,
        X_cond_cent_cpu: torch.Tensor | None = None,
        d_state: int = 4,
        lag_values_cpu: torch.Tensor | None = None,
        lag_mask_cpu: torch.Tensor | None = None,
        device: str = "cpu",
        chunk_size: int = 1024,
        use_amp: bool = True,
        progress_every: int = 2000,
        warmup_steps: int = 32,
        X_true_cpu: torch.Tensor | None = None,
):
    from contextlib import nullcontext

    student.eval()
    if ygx_model is not None:
        ygx_model.eval()

    if use_innovation:
        assert ygx_model is not None, "ygx_model required when use_innovation=True"
        assert X_cond_cent_cpu is not None, "X_cond_cent_cpu required when use_innovation=True"

    Nc = int(y_in_cpu.size(0))
    d_x = int(x0_cpu.numel())

    if hasattr(student, 'kappa_vec') and kappa_corr > 0.0:
        kappa_eff = float(kappa_corr) * student.kappa_vec().view(1, -1)
    else:
        kappa_eff = float(kappa_corr)

    prev = x0_cpu.view(1, -1).to(device=device, dtype=torch.float32)
    if clamp_val is not None and clamp_val > 0:
        prev = prev.clamp(-float(clamp_val), float(clamp_val))

    b_prev = torch.zeros((1, d_x), device=device, dtype=torch.float32)
        # Innovation buffer: 8-step sliding window of scalar innovations (Section 2.6)
        # Fed to innov_direct MLP alongside x_prev for fast correction
    innov_buffer = torch.zeros((1, 8), device=device, dtype=torch.float32)


    mu_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
    b_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
    c_roll = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
    prevfed = torch.empty((Nc, d_x), device=device, dtype=torch.float32)
    prev_unclamped = torch.empty((Nc, d_x), device=device, dtype=torch.float32)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and torch.cuda.is_available() and str(device).startswith("cuda"))
        else nullcontext()
    )


    import time as _time
    t0 = _time.time()

        # Zero-pad 4-dim student state to match teacher input dimensionality
    def _pad_for_teacher(x4, d_teacher):
        if x4.shape[-1] == d_teacher:
            return x4
        pad = torch.zeros(*x4.shape[:-1], d_teacher - x4.shape[-1], device=x4.device, dtype=x4.dtype)
        return torch.cat([x4, pad], dim=-1)

    for s in range(0, Nc, chunk_size):
        e = min(Nc, s + chunk_size)

        y_chunk = y_in_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
        yobs_chunk = y_std_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)

        lv_chunk = lag_values_cpu[s:e].to(device=device, dtype=torch.float32,
                                          non_blocking=True) if lag_values_cpu is not None else None
        lm_chunk = lag_mask_cpu[s:e].to(device=device, dtype=torch.float32,
                                        non_blocking=True) if lag_mask_cpu is not None else None

        cond_chunk = None
        if use_innovation:
            cond_chunk = X_cond_cent_cpu[s:e].to(device=device, dtype=torch.float32, non_blocking=True)

        for i in range(e - s):
            t = s + i

            # Store BEFORE clamping
            prev_unclamped[t] = prev.squeeze(0)

            prev_fed = prev
            if clamp_val is not None and clamp_val > 0:
                prev_fed = prev_fed.clamp(-float(clamp_val), float(clamp_val))
            prevfed[t] = prev_fed.squeeze(0)

            y_t = y_chunk[i:i + 1]

            lv_t = lv_chunk[i:i + 1] if lv_chunk is not None else None
            if (lv_t is not None) and (t > 0) and hasattr(student, 'lags'):
                lv_t = lv_t.clone()
                for k_idx, lag_dist in enumerate(student.lags):
                    if t >= lag_dist:
                        pred_prev = mu_roll[t - lag_dist].view(1, -1)
                        lv_t[:, k_idx, :] = pred_prev
            lm_t = lm_chunk[i:i + 1] if lm_chunk is not None else None

            innov_prev= None
            if use_innovation:
                with torch.enable_grad():
                    x_req = prev_fed.detach().requires_grad_(True)
                    curr_cond = cond_chunk[i:i + 1].clone()
                    curr_cond[:, :d_state] = x_req
                    d_teacher = int(curr_cond.shape[-1])
                    y_prev_hat = ygx_model(_pad_for_teacher(curr_cond, d_teacher),
                                           cond_gain_scale=cond_gain_for_teacher)[0]
                    J = torch.autograd.grad(y_prev_hat.sum(), x_req, create_graph=False)[0]

                innov_scalar = (yobs_chunk[i:i + 1] - y_prev_hat.detach())
                J = J.detach()
                J_sq = J.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-6)
                innovation_4d = innov_scalar * J / J_sq
                norm_4d = innovation_4d.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
                scalar_mag = innov_scalar.abs()
                innovation = (innovation_4d * (scalar_mag / norm_4d)).detach()
                innov_buffer = torch.cat([innov_buffer[:, 1:], innov_scalar.detach()], dim=1)

            # --- Y-ONLY PROPOSAL ---
            in_warmup = (warmup_steps > 0) and (t < warmup_steps) and (X_true_cpu is not None)
            xy = None
            if kappa_corr > 0.0 and not in_warmup:
                out_yonly = student(y_t, x_prev=None, b_prev=None, sample=False,
                                    innov_prev=None, lag_values=lv_t, lag_mask=lm_t)
                xy = out_yonly[3]

            # ONLY student main forward under amp_ctx (matches training)
            kappa_this_step = 0.0 if in_warmup else kappa_eff
            with amp_ctx:
                out = student(y_t, x_prev=prev_fed, b_prev=b_prev, sample=False,
                              x_yonly_prop=xy, kappa=kappa_this_step, innov_prev=innov_buffer if use_innovation else None,
                              lag_values=lv_t, lag_mask=lm_t)

            xmu = out[3]


            b_prev = out[6]
            c_t = out[7]

            mu_roll[t] = xmu.squeeze(0)
            b_roll[t] = b_prev.squeeze(0)
            c_roll[t] = c_t.squeeze(0)
            if (warmup_steps > 0) and (t < warmup_steps) and (X_true_cpu is not None):
                prev = X_true_cpu[t:t + 1].to(device=device, dtype=torch.float32)
                if clamp_val is not None and clamp_val > 0:
                    prev = prev.clamp(-float(clamp_val), float(clamp_val))
                # keep b_prev from model output — do NOT reset
            else:
                prev = xmu

        chunk_num = (s // chunk_size) + 1
        total_chunks = math.ceil(Nc / chunk_size)
        elapsed = _time.time() - t0
        it_s = e / max(1e-9, elapsed)
        eta = (Nc - e) / max(1, it_s)
        print(
            f"[rollout] chunk {chunk_num}/{total_chunks} | t={e}/{Nc} ({100 * e / Nc:.1f}%) | {it_s:.1f} it/s | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")

        del y_chunk, yobs_chunk, lv_chunk, lm_chunk
        if cond_chunk is not None:
            del cond_chunk

    return mu_roll.cpu(), b_roll.cpu(), c_roll.cpu(), prevfed.cpu(), prev_unclamped.cpu()