# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 6 — Ternary SwiGLU & RMSNorm.

Attention mixes information across positions; it does not transform it. The
other half of a transformer block — the part that does the per-position
computation — is the feed-forward network, sandwiched between two
normalization layers. Chapter 5 built the mixing half and made it see the
graph. This chapter builds the transforming half and makes it cheap.

The normalizer is RMSNorm, a radial projection that strips the magnitude of
each activation vector and keeps its direction, so the network's scale is set
by a learned gain rather than by the depth of the stack. The transformer is
SwiGLU, a gated feed-forward whose gate multiplies a *filtered* copy of its
own input, and whose hidden width follows the 8/3 convention that makes a
three-matrix gated block cost exactly what a two-matrix block costs. Both are
made ternary by Chapter 4's synapses, and both are pinned against the
production SDK.

Book theorems proved in the chapter and pinned by the tests:

  B.6.1  RMSNorm scale law. With n(x) = x / sqrt(mean(x^2) + eps), for every
         c > 0 the normalized vector is a positive multiple of the original,
         n(cx) = rho(c) n(x), with rho(c) = sqrt((m+eps)/(m+eps/c^2)). The
         direction is exactly invariant; the magnitude is invariant to within
         eps/(2m) for c >= 1, and exactly invariant when eps = 0.
  B.6.2  SiLU gate suppression. The gate g(z) = z*sigma(z) satisfies
         0 < g(z) < z for z > 0 and z < g(z) < 0 for z < 0, is bounded below
         by g_min = -0.278465 at z* = -1.278465, and satisfies g(z) -> 0 as
         z -> -infinity while g(z)/z -> 1 as z -> +infinity.
  B.6.3  Parameter accounting. With d_ff = floor(8/3 * d_model) the gated
         block has 3 d_model d_ff weights — the same 8 d_model^2 budget as a
         conventional hidden-4x feed-forward — and packs into
         ceil(3 d_model d_ff / 5) bytes.

Run the demonstration ledger::

    python3 code/gftac/ch06_swiglu.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow standalone execution: python3 code/gftac/ch06_swiglu.py
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch04_ternary import TernaryLinear, packed_bytes

__all__ = [
    "BookRMSNorm",
    "BookSwiGLU",
    "int_activation_quantize",
    "silu_minimum",
    "swiglu_param_count",
    "swiglu_packed_bytes",
    "demo_ledger",
]


# ----------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------

class BookRMSNorm(nn.Module):
    """Root-mean-square normalization: a radial projection plus a gain.

    Unlike LayerNorm, RMSNorm subtracts no mean — it divides the activation
    vector by the root mean square of its entries and multiplies by a learned
    per-channel gain ``scale``. The absence of the mean is not an oversight:
    the residual stream already carries a running mean, so re-centering at
    every block is redundant, and dropping it removes a reduction from the
    hot path. The map is a projection onto the (approximate) unit-RMS sphere,
    so the output magnitude is set by ``scale`` alone and not by the input
    scale — the property Theorem B.6.1 makes exact.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


# ----------------------------------------------------------------------
# Optional int activation quantization (mirrors the SDK's default path)
# ----------------------------------------------------------------------

def int_activation_quantize(
    x: torch.Tensor, bits: int = 8, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int quantization of activations.

    Returns ``(x_q, scale)`` with ``scale = q_max / (max|x| + eps)`` taken over
    the last (feature) axis, so every token gets its own dynamic range. This
    is the SDK's ``AbsMaxQuantizer.quantize_activation_int``, reproduced so
    the book can match the SDK's *default* feed-forward path, not only its
    weight-only path.
    """
    q_max = (1 << (bits - 1)) - 1
    scale = q_max / (x.abs().amax(dim=-1, keepdim=True) + eps)
    x_q = torch.clamp(torch.round(x * scale), -q_max, q_max)
    return x_q, scale


# ----------------------------------------------------------------------
# Ternary SwiGLU
# ----------------------------------------------------------------------

class BookSwiGLU(nn.Module):
    """Gated feed-forward network with ternary weights.

    The forward pass is the SwiGLU of the transformer literature,

        out = (SiLU(x W_gate) * (x W_up)) W_down,

    with the three projections of Chapter 4's :class:`TernaryLinear`. The
    hidden width defaults to ``floor(8/3 * d_model)`` (Theorem B.6.3). When
    ``act_bits`` is set, each projection's input is int-quantized first, which
    reproduces the SDK's default (8-bit) feed-forward; the default
    ``act_bits=None`` is the weight-only path the book has used since
    Chapter 4.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        act_bits: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(8 / 3 * d_model)
        self.act_bits = act_bits

        self.w_gate = TernaryLinear(d_model, self.d_ff, bias=False)
        self.w_up = TernaryLinear(d_model, self.d_ff, bias=False)
        self.w_down = TernaryLinear(self.d_ff, d_model, bias=False)

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_bits is None or self.act_bits >= 32:
            return x
        x_q, scale = int_activation_quantize(x, self.act_bits)
        return x + (x_q / scale - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(self._q(x)))
        up = self.w_up(self._q(x))
        return self.w_down(self._q(gate * up))


# ----------------------------------------------------------------------
# B.6.2 — the SiLU gate
# ----------------------------------------------------------------------

def silu_minimum(grid: int = 4_000_001) -> Tuple[float, float]:
    """Locate the global minimum of SiLU in double precision.

    Returns ``(z_star, g_min)``. The grid is computed in float64: in float32
    the minimum sits in a flat basin roughly ``4e-4`` wide where round-off
    dominates the true curvature, so a float32 arg-min returns an arbitrary
    point in the basin. The stationarity condition ``1 + z(1 - sigma(z)) = 0``
    is solved to grid resolution in float64, giving ``z* = -1.2784645``.
    """
    z = torch.linspace(-2.0, -0.5, grid, dtype=torch.float64)
    g = F.silu(z)
    i = int(g.argmin())
    return float(z[i]), float(g[i])


# ----------------------------------------------------------------------
# B.6.3 — parameter accounting
# ----------------------------------------------------------------------

def swiglu_param_count(d_model: int, d_ff: Optional[int] = None) -> int:
    """Weight count of a gated feed-forward: 3 d_model d_ff."""
    d_ff = d_ff if d_ff is not None else int(8 / 3 * d_model)
    return 3 * d_model * d_ff


def swiglu_packed_bytes(d_model: int, d_ff: Optional[int] = None) -> int:
    """Packed ternary footprint of a gated feed-forward, in bytes."""
    return packed_bytes(swiglu_param_count(d_model, d_ff))


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def demo_ledger() -> str:
    lines: List[str] = []
    lines.append("Chapter 6 — Ternary SwiGLU & RMSNorm, measured")
    lines.append("=" * 68)

    # -- B.6.1: RMSNorm scale law ---------------------------------------
    lines.append("B.6.1 — RMSNorm is a radial map (direction invariant)")
    torch.manual_seed(0)
    x = torch.randn(1, 8)
    m = float(x.pow(2).mean())
    eps = 1e-6
    n_x = x / math.sqrt(m + eps)
    lines.append(f"  input RMS        = {float(x.pow(2).mean().sqrt()):.6f}"
                 f"   (mean(x^2) = {m:.6f})")
    lines.append(f"  RMS(n(x))        = {float(n_x.pow(2).mean().sqrt()):.6f}"
                 f"   target sqrt(m/(m+eps)) = {math.sqrt(m / (m + eps)):.6f}")
    lines.append(f"  {'c':>8}{'cos(n(cx), n(x))':>20}{'rho(c)':>14}{'rho(c) pred':>14}")
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        n_cx = (c * x) / math.sqrt((c * x).pow(2).mean() + eps)
        cos = float(F.cosine_similarity(n_cx.flatten(), n_x.flatten(), dim=0))
        rho = float(n_cx.norm() / n_x.norm())
        rho_pred = math.sqrt((m + eps) / (m + eps / c ** 2))
        lines.append(f"  {c:>8.2f}{cos:>20.10f}{rho:>14.6f}{rho_pred:>14.6f}")
    lines.append("")

    # -- B.6.2: SiLU gate ------------------------------------------------
    z_star, g_min = silu_minimum()
    lines.append("B.6.2 — the SiLU gate g(z) = z*sigma(z)")
    lines.append(f"  global minimum at z* = {z_star:.6f},  g(z*) = {g_min:.6f}")
    lines.append(f"  {'z':>8}{'g(z)':>12}{'0<g<z or z<g<0':>18}")
    for z in (-4.0, -2.0, -1.278465, -0.5, 0.0, 0.5, 2.0, 4.0):
        g = float(F.silu(torch.tensor(z)))
        inside = (0.0 < g < z) or (z < g < 0.0) or (z == 0.0 and g == 0.0)
        lines.append(f"  {z:>8.3f}{g:>12.6f}{str(inside):>18}")
    lines.append("")

    # -- B.6.2: gate sparsity --------------------------------------------
    torch.manual_seed(1)
    z = torch.randn(1_000_000)
    g = F.silu(z)
    lines.append("B.6.2 — gate statistics over z ~ N(0, 1)")
    lines.append(f"  E[g(z)]                = {float(g.mean()):.6f}")
    lines.append(f"  E[|g(z)|]              = {float(g.abs().mean()):.6f}")
    lines.append(f"  P(g(z) < 0)            = {float((g < 0).float().mean()):.6f}"
                 f"   (exactly 1/2 by symmetry)")
    lines.append(f"  P(|g(z)| < 0.1)        = {float((g.abs() < 0.1).float().mean()):.6f}"
                 f"   (gate near-zero fraction)")
    lines.append(f"  bounded below by g_min = {g_min:.6f} (never exceeds this in magnitude below)")
    lines.append("")

    # -- B.6.3: parameter accounting -------------------------------------
    lines.append("B.6.3 — parameter accounting (8/3 convention)")
    for d_model in (64,):
        d_ff = int(8 / 3 * d_model)
        n = swiglu_param_count(d_model, d_ff)
        lines.append(f"  d_model = {d_model}, d_ff = floor(8/3 * d_model) = {d_ff}")
        lines.append(f"  weights = 3 * {d_model} * {d_ff} = {n:,}")
        lines.append(f"  8 * d_model^2 (unfloored) = {8 * d_model ** 2:,}"
                     f"   (gated FFN matches a hidden-4x FFN)")
        lines.append(f"  packed ternary footprint = ceil({n:,}/5) = "
                     f"{swiglu_packed_bytes(d_model):,} B")
    lines.append("")

    # -- parity with the SDK ---------------------------------------------
    from geoformer.layers.ffn import TernarySwiGLU as SDKFFN
    from geoformer.models.block import RMSNorm as SDKRMSNorm

    torch.manual_seed(2)
    d_model, n = 16, 5
    h = torch.randn(2, n, d_model)

    sdk_norm = SDKRMSNorm(d_model)
    book_norm = BookRMSNorm(d_model)
    book_norm.load_state_dict(sdk_norm.state_dict())
    with torch.no_grad():
        diff_norm = float((book_norm(h) - sdk_norm(h)).abs().max())
    lines.append("Parity — RMSNorm vs geoformer.models.block.RMSNorm")
    lines.append(f"  max |book - sdk| = {diff_norm:.3e}")
    lines.append("")

    # weight-only FFN parity (act_bits disabled on both)
    sdk_ffn = SDKFFN(d_model=d_model, mode="ternary")
    for lin in (sdk_ffn.w_gate, sdk_ffn.w_up, sdk_ffn.w_down):
        lin.act_bits = None
    book_ffn = BookSwiGLU(d_model=d_model, act_bits=None)
    book_ffn.load_state_dict(sdk_ffn.state_dict())
    with torch.no_grad():
        diff_ffn = float((book_ffn(h) - sdk_ffn(h)).abs().max())
    lines.append("Parity — TernarySwiGLU, weight-only (act_bits=None)")
    lines.append(f"  max |book - sdk| = {diff_ffn:.3e}")
    lines.append("")

    # default FFN parity (8-bit activations on both)
    torch.manual_seed(3)
    sdk_ffn8 = SDKFFN(d_model=d_model, mode="ternary")  # act_bits=8 default
    book_ffn8 = BookSwiGLU(d_model=d_model, act_bits=8)
    book_ffn8.load_state_dict(sdk_ffn8.state_dict())
    with torch.no_grad():
        diff_ffn8 = float((book_ffn8(h) - sdk_ffn8(h)).abs().max())
    lines.append("Parity — TernarySwiGLU, SDK default (8-bit activations)")
    lines.append(f"  max |book - sdk| = {diff_ffn8:.3e}")
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
