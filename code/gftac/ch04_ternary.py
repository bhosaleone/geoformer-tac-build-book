# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 4 — Ternary Synapses: BitNet b1.58.

The lattice of Chapter 2 and the dynamics of Chapter 3 are only useful if
they can be *carried* — stored, shipped, and executed on hardware that has
no floating-point multiplier to spare. This chapter does the carrying. It
replaces the high-precision synapse with a single **trit**, an element of
{-1, 0, +1}, and pays for that drastic reduction with three pieces of
machinery: a scale that keeps the weights in range, a gradient estimator
that lets a non-differentiable rounding be trained, and a packing scheme
that turns log2(3) bits of information into whole bytes.

Book theorems proved in the chapter and pinned by the tests:

  B.4.1  Ternary quantization is scale-invariant and variance-preserving.
         The scale gamma = mean|w| is homogeneous of degree one, so the
         normalized weight w/gamma — and hence the trit pattern
         w_q = clip(round(w/gamma), -1, 1) — is unchanged when w is
         rescaled by any c > 0. The effective weight w_eff = w_q * gamma
         therefore carries second moment E[w_eff^2] = E[w_q^2] * gamma^2,
         a fixed fraction of gamma^2 set by the *shape* of w's
         distribution and not by its magnitude.
  B.4.2  Straight-through estimation has the identity gradient. With the
         effective weight w_eff = w + (w_q gamma - w).detach(), the
         parameter gradient is exactly dL/dw_eff, the gradient the network
         would use if quantization were the identity map.
  B.4.3  Five trits fit in one byte. Since 3^5 = 243 < 256, the map
         (t_0,...,t_4) -> sum_k (t_k + 1) 3^k is injective into [0, 255],
         so any weight vector packs losslessly at exactly ceil(n/5) bytes
         (1.6 bits/parameter).

Run the demonstration ledger::

    python3 code/gftac/ch04_ternary.py
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RoundSTE",
    "ternary_quantize",
    "TernaryLinear",
    "variance_preserving_init_",
    "pack_trits",
    "unpack_trits",
    "packed_bytes",
    "packed_bits_per_parameter",
    "FOUNDATION_PARAMS",
    "demo_ledger",
]

# The GeoFormer foundation model (SDK config: in_features=1433,
# num_classes=7, d_model=64, n_layers=3, n_heads=4) has this many trainable
# parameters. The parity test re-derives it live from the SDK.
FOUNDATION_PARAMS = 249_665


# ----------------------------------------------------------------------
# The rounding operator, with a straight-through gradient
# ----------------------------------------------------------------------

class RoundSTE(torch.autograd.Function):
    """Round in the forward pass; pass the gradient through unchanged.

    ``round`` has zero derivative almost everywhere, so a literal autograd
    through it would learn nothing. The straight-through estimator (STE)
    replaces that zero with the identity: the forward value is the rounded
    one, the backward value is whatever arrived. It is the deliberate,
    documented lie that makes quantized networks trainable.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# ----------------------------------------------------------------------
# B.4.1 — quantization and the scale
# ----------------------------------------------------------------------

def ternary_quantize(
    w: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58 ternary quantization of ``w``.

    Returns ``(w_q, gamma)`` where ``gamma = mean|w| + eps`` is the
    per-tensor scale and ``w_q = clip(round(w/gamma), -1, 1)`` is the trit
    tensor. The effective, de-quantized weight is ``w_q * gamma``. The
    ``eps`` guards the all-zero weight tensor; it is the same guard the
    production SDK uses, so the two agree bit-for-bit.
    """
    gamma = torch.mean(torch.abs(w)) + eps
    w_scaled = w / gamma
    w_q = torch.clamp(RoundSTE.apply(w_scaled), -1.0, 1.0)
    return w_q, gamma


def variance_preserving_init_(tensor: torch.Tensor, mode: str = "ternary") -> torch.Tensor:
    """Initialize so that quantization does not attenuate the signal.

    For ternary weights the scale-normalized trits have second moment
    ``E[w_q^2] = 2/3`` (Theorem B.4.1), so a Gaussian draw of standard
    deviation ``sqrt(1.5/fan_in)`` yields an effective weight variance that
    keeps activations O(1) through a stack of layers rather than decaying
    geometrically. Integer modes use ``sqrt(2/fan_in)``.
    """
    with torch.no_grad():
        fan_in = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        if mode == "ternary":
            std = math.sqrt(1.5 / max(1, fan_in))
        elif mode.startswith("int"):
            std = math.sqrt(2.0 / max(1, fan_in))
        else:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            return tensor
        tensor.normal_(0.0, std)
    return tensor


class TernaryLinear(nn.Module):
    """A linear layer whose weights are ternary at inference.

    The forward pass de-quantizes to ``w_eff`` and runs an ordinary
    ``F.linear``. The trick is the detach in ``effective_weight``: it makes
    the parameter gradient exactly the gradient of the de-quantized layer
    (Theorem B.4.2), so backpropagation trains the latent full-precision
    weight while inference reads only its trits.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        variance_preserving_init_(self.weight, mode="ternary")
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self) -> torch.Tensor:
        """The de-quantized weight, differentiable as the identity on ``w``."""
        w_q, gamma = ternary_quantize(self.weight)
        return self.weight + (w_q * gamma - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ----------------------------------------------------------------------
# B.4.3 — packing five trits into a byte
# ----------------------------------------------------------------------

def packed_bytes(n: int) -> int:
    """Bytes needed to pack ``n`` trits at five per byte: ``ceil(n/5)``."""
    return (int(n) + 4) // 5


def packed_bits_per_parameter() -> float:
    """The asymptotic cost of a trit, in bits: ``8/5 = 1.6``."""
    return 8.0 / 5.0


def pack_trits(trits: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
    """Pack trits in {-1, 0, +1} five to a byte, little-endian by position.

    The byte is ``sum_{k=0}^{4} (t_k + 1) 3^k``. Because 3^5 = 243 <= 255 the
    map is injective, so packing is lossless (Theorem B.4.3). A trailing
    group with fewer than five trits is zero-padded; ``unpack_trits`` takes
    the original length to drop the padding.
    """
    t = torch.as_tensor(trits, dtype=torch.long).flatten()
    if t.numel() and bool(((t < -1) | (t > 1)).any()):
        raise ValueError("trits must lie in {-1, 0, +1}")
    digits = (t + 1).tolist()
    out: List[int] = []
    for b in range(packed_bytes(len(digits))):
        byte = 0
        for k in range(5):
            idx = b * 5 + k
            d = digits[idx] if idx < len(digits) else 0
            byte += d * (3 ** k)
        out.append(byte)
    return torch.tensor(out, dtype=torch.uint8)


def unpack_trits(
    packed: Union[torch.Tensor, Sequence[int]], n: int
) -> torch.Tensor:
    """Invert :func:`pack_trits`, returning exactly ``n`` trits."""
    p = torch.as_tensor(packed, dtype=torch.long).flatten().tolist()
    out: List[int] = []
    for byte in p:
        for _ in range(5):
            out.append((byte % 3) - 1)
            byte //= 3
    return torch.tensor(out[:n], dtype=torch.long)


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def demo_ledger() -> str:
    """The Chapter 4 demonstration: quantization, STE, packing, footprint."""
    lines: List[str] = []
    lines.append("Chapter 4 — Ternary Synapses (BitNet b1.58), measured")
    lines.append("=" * 68)

    # -- B.4.1: quantization and the scale -------------------------------
    torch.manual_seed(0)
    w = torch.tensor([0.20, -0.05, 0.90, -1.10, 0.03, 0.44, -0.31, 0.72])
    w_q, gamma = ternary_quantize(w)
    lines.append("B.4.1 — quantization (gamma = mean|w| + eps)")
    lines.append(f"  gamma = {float(gamma):.6f}")
    lines.append("  w          w/gamma     round   trit")
    for i in range(w.numel()):
        scaled = float(w[i] / gamma)
        rounded = int(round(scaled))
        lines.append(
            f"  {float(w[i]):+.3f}    {scaled:+.4f}    {rounded:+d}    {int(w_q[i]):+d}"
        )
    eff = w_q * gamma
    lines.append(
        f"  effective weights w_q*gamma: {[round(float(v), 4) for v in eff]}"
    )
    lines.append(
        f"  E[w_q^2] over this tensor = {float((w_q ** 2).mean()):.4f} "
        f"(2/3 = 0.6667 in expectation)"
    )
    lines.append("")

    # -- B.4.2: the STE gradient identity --------------------------------
    torch.manual_seed(1)
    layer = TernaryLinear(4, 3, bias=False)
    x = torch.randn(2, 4)
    out = layer(x)
    out.sum().backward()
    expected = torch.ones_like(out).T @ x  # dL/dw_eff for L = sum(out)
    ste_ok = torch.allclose(layer.weight.grad, expected, atol=1e-6)
    lines.append("B.4.2 — straight-through gradient identity")
    lines.append(f"  parameter grad:      {[round(float(v), 4) for v in layer.weight.grad[0]]}")
    lines.append(f"  dL/dw_eff (expected): {[round(float(v), 4) for v in expected[0]]}")
    lines.append(f"  grad == dL/dw_eff exactly: {ste_ok}")

    # The map is locally constant: a tiny nudge does not move the trits,
    # so the *true* derivative is zero almost everywhere. STE substitutes
    # the identity there, which is what makes the layer trainable.
    w1 = layer.weight.detach().clone()
    q1, _ = ternary_quantize(w1)
    q2, _ = ternary_quantize(w1 + 1e-3)
    changed = int((q1 != q2).sum())
    lines.append(
        f"  trits changed under a +1e-3 nudge: {changed} of {q1.numel()} "
        f"(true gradient ~ 0 a.e.)"
    )
    lines.append(
        f"  STE parameter gradient norm: {float(layer.weight.grad.norm()):.4f} "
        f"(nonzero, usable)"
    )
    lines.append("")

    # -- B.4.3: packing five trits per byte ------------------------------
    lines.append("B.4.3 — five trits per byte (3^5 = 243 <= 255)")
    trits = torch.tensor([1, 0, -1, 1, 0, -1, -1, 1, 1, 1, 0, 0, -1])
    packed = pack_trits(trits)
    restored = unpack_trits(packed, trits.numel())
    lines.append(f"  trits   ({trits.numel()}): {trits.tolist()}")
    lines.append(f"  packed  ({packed.numel()} B): {packed.tolist()}")
    lines.append(f"  restored    : {restored.tolist()}")
    lines.append(
        f"  roundtrip exact: {bool(torch.equal(restored, trits))} "
        f"(13 trits -> {packed_bytes(13)} bytes = ceil(13/5))"
    )
    lines.append("")

    # -- the foundation footprint ----------------------------------------
    n = FOUNDATION_PARAMS
    nb = packed_bytes(n)
    kib = nb / 1024.0
    lines.append("Footprint — the GeoFormer foundation model")
    lines.append(f"  parameters           : {n:,}")
    lines.append(f"  packed bytes         : ceil({n:,}/5) = {nb:,} B")
    lines.append(f"  in KiB               : {kib:.2f} KiB")
    lines.append(f"  bits per parameter   : {packed_bits_per_parameter():.4f} "
                 f"(vs 32 for FP32, a {32 / packed_bits_per_parameter():.1f}x reduction)")
    lines.append("")

    # -- B.4.1 revisited: variance preservation --------------------------
    torch.manual_seed(2)
    big = torch.empty(256, 512)
    variance_preserving_init_(big, mode="ternary")
    bq, bg = ternary_quantize(big)
    w_eff = bq * bg
    e_q2 = float((bq ** 2).mean())
    var_eff = float(w_eff.var())
    target = (2.0 / 3.0) * float(bg ** 2)
    lines.append("B.4.1 — variance preservation (256x512 init, fan_in=512)")
    lines.append(f"  init std             : {float(big.std()):.6f}  "
                 f"(sqrt(1.5/fan_in) = {math.sqrt(1.5 / 512):.6f})")
    lines.append(f"  E[w_q^2] measured    : {e_q2:.4f}  "
                 f"(shape-set constant; idealized target 2/3 = 0.6667)")
    lines.append(f"  Var(w_eff) measured  : {var_eff:.6f}")
    lines.append(f"  (2/3) gamma^2 target : {target:.6f}   ratio {var_eff / target:.4f}")
    lines.append(
        f"  scale invariance     : quantize(10*w) trits == quantize(w) trits "
        f"-> {bool(torch.equal(ternary_quantize(big)[0], ternary_quantize(10.0 * big)[0]))}"
    )
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
