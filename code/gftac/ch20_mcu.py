# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 20 — MCU Deployment: GeoFormer on an Arduino Uno.

The whole book has run on machines with floating-point units. This chapter
takes the smallest kernel it can honestly take — the discrete, row-normalized
directed diffusion ``h_{t+1} = h_t P`` of the ATmega328P firmware — and
reproduces it exactly in integer arithmetic. It then measures what the
truncation costs, packs the weights the way the firmware's ``mcu_weights.h``
packs them, exports a PROGMEM C header, accounts for Flash and SRAM, and runs
the firmware's own adversarial probes in simulation.

The reference firmware is ``firmware/uno_nano_geoformer/main.c``. Its fixed
point convention is ``FP_ONE = 255`` (so that ``1.0`` is the byte ``255``), and
its transition entry is

    P_hat[i][j] = floor(W[i][j] * 255 / row_sum(W[i]))

with the diffusion step

    h_next[j] = floor( ( sum_i h[i] * P_hat[i][j] ) / 255 ).

Book theorems proved in the chapter and pinned by the tests:

  B.20.1  Fixed-point representation error.  For ``x in [0, 1]`` the map
          ``Q(x) = round(255 x)`` and its inverse ``D(q) = q / 255`` satisfy
          ``|D(Q(x)) - x| <= 1/(2 * 255) = 1/510``.
  B.20.2  Accumulated diffusion error.  With the exact row-stochastic
          ``P = W / row_sum(W)`` and the truncated ``P_hat = floor(255 P)/255``,
          the k-step deviation obeys
          ``||h^(k) - h_hat^(k)||_inf <= k * ||h^(0)||_1 / 255``.
  B.20.3  Packing rates and the MCU footprint law.  Ternary weights pack at
          five trits per byte (``8/5 = 1.6`` bits/param) and 4-bit
          conductances at two codes per byte (4 bits/param); the Flash image
          of the 32-node model is ``N^2 + 2 N d + V + C`` bytes.

Everything here is **MEASURED** on this machine and pinned by
``tests/test_ch20_mcu.py`` and ``tests/test_ch20_tasks.py``. The firmware's
latencies and energies are **MODELED** from the published benchmark dossier;
no physical board was attached for this chapter.

Author: Atmabhan Pandit / Shrikant Bhosale (Eklavya Project)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FP_ONE",
    "FP_BITS",
    "ATmega328P_FLASH",
    "ATmega328P_SRAM",
    "MCUModel",
    "encode_fp",
    "decode_fp",
    "lattice_from_edges",
    "make_uno32_model",
    "make_small_model",
    "qtac_diffusion_step",
    "float_diffusion_step",
    "forward_inference",
    "float_forward_inference",
    "ternary_quantize",
    "quantize_4bit",
    "pack_trits",
    "unpack_trits",
    "pack_4bit",
    "unpack_4bit",
    "export_c_header",
    "parse_c_header",
    "flash_footprint_bytes",
    "sram_footprint_bytes",
    "inject_shortcut",
    "lfsr_perturb",
    "isolation_activation",
    "UNO32_TEST_2HOP",
    "UNO32_TEST_3HOP",
    "UNO32_ADV_PROBES",
]

# ----------------------------------------------------------------------
# Fixed-point constants (mirror main.c: 1.0 == 255)
# ----------------------------------------------------------------------

FP_ONE = 255
FP_BITS = 8

ATmega328P_FLASH = 32_768   # bytes
ATmega328P_SRAM = 2_048     # bytes


# ----------------------------------------------------------------------
# The exported UNO-32 model, transcribed from firmware/mcu_weights.h
# ----------------------------------------------------------------------

# Sparse metric conductance lattice W (32x32); each arc is (source, target, w).
_UNO32_LATTICE_EDGES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 10),
    (1, 2, 10), (1, 4, 10),
    (2, 3, 10),
    (4, 5, 10),
    (6, 7, 10),
    (7, 8, 10), (7, 10, 10),
    (8, 9, 10),
    (10, 11, 10),
    (12, 13, 10),
    (13, 14, 10),
    (14, 15, 10),
    (16, 17, 10),
    (17, 18, 10),
    (18, 19, 10),
    (20, 21, 10), (20, 24, 10),
    (21, 22, 10),
    (22, 23, 10),
    (24, 25, 10),
    (26, 27, 10),
    (27, 28, 10),
    (28, 29, 10),
)

# 1.58-bit ternary input projection W_IN (16x32), values in {-1, 0, +1}.
_UNO32_W_IN = """\
   1  1  0 -1  0 -1 -1 -1 -1  1  1 -1 -1  0 -1  1  1  0  0  0 -1  1  0  1  1  1  0  1  0  0  0  1
  -1 -1  0  1  0  0  1  0 -1  1 -1  0 -1  1 -1  0 -1 -1  1  0  0  0 -1 -1 -1  0  1  1  0  1 -1  1
   1  1  1  0  0  0 -1  1  1  0  0  0  1  0 -1 -1 -1  0 -1  0  1  1  0  0  1  0 -1  0 -1 -1  1  0
  -1  1  1  0  0  0  1 -1  0  1 -1  0  1  1  1  0 -1  1 -1  0  1  1 -1  0 -1  0 -1  1  1  0  1  0
   1  1 -1 -1 -1  1  0  1  1  0 -1  1 -1  1  0  1 -1  1  1  1  0 -1  0  0 -1  1  0  1 -1  0 -1  1
  -1 -1 -1  0  1 -1 -1 -1 -1  1  1  0  0  1  1  1 -1 -1  0  0 -1 -1 -1  0  0 -1 -1  0  1  0  1  1
   1 -1  0  0 -1  0  1  1  0  1  0 -1 -1  1  1  0 -1  0 -1  0 -1  1  1 -1  0  1 -1  1  1  0  1  1
  -1  1  1 -1  0  0  1  1  0 -1 -1 -1 -1  1 -1  0 -1  1 -1  0 -1 -1 -1  1  1 -1  1 -1  0  0  0  0
   0  1  0 -1 -1  0 -1  0 -1  0 -1  0  0  1 -1  0 -1  0 -1  0 -1 -1  1  0  1  1  1  1  1  1  0  0
   1 -1 -1  0 -1  0  0 -1  1  0  0  1  1  1  1  1 -1 -1  1 -1  0  0 -1 -1  0 -1 -1 -1  0  0  1 -1
   0  0  1 -1  1  0  1  1  0  1 -1 -1  0 -1 -1  0  1 -1  1  0  1 -1  0  0  0  1 -1  1  1 -1  0  0
  -1  0  1 -1 -1 -1 -1  1  1  1 -1  0 -1  0  1  1  0 -1  1  1  1  1  1  1  0 -1 -1  1 -1  1  0  0
   0  0 -1  0  1 -1  0  0 -1 -1  0  1  1  0  0  1  0  0 -1  0  1 -1 -1  0  1 -1  1  1 -1  0  0  1
   1 -1 -1  0 -1  0  0  0 -1 -1  0  0  1 -1 -1  1  0  1 -1  1  1  1  1  1  1 -1  1  0  1  1  1 -1
  -1  1 -1  0 -1 -1  0  1  1  0  0  0  1 -1 -1  1  1  0 -1  0 -1  1 -1 -1  1  0  1 -1  1  1  1 -1
   1  1  0  0 -1  0 -1  0 -1  0  0 -1  0 -1  0  0  0 -1  1  0 -1  0 -1  1 -1 -1  1  1  1  0  0  0
"""

# 1.58-bit ternary output projection W_OUT (32x16), values in {-1, 0, +1}.
_UNO32_W_OUT = """\
   0 -1  1 -1  1 -1  0 -1  0  1  0 -1 -1  1 -1  1
   1 -1  1  1  1 -1 -1  0  1 -1  0  0  0 -1  1  1
  -1  0  0  1 -1 -1  0  1  1 -1  1  1 -1 -1 -1  0
   1  1  0  0  0  0  0  0  0  0  1  0  0  0  0  0
   0  1  0 -1 -1  1 -1  1 -1 -1  1  0  1 -1 -1 -1
   0  1  0  0  0  0  0 -1  0  0  0  0  0  0  1  0
  -1  1 -1  1  0 -1  1  1 -1  1  1 -1  0  0  0  0
  -1  0  1 -1  1 -1  1  1  0 -1  1  1  0  1  1  0
  -1 -1  1  0  1 -1 -1  0 -1  1 -1  1 -1 -1  1 -1
   0  1  0  0  0  1  0  0  0  0  0  1  0  1  0  0
   1 -1  0 -1 -1  1  0 -1 -1  0 -1 -1 -1  0  0  0
   0  0  0  1  0  0 -1  1  0  0  1  0  0  1  1  0
  -1  0  1  1 -1  0 -1 -1  0  1  0 -1  1  1  1  1
   1  1  0  1  1  1  1  1  1  1 -1  0  0 -1 -1 -1
  -1 -1  0  1  0  1  0 -1 -1  1 -1  1  1 -1 -1 -1
   0  0 -1  0  0  0 -1  1  1  1  1  0  0  0  0  0
   1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1  1  0  0  1  0
   0 -1  0  1  1 -1  0  1  0 -1 -1 -1  0  1  0 -1
   0  1 -1 -1  0  0 -1 -1 -1  1  1  0 -1 -1 -1  1
   0  0  0  0  0  0  0  0  0  0  0  1  0  1  0  1
  -1  1  1  1  1 -1 -1 -1 -1  0  1  1  1  1 -1  0
   1  0  1  1  0  0  1 -1 -1 -1 -1  1 -1  1  1  1
   0 -1 -1 -1  0 -1  1 -1  1 -1  0  1 -1  1 -1  0
   0  0 -1 -1 -1 -1  0  0  0  0  0  1  0  0  0  0
   1 -1  1 -1 -1  0  1  1  1  1  0  0  1  1  1 -1
   1  0  0  0  1  0  0  0  0  0  1  0  0  1  1  0
   0  1 -1 -1  0 -1 -1  0  1 -1 -1  0  1  1  1  1
   1  1  0  1  1  1  1 -1  1 -1  1  1  1  1 -1  1
   0  0 -1  1 -1  0  1  0  1  0  1 -1 -1  1  1  1
   0  0  0 -1  0  0 -1  0  0  0  1 -1 -1  0  0 -1
   0  1  0  0  1  0 -1  0  0  0  0  0  0  1  0  0
   0  0  0  0  0  0  1  0  0  0 -1  0  1  1  0  1
"""

# Exported test vectors from mcu_weights.h (src, tgt).
UNO32_TEST_2HOP: Tuple[Tuple[int, int], ...] = (
    (0, 2), (1, 3), (6, 8), (7, 9), (12, 14), (16, 18), (20, 22), (20, 25), (26, 28),
)
UNO32_TEST_3HOP: Tuple[Tuple[int, int], ...] = (
    (0, 3), (6, 9), (12, 15), (16, 19), (20, 23), (26, 29),
)
UNO32_ADV_PROBES: Tuple[Tuple[int, int], ...] = (
    (0, 9), (0, 22), (20, 3), (26, 15),
)


def _parse_grid(text: str, rows: int, cols: int) -> List[List[int]]:
    """Parse a whitespace-separated integer grid of ``rows`` x ``cols``."""
    vals = [int(tok) for tok in text.split()]
    if len(vals) != rows * cols:
        raise ValueError(f"expected {rows * cols} integers, got {len(vals)}")
    return [vals[r * cols:(r + 1) * cols] for r in range(rows)]


# ----------------------------------------------------------------------
# The model container
# ----------------------------------------------------------------------

@dataclass
class MCUModel:
    """A deployable model: a conductance lattice plus two ternary projections."""

    num_nodes: int
    d_model: int
    lattice: List[List[int]]       # N x N uint8 conductances
    w_in: List[List[int]]          # D x N ternary
    w_out: List[List[int]]         # N x D ternary

    def copy_lattice(self) -> List[List[int]]:
        return [row[:] for row in self.lattice]


def lattice_from_edges(num_nodes: int, edges: Sequence[Tuple[int, int, int]]) -> List[List[int]]:
    """Build a dense ``N x N`` lattice from a list of ``(src, dst, w)`` arcs."""
    W = [[0] * num_nodes for _ in range(num_nodes)]
    for src, dst, w in edges:
        W[src][dst] = int(w)
    return W


def make_uno32_model() -> MCUModel:
    """The exact 32-node model exported to the ATmega328P firmware."""
    return MCUModel(
        num_nodes=32,
        d_model=16,
        lattice=lattice_from_edges(32, _UNO32_LATTICE_EDGES),
        w_in=_parse_grid(_UNO32_W_IN, 16, 32),
        w_out=_parse_grid(_UNO32_W_OUT, 32, 16),
    )


def make_small_model() -> MCUModel:
    """A 16-node clinical-triage model (the .ino default graph), D=8.

    The projections are a deterministic ternary pattern; the lattice is the
    hand-carved graph of ``uno_nano_geoformer.ino``.
    """
    edges = (
        (0, 1, 8), (1, 2, 8), (2, 3, 8),
        (1, 4, 4), (4, 5, 8),
        (6, 7, 8), (7, 8, 8), (8, 9, 8),
        (10, 11, 10), (11, 12, 10),
    )
    n, d = 16, 8
    return MCUModel(
        num_nodes=n,
        d_model=d,
        lattice=lattice_from_edges(n, edges),
        w_in=[[((dd * 7 + j * 3) % 3) - 1 for j in range(n)] for dd in range(d)],
        w_out=[[((i * 5 + dd * 2) % 3) - 1 for dd in range(d)] for i in range(n)],
    )


# ----------------------------------------------------------------------
# B.20.1 — fixed-point representation
# ----------------------------------------------------------------------

def encode_fp(x: float) -> int:
    """Encode ``x in [0, 1]`` as a byte on the ``FP_ONE = 255`` scale.

    Round-half-up to the nearest code, clamped to ``[0, 255]``.
    """
    return max(0, min(FP_ONE, int(math.floor(x * FP_ONE + 0.5))))


def decode_fp(q: int) -> float:
    """Decode a fixed-point code back to ``[0, 1]``: ``q / 255``."""
    return int(q) / FP_ONE


# ----------------------------------------------------------------------
# The discrete row-normalized diffusion kernel (mirror of main.c)
# ----------------------------------------------------------------------

def _row_sums(lattice: List[List[int]]) -> List[int]:
    return [sum(row) for row in lattice]


def qtac_diffusion_step(lattice: List[List[int]], h: Sequence[int]) -> List[int]:
    """One integer diffusion step ``h_next = h P_hat`` exactly as ``main.c``.

    ``P_hat[i][j] = (W[i][j] * 255) // row_sum(W[i])`` and
    ``h_next[j] = (sum_i h[i] * P_hat[i][j]) // 255``.
    """
    n = len(lattice)
    rsum = _row_sums(lattice)
    h_next = [0] * n
    for j in range(n):
        acc = 0
        for i in range(n):
            if h[i] > 0 and lattice[i][j] > 0:
                if rsum[i] > 0:
                    prob = (lattice[i][j] * FP_ONE) // rsum[i]
                    acc += h[i] * prob
        h_next[j] = acc // FP_ONE
    return h_next


def float_diffusion_step(lattice: List[List[int]], h: Sequence[float]) -> List[float]:
    """The float reference step ``h_next = h P`` with ``P[i][j] = W[i][j]/row_sum``."""
    n = len(lattice)
    rsum = _row_sums(lattice)
    h_next = [0.0] * n
    for i in range(n):
        if h[i] == 0.0 or rsum[i] <= 0:
            continue
        for j in range(n):
            if lattice[i][j]:
                h_next[j] += h[i] * lattice[i][j] / rsum[i]
    return h_next


def _project_initial(model: MCUModel, src: int) -> List[int]:
    """Ternary in/out projection + source pulse, mirroring ``forward_inference``."""
    h_latent = [model.w_in[d][src] for d in range(model.d_model)]
    h = [0] * model.num_nodes
    for n in range(model.num_nodes):
        s = sum(h_latent[d] * model.w_out[n][d] for d in range(model.d_model))
        h[n] = min(s, FP_ONE) if s > 0 else 0
    h[src] = FP_ONE
    return h


def _argmax_confidence(h: Sequence[int]) -> Tuple[int, int]:
    best, mx, total = 0, 0, 0
    for n in range(len(h)):
        total += h[n]
        if h[n] > mx:
            mx, best = h[n], n
    conf = (mx * 100) // total if total > 0 else 0
    return best, conf


def forward_inference(
    model: MCUModel, src: int, tau: int, lattice: Optional[List[List[int]]] = None
) -> Tuple[int, int, List[int]]:
    """The firmware's integer forward pass.

    Returns ``(best_node, confidence_pct, h)``. Pass ``lattice`` to run on a
    warped or noisy copy without mutating the model.
    """
    W = model.lattice if lattice is None else lattice
    h = _project_initial(model, src)
    for _ in range(tau):
        h = qtac_diffusion_step(W, h)
    best, conf = _argmax_confidence(h)
    return best, conf, h


def float_forward_inference(
    model: MCUModel, src: int, tau: int, lattice: Optional[List[List[int]]] = None
) -> Tuple[int, List[float]]:
    """The float reference pipeline on the same ``0..255`` scale as the integer one."""
    W = model.lattice if lattice is None else lattice
    h_latent = [model.w_in[d][src] for d in range(model.d_model)]
    h = [0.0] * model.num_nodes
    for n in range(model.num_nodes):
        s = sum(h_latent[d] * model.w_out[n][d] for d in range(model.d_model))
        h[n] = float(s) if s > 0 else 0.0
    h[src] = float(FP_ONE)
    for _ in range(tau):
        h = float_diffusion_step(W, h)
    best = max(range(len(h)), key=lambda n: h[n])
    return best, h


# ----------------------------------------------------------------------
# B.20.3 — quantization and packing
# ----------------------------------------------------------------------

def _round_half_away(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def ternary_quantize(values: Sequence[float], eps: float = 1e-8) -> Tuple[List[int], float]:
    """BitNet b1.58 quantization: ``w_q = clip(round(w / gamma), -1, 1)``.

    ``gamma = mean|w| + eps`` is the per-tensor scale. Returns
    ``(trits, gamma)``.
    """
    flat = list(values)
    gamma = sum(abs(v) for v in flat) / max(1, len(flat)) + eps
    trits = [max(-1, min(1, _round_half_away(v / gamma))) for v in flat]
    return trits, gamma


def quantize_4bit(values: Sequence[float], vmax: float = 1.0) -> List[int]:
    """Quantize non-negative values to 4-bit codes ``0..15`` scaled by ``vmax``."""
    if vmax <= 0:
        return [0] * len(values)
    return [max(0, min(15, _round_half_away(v / vmax * 15.0))) for v in values]


def pack_trits(trits: Sequence[int]) -> List[int]:
    """Pack trits in {-1, 0, +1} five to a byte, little-endian (Ch. 4 convention).

    Byte = ``sum_{k=0}^{4} (t_k + 1) 3^k``; since ``3^5 = 243 <= 255`` the map
    is injective. A trailing group is zero-padded.
    """
    digits = [int(t) + 1 for t in trits]
    if any(d < 0 or d > 2 for d in digits):
        raise ValueError("trits must lie in {-1, 0, +1}")
    out: List[int] = []
    for b in range((len(digits) + 4) // 5):
        byte = 0
        for k in range(5):
            idx = b * 5 + k
            d = digits[idx] if idx < len(digits) else 0
            byte += d * (3 ** k)
        out.append(byte)
    return out


def unpack_trits(packed: Sequence[int], n: int) -> List[int]:
    """Invert :func:`pack_trits`, returning exactly ``n`` trits."""
    out: List[int] = []
    for byte in packed:
        b = int(byte)
        for _ in range(5):
            out.append((b % 3) - 1)
            b //= 3
    return out[:n]


def pack_4bit(values: Sequence[int]) -> List[int]:
    """Pack 4-bit codes two to a byte: ``lo | (hi << 4)``."""
    vals = [int(v) & 0xF for v in values]
    out: List[int] = []
    for b in range((len(vals) + 1) // 2):
        lo = vals[2 * b]
        hi = vals[2 * b + 1] if 2 * b + 1 < len(vals) else 0
        out.append(lo | (hi << 4))
    return out


def unpack_4bit(packed: Sequence[int], n: int) -> List[int]:
    """Invert :func:`pack_4bit`, returning exactly ``n`` codes."""
    out: List[int] = []
    for byte in packed:
        b = int(byte) & 0xFF
        out.append(b & 0xF)
        out.append((b >> 4) & 0xF)
    return out[:n]


# ----------------------------------------------------------------------
# C-header (PROGMEM) export and parse
# ----------------------------------------------------------------------

def export_c_header(
    model: MCUModel,
    path: Optional[str] = None,
    guard: str = "MCU_WEIGHTS_H",
    program_name: str = "gftac.ch20_mcu",
) -> str:
    """Emit the model as a PROGMEM C header, mirroring ``mcu_weights.h``.

    Returns the header text; when ``path`` is given the text is also written.
    """
    n, d = model.num_nodes, model.d_model
    lines: List[str] = []
    lines.append("/*")
    lines.append(" * GeoFormer-MCU Pre-Trained Weights (ATmega328P)")
    lines.append(f" * Auto-generated by {program_name}.export_c_header")
    lines.append(" */")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <avr/pgmspace.h>")
    lines.append("")
    lines.append(f"#define MCU_NUM_NODES {n}")
    lines.append(f"#define MCU_D_MODEL {d}")
    lines.append("")
    lines.append(f"// 4-bit Metric Conductance Lattice W ({n}x{n})")
    lines.append(f"const uint8_t W_LATTICE[MCU_NUM_NODES][MCU_NUM_NODES] PROGMEM = {{")
    for i in range(n):
        row = ", ".join(str(int(model.lattice[i][j])) for j in range(n))
        lines.append(f"    {{ {row} }}, // Node {i}")
    lines.append("};")
    lines.append("")
    lines.append(f"// 1.58-bit Ternary Input Projection W_IN ({d}x{n})")
    lines.append(f"const int8_t W_IN[MCU_D_MODEL][MCU_NUM_NODES] PROGMEM = {{")
    for dd in range(d):
        row = ", ".join(str(int(model.w_in[dd][j])) for j in range(n))
        lines.append(f"    {{ {row} }}, // Latent dim {dd}")
    lines.append("};")
    lines.append("")
    lines.append(f"// 1.58-bit Ternary Output Projection W_OUT ({n}x{d})")
    lines.append(f"const int8_t W_OUT[MCU_NUM_NODES][MCU_D_MODEL] PROGMEM = {{")
    for i in range(n):
        row = ", ".join(str(int(model.w_out[i][dd])) for dd in range(d))
        lines.append(f"    {{ {row} }}, // Node {i}")
    lines.append("};")
    lines.append("")
    lines.append(f"#endif // {guard}")
    text = "\n".join(lines) + "\n"
    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def _parse_c_array(text: str, name: str) -> List[List[int]]:
    m = re.search(
        re.escape(name) + r"\[[^\]]*\]\[[^\]]*\]\s*PROGMEM\s*=\s*\{(.*?)\n\};",
        text,
        re.S,
    )
    if m is None:
        raise ValueError(f"array {name} not found in header")
    rows: List[List[int]] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        inner = line[line.index("{") + 1:line.rindex("}")]
        rows.append([int(x) for x in inner.split(",") if x.strip()])
    return rows


def parse_c_header(text: str) -> Dict[str, object]:
    """Parse a header emitted by :func:`export_c_header` back into arrays."""
    n = int(re.search(r"#define\s+MCU_NUM_NODES\s+(\d+)", text).group(1))
    d = int(re.search(r"#define\s+MCU_D_MODEL\s+(\d+)", text).group(1))
    return {
        "num_nodes": n,
        "d_model": d,
        "w_lattice": _parse_c_array(text, "W_LATTICE"),
        "w_in": _parse_c_array(text, "W_IN"),
        "w_out": _parse_c_array(text, "W_OUT"),
    }


# ----------------------------------------------------------------------
# Footprint accountant
# ----------------------------------------------------------------------

def flash_footprint_bytes(
    model: MCUModel, num_vectors: int = 0, program_bytes: int = 5760
) -> Dict[str, int]:
    """Flash (PROGMEM) byte accounting for the exported model.

    Weights are stored one byte per cell/trit (the firmware's layout); vectors
    are ``src``/``tgt`` byte pairs; ``program_bytes`` is the audited code size.
    """
    n, d = model.num_nodes, model.d_model
    comp = {
        "W_LATTICE": n * n,
        "W_IN": d * n,
        "W_OUT": n * d,
        "test_vectors": 2 * int(num_vectors),
        "program_code": int(program_bytes),
    }
    comp["total"] = sum(comp.values())
    return comp


def sram_footprint_bytes(model: MCUModel) -> Dict[str, int]:
    """SRAM byte accounting for the runtime working set (mirror of main.c)."""
    n, d = model.num_nodes, model.d_model
    comp = {
        "W_ram": n * n,
        "h_curr": n,
        "h_next": n,
        "h_latent": 2 * d,
    }
    comp["total"] = sum(comp.values())
    return comp


# ----------------------------------------------------------------------
# Adversarial probes (mirror of main.c's four stress tests)
# ----------------------------------------------------------------------

def inject_shortcut(lattice: List[List[int]], src: int, dst: int, weight: int) -> List[List[int]]:
    """Return a copy of ``lattice`` with a spurious arc ``src -> dst`` written."""
    W = [row[:] for row in lattice]
    W[src][dst] = int(weight)
    return W


def lfsr_perturb(lattice: List[List[int]], seed: int = 0xACE1) -> Tuple[List[List[int]], int]:
    """Reproduce the firmware's 16-bit Galois LFSR conductance perturbation.

    Iterates the lattice in row-major order; for each active cell, advances the
    LFSR (tap mask ``0xB400``) and, when the low two bits are zero, nudges the
    conductance by ``-1`` (or ``+1`` if it is already 1). Returns the perturbed
    lattice and the number of perturbed cells.
    """
    W = [row[:] for row in lattice]
    n = len(W)
    state = seed & 0xFFFF
    perturbed = 0
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                state = ((state >> 1) ^ (0xB400 if (state & 1) else 0)) & 0xFFFF
                if (state & 0x03) == 0:
                    if W[i][j] > 1:
                        W[i][j] -= 1
                    else:
                        W[i][j] += 1
                    perturbed += 1
    return W, perturbed


def isolation_activation(model: MCUModel, src: int, disallowed: int, tau: int = 3) -> int:
    """Run a query and return the activation of a structurally disconnected node."""
    _, _, h = forward_inference(model, src, tau)
    return h[disallowed]
