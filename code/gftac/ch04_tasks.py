# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 4 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch04_tasks.py 3        # a single task
    python3 code/gftac/ch04_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch04_tasks.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch04_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch04_ternary import (
    FOUNDATION_PARAMS,
    TernaryLinear,
    pack_trits,
    packed_bits_per_parameter,
    packed_bytes,
    ternary_quantize,
    unpack_trits,
    variance_preserving_init_,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§4.2 — quantization: one scale, then round and clip to a trit."""
    w = torch.tensor([0.20, -0.05, 0.90, -1.10, 0.03, 0.44, -0.31, 0.72])
    w_q, gamma = ternary_quantize(w)
    print(f"gamma = mean|w| = {float(gamma):.6f}")
    for i in range(w.numel()):
        print(f"w = {float(w[i]):+.3f}  ->  w/gamma = {float(w[i] / gamma):+.4f}"
              f"  ->  trit {int(w_q[i]):+d}")


def task_2() -> None:
    """§4.3 — B.4.1: rescaling the weights leaves the trits untouched."""
    torch.manual_seed(0)
    w = torch.randn(4, 4)
    q1, g1 = ternary_quantize(w)
    q2, g2 = ternary_quantize(1000.0 * w)
    t1 = [int(v) for v in q1.flatten().tolist()]
    t2 = [int(v) for v in q2.flatten().tolist()]
    print(f"quantize(w)      trits: {t1}")
    print(f"quantize(1000*w) trits: {t2}")
    print(f"trit patterns identical: {torch.equal(q1, q2)}")
    print(f"gamma scaled by 1000: {float(g2):.4f} vs {1000.0 * float(g1):.4f}")


def task_3() -> None:
    """§4.4 — B.4.2: the STE gradient equals the de-quantized gradient."""
    torch.manual_seed(1)
    layer = TernaryLinear(4, 3, bias=False)
    x = torch.randn(2, 4)
    out = layer(x)
    out.sum().backward()
    expected = torch.ones_like(out).T @ x
    print(f"parameter gradient: {[round(float(v), 4) for v in layer.weight.grad[0]]}")
    print(f"dL/dw_eff expected: {[round(float(v), 4) for v in expected[0]]}")
    print(f"STE identity holds: {torch.allclose(layer.weight.grad, expected, atol=1e-6)}")
    print(f"gradient norm (nonzero, usable): {float(layer.weight.grad.norm()):.4f}")


def task_4() -> None:
    """§4.5 — B.4.3: five trits per byte, lossless."""
    trits = torch.tensor([1, 0, -1, 1, 0, -1, -1, 1, 1, 1, 0, 0, -1])
    packed = pack_trits(trits)
    restored = unpack_trits(packed, trits.numel())
    print(f"trits ({trits.numel()}): {trits.tolist()}")
    print(f"packed ({packed.numel()} bytes): {packed.tolist()}")
    print(f"restored: {restored.tolist()}")
    print(f"lossless: {torch.equal(restored, trits)}  "
          f"(ceil(13/5) = {packed_bytes(13)} bytes)")


def task_5() -> None:
    """§4.6 — the footprint of the foundation model."""
    n = FOUNDATION_PARAMS
    print(f"foundation parameters: {n:,}")
    print(f"packed bytes: ceil({n:,}/5) = {packed_bytes(n):,} B")
    print(f"footprint: {packed_bytes(n) / 1024:.2f} KiB  "
          f"(FP32 would need {n * 4 / 1024:.1f} KiB)")
    print(f"bits/param: {packed_bits_per_parameter():.4f}  "
          f"(information floor log2(3) = {math.log2(3):.4f})")


def task_6() -> None:
    """§4.3 — B.4.1: variance-preserving initialization."""
    torch.manual_seed(2)
    w = torch.empty(256, 512)
    variance_preserving_init_(w, mode="ternary")
    w_q, gamma = ternary_quantize(w)
    print(f"init std = {float(w.std()):.6f}   "
          f"target sqrt(1.5/fan_in) = {math.sqrt(1.5 / 512):.6f}")
    print(f"E[w_q^2] = {float((w_q ** 2).mean()):.4f}")
    print(f"Var(w_eff) = {float((w_q * gamma).var()):.6f}   "
          f"(2/3)gamma^2 = {(2.0 / 3.0) * float(gamma ** 2):.6f}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 4, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
