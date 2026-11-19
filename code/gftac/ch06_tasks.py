# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 6 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch06_tasks.py 3        # a single task
    python3 code/gftac/ch06_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch06_tasks.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

# Allow standalone execution: python3 code/gftac/ch06_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch06_swiglu import (
    BookRMSNorm,
    BookSwiGLU,
    silu_minimum,
    swiglu_packed_bytes,
    swiglu_param_count,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§6.2 — B.6.1: RMSNorm is a radial map; the direction never moves."""
    torch.manual_seed(0)
    x = torch.randn(1, 8)
    eps = 1e-6
    m = float(x.pow(2).mean())
    n_x = x / math.sqrt(m + eps)
    print(f"RMS(n(x)) = {float(n_x.pow(2).mean().sqrt()):.6f}")
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        n_cx = (c * x) / math.sqrt((c * x).pow(2).mean() + eps)
        cos = float(F.cosine_similarity(n_cx.flatten(), n_x.flatten(), dim=0))
        rho = float(n_cx.norm() / n_x.norm())
        rho_pred = math.sqrt((m + eps) / (m + eps / c ** 2))
        print(f"c = {c:>6.2f}  cos = {cos:.10f}  rho = {rho:.6f}  pred = {rho_pred:.6f}")


def task_2() -> None:
    """§6.3 — B.6.2: the SiLU gate and its global minimum."""
    z_star, g_min = silu_minimum()
    print(f"global minimum at z* = {z_star:.6f},  g(z*) = {g_min:.6f}")
    for z in (-4.0, -2.0, -1.2784645, -0.5, 0.0, 0.5, 2.0, 4.0):
        print(f"z = {z:>8.3f}  ->  g(z) = {float(F.silu(torch.tensor(z))):>10.6f}")


def task_3() -> None:
    """§6.3 — B.6.2: gate statistics (the suppression is measurable)."""
    torch.manual_seed(1)
    z = torch.randn(1_000_000)
    g = F.silu(z)
    print(f"E[g(z)]         = {float(g.mean()):.6f}")
    print(f"E[|g(z)|]       = {float(g.abs().mean()):.6f}")
    print(f"P(g(z) < 0)     = {float((g < 0).float().mean()):.6f}")
    print(f"P(|g(z)| < 0.1) = {float((g.abs() < 0.1).float().mean()):.6f}")


def task_4() -> None:
    """§6.4 — B.6.3: the parameter accounting of the 8/3 convention."""
    d_model = 64
    d_ff = int(8 / 3 * d_model)
    n = swiglu_param_count(d_model, d_ff)
    print(f"d_model = {d_model}, d_ff = {d_ff}")
    print(f"3 * d_model * d_ff      = {n:,}")
    print(f"8 * d_model^2 (ideal)   = {8 * d_model ** 2:,}")
    print(f"packed ternary footprint = ceil({n:,}/5) = {swiglu_packed_bytes(d_model):,} B")


def task_5() -> None:
    """§6.5 — parity of RMSNorm with the SDK."""
    from geoformer.models.block import RMSNorm as SDKRMSNorm

    torch.manual_seed(2)
    sdk = SDKRMSNorm(16)
    book = BookRMSNorm(16)
    book.load_state_dict(sdk.state_dict())
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        diff = float((book(x) - sdk(x)).abs().max())
    print(f"max |book - sdk| = {diff:.3e}")
    print(f"bit-exact parity: {diff == 0.0}")


def task_6() -> None:
    """§6.5 — parity of TernarySwiGLU with the SDK (both paths)."""
    from geoformer.layers.ffn import TernarySwiGLU as SDKFFN

    torch.manual_seed(3)
    x = torch.randn(2, 5, 16)
    # weight-only path
    sdk_wo = SDKFFN(d_model=16, mode="ternary")
    for lin in (sdk_wo.w_gate, sdk_wo.w_up, sdk_wo.w_down):
        lin.act_bits = None
    book_wo = BookSwiGLU(16, act_bits=None)
    book_wo.load_state_dict(sdk_wo.state_dict())
    # SDK default (8-bit activation) path
    sdk_8 = SDKFFN(d_model=16, mode="ternary")
    book_8 = BookSwiGLU(16, act_bits=8)
    book_8.load_state_dict(sdk_8.state_dict())
    with torch.no_grad():
        d_wo = float((book_wo(x) - sdk_wo(x)).abs().max())
        d_8 = float((book_8(x) - sdk_8(x)).abs().max())
    print(f"weight-only (act_bits=None): max |book - sdk| = {d_wo:.3e}")
    print(f"SDK default (8-bit activations): max |book - sdk| = {d_8:.3e}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 6, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
