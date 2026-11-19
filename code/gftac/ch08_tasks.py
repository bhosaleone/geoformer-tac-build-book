# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 8 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch08_tasks.py 3        # a single task
    python3 code/gftac/ch08_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch08_tasks.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch08_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch08_cayley import (
    BookS5Projector,
    burnside_sum,
    irrep_dimensions,
    irrep_names,
    kolmogorov_bounds,
    kolmogorov_capacities,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§8.2 — B.8.1: The Frobenius-Schur identity of S_5."""
    dims = irrep_dimensions()
    names = irrep_names()
    print("S_5 Irreducible Representation Accounting:")
    print(f"{'Irrep Name':<26}{'dim':>5}{'dim^2':>8}")
    for name, d in zip(names, dims):
        print(f"{name:<26}{d:>5}{d * d:>8}")
    b_sum = burnside_sum()
    dim_sum = sum(dims)
    print(f"Burnside sum: sum(d_i^2) = {b_sum}  (|S_5| = 5! = 120)")
    print(f"Invariant subspace dim: sum(d_i) = {dim_sum}")


def task_2() -> None:
    """§8.3 — B.8.2: Projector symmetry, idempotence, and rank."""
    torch.manual_seed(0)
    d_model = 32
    proj = BookS5Projector(d_model=d_model)
    with torch.no_grad():
        P = proj.projector_matrix()
        sym_err = float((P - P.t()).abs().max().detach())
        idem_err = float((P @ P - P).abs().max().detach())
        rank = int(torch.linalg.matrix_rank(P))
        trace = float(torch.trace(P))
    print(f"d_model = {d_model},  target invariant rank = {proj.total_irrep_dim}")
    print(f"Symmetry:    |P - P^T|  = {sym_err:.3e}")
    print(f"Idempotence: |P^2 - P|  = {idem_err:.3e}")
    print(f"Matrix rank(P)          = {rank}")
    print(f"Matrix trace(P)         = {trace:.4f}")


def task_3() -> None:
    """§8.3 — B.8.2: Pythogorean energy split and orthogonality."""
    torch.manual_seed(0)
    proj = BookS5Projector(d_model=32)
    h = torch.randn(2, 6, 32)
    with torch.no_grad():
        h_sym, _ = proj(h)
        h_perp = h - h_sym
        norm_orig = float((h ** 2).sum(-1).mean().detach())
        norm_sym = float((h_sym ** 2).sum(-1).mean().detach())
        norm_perp = float((h_perp ** 2).sum(-1).mean().detach())
        ortho_ip = float((h_sym * h_perp).sum(-1).abs().mean().detach())
    print(f"||h||^2                 = {norm_orig:.6f}")
    print(f"||Ph||^2 (symmetric)    = {norm_sym:.6f}")
    print(f"||(I-P)h||^2 (residual) = {norm_perp:.6f}")
    print(f"||Ph||^2 + ||(I-P)h||^2 = {norm_sym + norm_perp:.6f}")
    print(f"Inner product <Ph, h_perp> = {ortho_ip:.3e}")


def task_4() -> None:
    """§8.4 — B.8.3: Kolmogorov capacity accounting across irrep channels."""
    caps = kolmogorov_capacities()
    names = irrep_names()
    dims = irrep_dimensions()
    bounds = kolmogorov_bounds()
    print("Channel Capacity Accounting (d_rho * log2(d_rho)):")
    for name, d, c in zip(names, dims, caps):
        print(f"  {name:<26} dim={d}   capacity = {c:>7.4f} bits")
    print(f"Total multi-channel capacity K_total = {bounds['K_total']:.4f} bits")
    print(f"Max single-channel capacity          = {bounds['K_channel_max']:.4f} bits")


def task_5() -> None:
    """§8.4 — B.8.3: Empirical Shannon complexity K_S across input distributions."""
    torch.manual_seed(0)
    proj = BookS5Projector(d_model=32)
    # 1. Random gaussian noise input (high entropy)
    h_rand = torch.randn(4, 8, 32)
    # 2. Concentrated 1D directional input (low entropy)
    h_mono = torch.zeros(4, 8, 32)
    h_mono[..., 0] = 5.0
    with torch.no_grad():
        _, K_rand = proj(h_rand)
        _, K_mono = proj(h_mono)
    print(f"Theoretical maximum K_S bound (log2(120)) = {math.log2(120.0):.6f} bits")
    print(f"K_S for random Gaussian activation       = {float(K_rand):.6f} bits")
    print(f"K_S for single-mode activation           = {float(K_mono):.6f} bits")
    print(f"Random K_S <= 6.9069: {float(K_rand) <= 6.9069 + 1e-4}")
    print(f"Single-mode K_S < Random K_S: {float(K_mono) < float(K_rand)}")


def task_6() -> None:
    """§8.5 — Parity check against production SDK S5CayleyProjector."""
    from geoformer.layers.cayley import S5CayleyProjector as SDKProj

    torch.manual_seed(1)
    d_model = 32
    sdk = SDKProj(d_model=d_model)
    book = BookS5Projector(d_model=d_model)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, 6, d_model)
    with torch.no_grad():
        h_sdk, k_sdk = sdk(h)
        h_bk, k_bk = book(h)
    print(f"max |h_sym_book - h_sym_sdk| = {float((h_bk - h_sdk).abs().max()):.3e}")
    print(f"|K_S_book - K_S_sdk|         = {float((k_bk - k_sdk).abs()):.3e}")
    print(f"Bit-exact projection: {torch.equal(h_bk, h_sdk)}")
    print(f"Bit-exact complexity: {torch.equal(k_bk, k_sdk)}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 8, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
