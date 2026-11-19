# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 7 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch07_tasks.py 3        # a single task
    python3 code/gftac/ch07_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch07_tasks.py.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch07_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch07_qtac import (
    BookQuantizedTAC,
    chain_laplacian,
    chebyshev_schedule,
    relax_variable,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]

_D_MODEL = 32
_N = 6


def _system(seed: int = 0):
    torch.manual_seed(seed)
    L = chain_laplacian(_N)
    h_init = torch.randn(1, _N, _D_MODEL)
    tac = BookQuantizedTAC(d_model=_D_MODEL, tau_max=200, delta_tau=0.1)
    with torch.no_grad():
        A = tac.system_matrix(L)
    return tac, L, h_init, A


def _step(h, h_init, L, tac, dt=None):
    dt = tac.delta_tau if dt is None else dt
    h_sym, _ = tac.s5_projector(h)
    dh = -((h - h_init) + tac.lambda_dir * torch.matmul(L.unsqueeze(0), h)
           + tac.lambda_s5 * (h - h_sym))
    return (h + dt * dh).detach()


def task_1() -> None:
    """§7.2 — B.7.1: the free energy descends, and the step-size condition."""
    tac, L, h_init, A = _system(0)
    eig = torch.linalg.eigvalsh(A)
    lam_max = float(eig.max())
    print(f"lambda_max(A) = {lam_max:.6f},  stability threshold 2/lambda_max = {2.0 / lam_max:.6f}")
    h = h_init.clone()
    for step in range(1, 41):
        if step in (1, 5, 10, 20, 40):
            print(f"tau = {step:>3}   F(h_tau) = {tac.free_energy(h, h_init, L):.10f}")
        h = _step(h, h_init, L, tac)
    # exact discrete step-size condition at h_0
    B = h_init.shape[0]
    g = (h_init.reshape(B, -1) @ A - h_init.reshape(B, -1))
    dt_star = 2.0 * float((g * g).sum()) / float((g @ A * g).sum())
    F0 = tac.free_energy(h_init, h_init, L)
    F_below = tac.free_energy(_step(h_init, h_init, L, tac, dt=0.99 * dt_star), h_init, L)
    F_above = tac.free_energy(_step(h_init, h_init, L, tac, dt=1.01 * dt_star), h_init, L)
    print(f"exact threshold dt* = 2||g||^2/(g^T A g) = {dt_star:.6f}")
    print(f"F(h_0) = {F0:.10f}")
    print(f"F at 0.99*dt* = {F_below:.10f}  (decrease: {F_below < F0})")
    print(f"F at 1.01*dt* = {F_above:.10f}  (increase: {F_above > F0})")


def task_2() -> None:
    """§7.3 — B.7.2: measured error against the Gronwall bound."""
    tac, L, h_init, A = _system(0)
    B = h_init.shape[0]
    eig = torch.linalg.eigvalsh(A)
    lam_min = float(eig.min())
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    e0 = float((h_init - h_star).norm())
    print(f"lambda_min(A) = {lam_min:.6f},  delta_tau = {tac.delta_tau}")
    print(f"||e_0|| = {e0:.6f}")
    print(f"{'tau':>5}{'||e_tau|| measured':>22}{'bound (1-dt*lmin)^tau':>24}")
    h = h_init.clone()
    for step in range(1, 41):
        h = _step(h, h_init, L, tac)
        if step in (1, 5, 10, 20, 40):
            err = float((h - h_star).norm())
            bound = (1 - tac.delta_tau * lam_min) ** step * e0
            print(f"{step:>5}{err:>22.8f}{bound:>24.8f}")


def task_3() -> None:
    """§7.3 — B.7.2: the loop halts inside its budget."""
    tac, L, h_init, A = _system(0)
    B = h_init.shape[0]
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    with torch.no_grad():
        out, tau, K_S = tac(h_init, L)
    print(f"schedule budget (tau_max) = {tac.tau_max}")
    print(f"actual_tau (halting step) = {tau}")
    print(f"halting step <= budget:   {tau <= tac.tau_max}")
    print(f"||h_tau - h*|| at halt    = {float((out - h_star).norm()):.3e}")
    print(f"final K_S (S5 readout)    = {float(K_S):.6f}")


def task_4() -> None:
    """§7.4 — B.7.3: the output depends on the multiset, not the order."""
    tac, L, h_init, _ = _system(0)
    base = [0.05, 0.1, 0.2, 0.4, 0.05, 0.1]
    perm = [0.4, 0.05, 0.1, 0.05, 0.2, 0.1]
    proj64 = copy.deepcopy(tac.s5_projector).double()
    with torch.no_grad():
        a32 = relax_variable(h_init, L, base, s5_projector=tac.s5_projector)
        b32 = relax_variable(h_init, L, perm, s5_projector=tac.s5_projector)
        a64 = relax_variable(h_init.double(), L.double(), base, s5_projector=proj64)
        b64 = relax_variable(h_init.double(), L.double(), perm, s5_projector=proj64)
    print(f"schedule          : {base}")
    print(f"permuted          : {perm}")
    print(f"max |out - out_perm|  float32 = {float((a32 - b32).abs().max()):.3e}")
    print(f"max |out - out_perm|  float64 = {float((a64 - b64).abs().max()):.3e}")


def task_5() -> None:
    """§7.4 — B.7.3: Chebyshev steps beat the best fixed step."""
    tac, L, h_init, A = _system(0)
    eig = torch.linalg.eigvalsh(A)
    lmin, lmax = float(eig.min()), float(eig.max())
    c, r = 0.5 * (lmin + lmax), 0.5 * (lmax - lmin)
    print(f"spectrum [{lmin:.6f}, {lmax:.6f}],  kappa = {lmax / lmin:.4f}")
    print(f"{'k':>4}{'best fixed':>16}{'chebyshev':>16}{'speed-up':>12}")
    for k in (1, 2, 3, 4, 6):
        fixed = ((lmax - lmin) / (lmax + lmin)) ** k
        cheb = 1.0 / math.cosh(k * math.acosh(c / r))
        print(f"{k:>4}{fixed:>16.8f}{cheb:>16.8f}{fixed / cheb:>11.1f}x")
    ds = chebyshev_schedule(4, lmin, lmax)
    print("chebyshev k=4 step sizes: " + ", ".join(f"{d:.4f}" for d in ds))


def task_6() -> None:
    """§7.5 — parity with the SDK QuantizedTAC."""
    from geoformer.layers.tac import QuantizedTAC as SDKTAC

    L = chain_laplacian(_N)
    torch.manual_seed(1)
    sdk = SDKTAC(d_model=_D_MODEL, tau_max=8, delta_tau=0.1, lambda_dir=0.05, lambda_s5=0.05)
    book = BookQuantizedTAC(d_model=_D_MODEL, tau_max=8, delta_tau=0.1,
                            lambda_dir=0.05, lambda_s5=0.05)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, _N, _D_MODEL)
    with torch.no_grad():
        h_sdk, tau_sdk, k_sdk = sdk(h, L)
        h_bk, tau_bk, k_bk = book(h, L)
    print(f"max |h_book - h_sdk| = {float((h_bk - h_sdk).abs().max()):.3e}")
    print(f"actual_tau: book {tau_bk}, sdk {tau_sdk}   (equal: {tau_bk == tau_sdk})")
    print(f"|K_S book - K_S sdk| = {float((k_bk - k_sdk).abs()):.3e}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 7, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
