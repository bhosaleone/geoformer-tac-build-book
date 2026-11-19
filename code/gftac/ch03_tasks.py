# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 3 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch03_tasks.py 4        # a single task
    python3 code/gftac/ch03_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch03_tasks.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch03_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch03_tac import (
    DoubleWellPotential,
    QuadraticPotential,
    TacRelaxer,
    predicted_steps,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§3.2 — the landscape: energy and gradient on the double well."""
    v = DoubleWellPotential()
    for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
        s = torch.tensor([x])
        print(f"s = {x:+.1f}:  F = {float(v.energy(s)):.4f}   "
              f"grad F = {float(v.grad(s)[0]):+.4f}")


def task_2() -> None:
    """§3.3 — B.3.1: energy never rises under the flow."""
    v = DoubleWellPotential()
    res = TacRelaxer(v, kappa=1.0).relax(
        0.6 * torch.ones(4), h=0.05, max_steps=2000, tol=1e-5
    )
    for k in (0, 1, 2, 5, 10, res.steps):
        print(f"step {k:>3}:  F = {res.energy_history[k]:.6f}")
    print(f"converged in {res.steps} steps; the energy column never rises")


def task_3() -> None:
    """§3.4 — B.3.2: predict the step count, then measure it."""
    q = QuadraticPotential.isotropic(16, curvature=1.0)
    relaxer = TacRelaxer(q, kappa=1.0)
    s0 = torch.ones(16) / math.sqrt(16.0)
    for tol in (1e-3, 1e-5, 1e-6):
        pred = predicted_steps(1.0, 1.0, 0.05, float(torch.linalg.vector_norm(s0)), tol)
        actual = relaxer.relax(s0, h=0.05, max_steps=5000, tol=tol).steps
        print(f"tol = {tol:.0e}:  predicted {pred:>3} steps, measured {actual:>3}")


def task_4() -> None:
    """§3.5 — B.3.3: the sign of the init chooses the basin."""
    v = DoubleWellPotential()
    for init in (0.01, -0.01, 0.0):
        res = TacRelaxer(v, kappa=1.0).relax(
            init * torch.ones(4), h=0.05, max_steps=2000, tol=1e-4
        )
        print(f"init {init:+.2f}*1  ->  final {res.final_state[0].item():+.5f}*1"
              f"   ({res.steps} steps)")


def task_5() -> None:
    """§3.6 — dimension independence: same step count for any N."""
    for n in (10, 100, 1000):
        q = QuadraticPotential.isotropic(n, curvature=1.0)
        res = TacRelaxer(q, kappa=1.0).relax(
            torch.ones(n) / math.sqrt(n), h=0.05, max_steps=5000, tol=1e-6
        )
        print(f"N = {n:>5}:  {res.steps} steps to the drift tolerance")


def task_6() -> None:
    """§3.7 — temperature: the harmonic well fluctuates as sqrt(T/lam)."""
    lam = 1.0
    for T in (0.02, 0.08):
        q = QuadraticPotential.isotropic(1, curvature=lam)
        r = TacRelaxer(q, kappa=1.0, temperature=T, seed=11)
        samples = []
        for _ in range(300):
            s = torch.zeros(1)
            for _ in range(250):
                s = r.step(s, 0.02)
            samples.append(float(s[0]))
        std = float(torch.tensor(samples).std())
        print(f"T = {T:.2f}:  measured std = {std:.4f}   "
              f"sqrt(T/lam) = {math.sqrt(T / lam):.4f}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 3, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
