# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 2 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch02_tasks.py 4        # a single task
    python3 code/gftac/ch02_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch02_tasks.py.
"""

from __future__ import annotations

import os
import statistics
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch02_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch02_metric_lattice import MetricLattice

__all__ = [
    "task_1", "task_2", "task_3", "task_4",
    "task_5", "task_6", "task_7",
]


def task_1() -> None:
    """§2.2 — one fact, one division: the metric appears."""
    lat = MetricLattice(num_entities=6)
    d0 = lat.resistance(0, 1)
    lat.observe_edge(0, 1, weight=lat.delta_w)
    d1 = lat.resistance(0, 1)
    print(f"before any fact: d_01 = {d0:.6e}   (reads 1/eps)")
    print(f"after one dw write: d_01 = {d1:.6f}")


def task_2() -> None:
    """§2.3 — the shadow buffer: sub-threshold writes are not lost."""
    lat = MetricLattice(num_entities=2)
    dw = lat.delta_w
    print(f"levels: {lat.levels}, dw = {dw:.4f}")
    for k in range(1, 6):
        lat.observe_edge(0, 1, weight=dw / 5)
        print(f"write {k} of dw/5: shadow = {lat.W_raw[0, 1].item():.4f}, "
              f"stored level = {lat.level(0, 1)}")


def task_3() -> None:
    """§2.3 — stochastic projection is unbiased: E[stored] == exact."""
    torch.manual_seed(0)
    levels = []
    for _ in range(1000):
        lat = MetricLattice(num_entities=2)
        lat.observe_edge(0, 1, weight=0.37 * lat.delta_w, stochastic=True)
        levels.append(lat.level(0, 1))
    print("1000 writes of 0.37*dw through the stochastic projector")
    print(f"mean stored level: {statistics.fmean(levels):.4f}   (exact: 0.3700)")


def task_4() -> None:
    """§2.4 — Theorem B.2.1's quantitative signature: the harmonic law."""
    lat = MetricLattice(num_entities=2)
    for _ in range(5):
        lat.observe_edge(0, 1, weight=lat.delta_w)
        k = lat.level(0, 1)
        print(f"level {k:>2}: d_01 = {lat.resistance(0, 1):.6f}   "
              f"(15/{k} = {15 / k:.6f})")


def task_5() -> None:
    """§2.5 — verify the pairwise identity (2.4) in your own terminal."""
    lat = MetricLattice(num_entities=6)
    lat.observe_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (0, 2)])
    torch.manual_seed(7)
    x = torch.randn(6)
    L = lat.laplacian()
    qf = (x @ L @ x).item()
    A = lat.W + lat.W.T
    pairwise = ((A * (x[:, None] - x[None, :]) ** 2).sum() / 2).item()
    print(f"x^T L x                    = {qf:.6f}")
    print(f"sum_(i<j) A_ij (x_i-x_j)^2 = {pairwise:.6f}")
    print(f"difference                  = {abs(qf - pairwise):.2e}")


def task_6() -> None:
    """§2.6 — Theorem B.2.3: the floor, its tightness, and its boundary."""
    uni = MetricLattice(num_entities=6)
    for i in range(6):
        for j in range(6):
            if i != j:
                uni.W[i, j] = uni.delta_w
    w_min = 2 * uni.delta_w
    print(f"uniform complete lattice: lambda_2 = {uni.fiedler_value():.6f}")
    print(f"the floor N*w_min          = {6 * w_min:.6f}   (tight: equal)")
    chain = MetricLattice(num_entities=5)
    chain.observe_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    print(f"chain world: lambda_2 = {chain.fiedler_value():.6f}   "
          f"(no floor claimed: sparse)")


def task_7() -> None:
    """§2.7 — dead ends are reported, not hidden."""
    lat = MetricLattice(num_entities=5)
    lat.observe_edges([(0, 1), (1, 2), (2, 3)])
    P, dead = lat.directed_kernel()
    print(f"dead ends at nodes: {dead.nonzero().flatten().tolist()}")
    print(f"live row 2: {[round(v, 3) for v in P[2].tolist()]}")
    print(f"dead row 4: {P[4].tolist()}")
    print(f"row sums: {[round(P[i].sum().item(), 6) for i in range(5)]}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 2, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
