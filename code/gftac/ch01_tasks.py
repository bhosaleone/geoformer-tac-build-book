# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 1 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch01_tasks.py 1        # a single task
    python3 code/gftac/ch01_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch01_tasks.py.
"""

from __future__ import annotations

import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch01_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch01_compositional_collapse import (
    OneHopMemorizer,
    build_chain_world,
    propagate_distribution,
    transition_matrix,
)

__all__ = ["task_1", "task_2", "task_3"]


def task_1() -> None:
    """§1.3 — the world: nine facts, honest dead ends."""
    n, edges, chains = build_chain_world(num_chains=3, chain_len=4)
    P = transition_matrix(n, edges)
    print(f"{n} entities, {len(edges)} one-hop facts, {len(chains)} chains")
    print(f"chain 0: {chains[0]}")
    print(f"P[0] = {P[0].tolist()}")
    dead = [i for i in range(n) if P[i].sum().item() == 0.0]
    print(f"dead ends (all-zero rows): nodes {dead}")


def task_2() -> None:
    """§1.3 — the geometric readout: composition by matrix power."""
    n, edges, chains = build_chain_world()
    P = transition_matrix(n, edges)
    start = chains[0][0]
    z = propagate_distribution(P, start, hops=2)
    target = chains[0][2]
    print(f"z = e_{start} P^2 = {[round(v, 3) for v in z.tolist()]}")
    print(f"argmax(z) = {int(z.argmax().item())}  (true two-hop target: {target})")


def task_3() -> None:
    """§1.3 — the flat model learns every fact, then echoes."""
    torch.manual_seed(0)
    n, edges, chains = build_chain_world()
    model = OneHopMemorizer(n, dim=32)
    hist = model.train_on_edges(edges, epochs=300, lr=0.05)
    for ep in (0, 149, 299):
        print(f"epoch {ep + 1:>3}: cross-entropy {hist[ep]:.4f}")
    acc = sum(model.predict(s) == t for (s, t) in edges) / len(edges)
    print(f"one-hop train accuracy: {acc:.2f}")
    a, b, c = chains[0][:3]
    pred = model.predict(a)
    print(f"zero-shot (start at {a}): model answers {pred}; "
          f"successor is {b}, true target is {c}")


TASKS = [task_1, task_2, task_3]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 1, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
