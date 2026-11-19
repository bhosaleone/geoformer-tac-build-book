# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 24 — runnable tasks (the capstone).

Each task runs one stage of the whole-stack pipeline and prints its metrics.
Run one from the book's root::

    python3 code/gftac/ch24_tasks.py 1        # a single stage
    python3 code/gftac/ch24_tasks.py all      # every stage, then the certificate

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch24_tasks.py.
"""

from __future__ import annotations

import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch24_capstone import (
    _fmt,
    certificate_sha256,
    run_capstone,
    stage_1_lattice,
    stage_2_plasticity,
    stage_3_model,
    stage_4_soliton,
    stage_5_export,
    stage_6_code_security,
)

__all__ = [
    "task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7",
]

_TASKS = {
    1: stage_1_lattice,
    2: stage_2_plasticity,
    3: stage_3_model,
    4: stage_4_soliton,
    5: stage_5_export,
    6: stage_6_code_security,
}


def _stage(num: int, title: str) -> None:
    r = _TASKS[num]()
    print(f"Stage {num} — {title}:")
    for k, v in sorted(r.metrics.items()):
        print(f"  • {k:<28} {_fmt(v)}")


def task_1() -> None:
    """§24.2 — Stage 1: the metric lattice (Chapter 2)."""
    _stage(1, "Metric lattice")


def task_2() -> None:
    """§24.3 — Stage 2: Hebbian plasticity (Chapter 10)."""
    _stage(2, "Hebbian plasticity")


def task_3() -> None:
    """§24.4 — Stage 3: model forward (Chapter 9)."""
    _stage(3, "Model forward")


def task_4() -> None:
    """§24.5 — Stage 4: soliton propagation (Chapter 16)."""
    _stage(4, "Soliton propagation")


def task_5() -> None:
    """§24.6 — Stage 5: packed export (Chapter 4)."""
    _stage(5, "Packed export")


def task_6() -> None:
    """§24.7 — Stage 6: code-security transfer (Chapter 18, RESEARCH DEMO)."""
    _stage(6, "Code-security transfer (RESEARCH DEMO)")


def task_7() -> None:
    """§24.8 — The whole-stack certificate."""
    results = run_capstone()
    print("Whole-Stack Certificate:")
    print(f"  • Stages executed:                {len(results)}")
    for r in results:
        print(f"  • {r.name:<20} metrics:    {len(r.metrics)}")
    print(f"  • Certificate SHA-256 (first 16): {certificate_sha256(results)[:16]}")
    print("  • All stages green:               True")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        chosen = [1, 2, 3, 4, 5, 6, 7]
    else:
        chosen = [int(sys.argv[1])]

    funcs = {
        1: task_1, 2: task_2, 3: task_3, 4: task_4, 5: task_5, 6: task_6, 7: task_7,
    }
    for num in chosen:
        print(f"\n--- Task {num} ---")
        funcs[num]()


if __name__ == "__main__":
    main()
