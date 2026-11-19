# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch10_tasks.py — pin stdout of Chapter 10 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch10_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch10_tasks.task_1)
    assert "Batched update: 4 pairs submitted, 3 edges modified" in out
    assert "W[0, 1] = 0.0667  (level: 1)" in out
    assert "Self-loop W[1, 1] = 0.0000 (hollow diagonal invariant preserved)" in out


def test_task_2():
    out = _capture(ch10_tasks.task_2)
    assert "Lattice delta_w = 0.0667 (15 levels)" in out
    assert "Step 1: raw = 0.0167" in out
    assert "level 0 (W = 0.0000)" in out
    assert "Step 5: raw = 0.0833" in out
    assert "level 1 (W = 0.0667)" in out


def test_task_3():
    out = _capture(ch10_tasks.task_3)
    assert "Invariant Battery Audit after 1000 randomized batched updates:" in out
    assert "(1) Non-negativity (W >= 0):               PASSED" in out
    assert "(2) Hollow diagonal (W_ii == 0):           PASSED" in out
    assert "(3) Bounded upper range (W <= w_max):      PASSED" in out
    assert "(4) Discrete quantized levels:             PASSED" in out
    assert "(5) Finite Frobenius energy:               PASSED" in out


def test_task_4():
    out = _capture(ch10_tasks.task_4)
    assert "Initial resistance d_01: 1.00e+08" in out
    assert "Level  1: W = 0.0667 | d_01 = 15.0000" in out
    assert "Level 15: W = 1.0000 | d_01 =  1.0000" in out


def test_task_5():
    out = _capture(ch10_tasks.task_5)
    assert "Before consolidation: W[0, 2] = 0.0000 (d_02 = 1.00e+08)" in out
    assert "[2-hop shortcut synthesized]" in out
    assert "[3-hop shortcut synthesized]" in out
    assert "Frobenius energy: before = 0.960000, after = 1.080000" in out


def test_task_6():
    out = _capture(ch10_tasks.task_6)
    assert "SDK Parity Audit:" in out
    assert "SDK Invariants:   Non-negative ✔ | Hollow diagonal ✔ | Energy finite ✔" in out
    assert "Book Invariants:  Non-negative ✔ | Hollow diagonal ✔ | Energy finite ✔" in out
    assert "Both engines synthesize multi-hop transitive shortcuts with bounded energy: True" in out
