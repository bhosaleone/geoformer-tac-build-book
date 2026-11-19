# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch12_tasks.py — pin stdout of Chapter 12 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch12_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch12_tasks.task_1)
    assert "Crossroad Graph Built: 32 entities, 8 chains" in out
    assert "Dead-end count (all-zero rows): 0" in out
    assert "No-Oracle-Sink Invariant: PASSED (no artificial boundaries)" in out


def test_task_2():
    out = _capture(ch12_tasks.task_2)
    assert "Crossroad Node 0 Conductance Audit:" in out
    assert "True Highway (-> 1):  W = 1.0000" in out
    assert "Decoy Loops:         W = 0.4000" in out
    assert "Conductance Ratio α₀ = W_true / W_decoy: 2.50  (target: 2.5)" in out
    assert "Highway is 2.5x cheaper than decoy path" in out


def test_task_3():
    out = _capture(ch12_tasks.task_3)
    assert "1-Hop Training Fact Count: 80" in out
    assert "Depth 2-Hop Queries: 4 pairs | Overlap with Train: 0 (Disjoint ✔)" in out
    assert "Depth 3-Hop Queries: 4 pairs | Overlap with Train: 0 (Disjoint ✔)" in out
    assert "Depth 4-Hop Queries: 4 pairs | Overlap with Train: 0 (Disjoint ✔)" in out


def test_task_4():
    out = _capture(ch12_tasks.task_4)
    assert "Train DataLoader: 12 batches of size 8" in out
    assert "Batch Tensor Shapes: src = [8], rel = [8], tgt = [8]" in out
    assert "Test Loaders available for depths: [2, 4]" in out


def test_task_5():
    out = _capture(ch12_tasks.task_5)
    assert "Graph Laplacian Spectrum:" in out
    assert "λ_1 (Ground State):       0.000000  (identically zero ✔)" in out
    assert "(positive spectral gap ✔)" in out


def test_task_6():
    out = _capture(ch12_tasks.task_6)
    assert "SDK Parity Audit:" in out
    assert "Both datasets enforce un-truncated continuous topologies: True" in out
