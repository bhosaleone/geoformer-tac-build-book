# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch18_tasks.py — Tests asserting each runnable task in Chapter 18 runs cleanly.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from gftac import ch18_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch18_tasks.task_1)
    assert "Dataset Scope at Scale:" in out
    assert "Theoretical Dense 3D Tensor:" in out
    assert "Sparse Conductance Store:" in out
    assert "Memory Compression Ratio:" in out


def test_task_2():
    out = _capture(ch18_tasks.task_2)
    assert "Ingested 50,000 edges across 5 relations" in out
    assert "Dense Allocation Equivalent:" in out
    assert "Sparse Physical RAM Allocated:" in out
    assert "Empirical Compression Factor:" in out
    assert "Invariant Battery Audit:        PASSED" in out


def test_task_3():
    out = _capture(ch18_tasks.task_3)
    assert "4-Hop Transitive Deduction Query: 42 -[0,1,2,3]-> ?" in out
    assert "Latency across 100,000 entities:" in out
    assert "Retrieved Top-1 Entity:          Node 99,999" in out
    assert "Exact Retrieval Verified:        True" in out


def test_task_4():
    out = _capture(ch18_tasks.task_4)
    assert "Dense vs. Sparse Numerical Equivalence" in out
    assert "Agreement within Machine Tolerance (< 1e-6):  True" in out


def test_task_5():
    out = _capture(ch18_tasks.task_5)
    assert "Autonomous Emergent Halting on 50,000-Entity Sparse Lattice:" in out
    assert "Autonomous Halting Step tau*:    5 hops" in out
    assert "Final Attractor Basin:           Node 105" in out
    assert "Correct Basin Captured:         True" in out


def test_task_6():
    out = _capture(ch18_tasks.task_6)
    assert "SDK Parity Audit (N=10,000):" in out
    assert "SDK Top Entity:                 Node 40" in out
    assert "Book Top Entity:                Node 40" in out
    assert "Full Parity Verified:           True" in out
