# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch14_tasks.py — Pin stdout of Chapter 14 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch14_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch14_tasks.task_1)
    assert "Retrieval Metrics Audit:" in out
    assert "Hits@1:  25.0%  (Target: 25.0%)" in out
    assert "Hits@3:  75.0%  (Target: 75.0%)" in out
    assert "Hits@5:  100.0%  (Target: 100.0%)" in out
    assert "MRR:     0.5208" in out
    assert "Hits@k Monotonicity Preserved: True" in out


def test_task_2():
    out = _capture(ch14_tasks.task_2)
    assert "Attractor State Diagnostics:" in out
    assert "Peaked State:  Modal Mass P(y) = 0.9500" in out
    assert "Uniform State: Modal Mass P(y) = 0.2000" in out
    assert "Sharp Soliton Signature Verified: True" in out


def test_task_3():
    out = _capture(ch14_tasks.task_3)
    assert "Simpson Quadrature Precision Audit (df = 8):" in out
    assert "Precision Criterion (< 1e-12): True" in out


def test_task_4():
    out = _capture(ch14_tasks.task_4)
    assert "Statistical Hypothesis Testing (ILLUSTRATIVE supplied values, NOT a measurement):" in out
    assert "GeoFormer+TAC Mean Hits@1: 96.90%" in out
    assert "Flat Baseline Mean Hits@1: 12.70%" in out
    assert "Statistically Decisive (p < 0.001): True" in out
    assert "Provenance: values hardcoded in ch14_tasks.py; statistic only." in out


def test_task_5():
    out = _capture(ch14_tasks.task_5)
    assert "Zero-Shot Disjointness Audit:" in out
    assert "Depth 2-hop Overlaps with 1-hop Train: 0" in out
    assert "Depth 3-hop Overlaps with 1-hop Train: 0" in out
    assert "Depth 4-hop Overlaps with 1-hop Train: 0" in out
    assert "Zero-Shot Disjointness Invariant: True ✔" in out


def test_task_6():
    out = _capture(ch14_tasks.task_6)
    assert "Emergent Attractor Halting:" in out
    assert "Emergent Halting Depth τ*: 7 steps (tol = 1e-4)" in out
    assert "Halting strictly terminates within finite schedule: True" in out
