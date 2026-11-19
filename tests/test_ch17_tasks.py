# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch17_tasks.py — Pin stdout of Chapter 17 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch17_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch17_tasks.task_1)
    assert "Lyapunov Free Energy F[z; q] along Reasoning Path (Goal = 4):" in out
    assert "Step 0 (Frontier at Node 0): Free Energy F = 3.1833" in out
    assert "Step 4 (Frontier at Node 4): Free Energy F = 0.4833" in out
    assert "Net Free Energy Contraction toward Goal: True" in out


def test_task_2():
    out = _capture(ch17_tasks.task_2)
    assert "Autonomous Halting on Continuing Graphs (Out-degree > 0 everywhere):" in out
    assert "Target Depth k =  4 hops -> Inferred Halting τ* =  4 steps [Match: True]" in out
    assert "Target Depth k =  8 hops -> Inferred Halting τ* =  8 steps [Match: True]" in out
    assert "Target Depth k = 12 hops -> Inferred Halting τ* = 12 steps [Match: True]" in out
    assert "Target Depth k = 16 hops -> Inferred Halting τ* = 16 steps [Match: True]" in out


def test_task_3():
    out = _capture(ch17_tasks.task_3)
    assert "Dual Halting Criterion Specifications:" in out
    assert "Lyapunov Drift Threshold ε_TAC:       1.0e-04" in out
    assert "Semantic Distance Threshold ε_q:      0.15" in out
    assert "Soliton Peak Confidence Threshold:    0.60" in out
    assert "Eliminates Artificial Sink Dependency: True ✔" in out


def test_task_4():
    out = _capture(ch17_tasks.task_4)
    assert "Certified 32-Hop Benchmark Protocol Ledger (5 Seeds):" in out
    assert "Eklavya Argmax Accuracy:    100.0% ± 0.0%" in out
    assert "Linear Diffusion Accuracy:  0.0% ± 0.0%" in out
    assert "Eklavya Target Mass P(y):   0.7928" in out
    assert "Eklavya Emergent Hops τ*:   [32, 32, 32, 32, 32] (all match 32 ✔)" in out


def test_task_5():
    out = _capture(ch17_tasks.task_5)
    assert "Statistical Hypothesis Testing (Eklavya vs Linear Diffusion at 16 Hops):" in out
    assert "Categorically Decisive (p < 1e-6): True" in out


def test_task_6():
    out = _capture(ch17_tasks.task_6)
    assert "SDK Parity Audit:" in out
    assert "Reference SDK module located: True" in out
    assert "Verified bit-exact ✔" in out
