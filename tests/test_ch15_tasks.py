# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch15_tasks.py — Pin stdout of Chapter 15 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch15_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch15_tasks.task_1)
    assert "Residual Drowning Simulation (eta = 0.5):" in out
    assert "Step 0 Frontier Ratio: 1.0000" in out
    assert "Step 2 Frontier Ratio: 0.4082" in out
    assert "Monotonic Dilution Verified (O(1/t)): True" in out


def test_task_2():
    out = _capture(ch15_tasks.task_2)
    assert "Symmetrization Echo Audit (2-Step Return Probability P(0 -> 0)):" in out
    assert "Directed Graph P_dir(0 -> 0):     0.0000  (Strictly Zero ✔)" in out
    assert "Symmetrized Graph P_sym(0 -> 0):  0.5000  (Echo Reflection Trap ❌)" in out
    assert "Spurious 2-Cycle Echo Detected:   True" in out


def test_task_3():
    out = _capture(ch15_tasks.task_3)
    assert "Exponential Dispersion vs. Random Chance Floor (N = 64, d = 3.0):" in out
    assert "Random Chance Floor (1/N):         0.0156" in out
    assert "Analytical Crossing Depth k*:      4 hops" in out
    assert "Depth 3-Hop Mass: 0.0370  [ABOVE FLOOR ✔]" in out
    assert "Depth 4-Hop Mass: 0.0123  [BELOW FLOOR ❌ (Drowned in noise)]" in out


def test_task_4():
    out = _capture(ch15_tasks.task_4)
    assert "Eklavya Soliton Propagation (gamma = 2.5, alpha_0 = 2.5):" in out
    assert "Final Target Mass P(y=4): 1.0000" in out
    assert "Final Attractor Entropy:  0.0000" in out
    assert "Soliton Mass Conserved (P(y) >= 0.5): True" in out


def test_task_5():
    out = _capture(ch15_tasks.task_5)
    assert "Comparative Diagnostic (4 Hops + Branching Decoys):" in out
    assert "Linear Diffusion (gamma = 1.0): P(y=4) = 0.2603" in out
    assert "Eklavya Soliton  (gamma = 2.5): P(y=4) = 1.0000" in out
    assert "higher target concentration" in out


def test_task_6():
    out = _capture(ch15_tasks.task_6)
    assert "Step 1 Drowning Theoretical Bound rho_1 <= 0.3333" in out
    assert "Analytical Crossing Depth for N=1000, d=4: k* = 5 hops" in out
    assert "All Theoretical Invariant Bounds Verified: True" in out
