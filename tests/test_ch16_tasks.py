# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch16_tasks.py — Pin stdout of Chapter 16 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch16_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch16_tasks.task_1)
    assert "Non-Linear Soliton Power Sharpening:" in out
    assert "Peak Amplitude Boost:" in out
    assert "0.8797" in out


def test_task_2():
    out = _capture(ch16_tasks.task_2)
    assert "Critical Exponent & Contraction Calculus (alpha_0 = 2.5, d_avg = 2.0):" in out
    assert "Critical Exponent gamma_crit:  0.7565" in out
    assert "rho = 0.2024  (rho < 1.0 ✔ Soliton regime)" in out
    assert "rho = 1.2649  (rho >= 1.0 ❌ Dispersive regime)" in out


def test_task_3():
    out = _capture(ch16_tasks.task_3)
    assert "32-Hop Soliton Preservation Ledger (alpha_0 = 2.5, gamma = 2.5):" in out
    assert "Hop 1 Peak Mass:  0.8317  (Bound: 0.7976)" in out
    assert "Hop 32 Peak Mass: 0.8124  (Bound: 0.7976)" in out
    assert "Theorem Q-3 Lower Bound Satisfied across all 32 hops: True" in out
    assert "Wavefront Amplitude Omega(1) Preserved: True" in out


def test_task_4():
    out = _capture(ch16_tasks.task_4)
    assert "Entropy-Adaptive Sharpening Schedule (gamma_0 = 2.0, beta = 0.5, max = 3.5):" in out
    assert "Entropy H = 0.10 nats -> Adaptive Exponent gamma(H) = 2.0500" in out
    assert "Automatic Sharpening Intensification with Entropy: True" in out


def test_task_5():
    out = _capture(ch16_tasks.task_5)
    assert "Falsification Boundary Diagnostic (gamma = 0.5 < gamma_crit):" in out
    assert "Contraction Ratio rho: 1.2649  (rho >= 1.0)" in out
    assert "Hop 8 Peak Mass:       0.0612" in out
    assert "Dispersive Leakage Confirmed: True" in out
    assert "Falsification boundary correctly predicted by Theorem B.16.1: True" in out


def test_task_6():
    out = _capture(ch16_tasks.task_6)
    assert "Ensemble Lower Bound Invariant Checks:" in out
    assert "Bound(32)=0.6250 [Valid: True]" in out
    assert "Bound(32)=0.8717 [Valid: True]" in out
    assert "Bound(32)=0.8125 [Valid: True]" in out
    assert "All Ensemble Lower Bounds Verified: True" in out
