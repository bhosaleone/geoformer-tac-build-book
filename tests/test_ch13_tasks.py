# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch13_tasks.py — pin stdout of Chapter 13 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch13_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch13_tasks.task_1)
    assert "Dual Path Kramers Step:" in out
    assert "Initial State x_0:     0.2000" in out
    assert "Gradient dV/dx:        -0.8000" in out
    assert "stepped toward global basin x*=1.0 ✔" in out


def test_task_2():
    out = _capture(ch13_tasks.task_2)
    assert "Kramers Thermal Diffusivity D(t) Modulation:" in out
    assert "Calendar r_t = -1.0: D(t) = 0.000050" in out
    assert "Calendar r_t =  0.0: D(t) = 0.000100" in out
    assert "Calendar r_t =  1.0: D(t) = 0.000150" in out


def test_task_3():
    out = _capture(ch13_tasks.task_3)
    assert "Two-Timescale Epoch Step Executed:" in out
    assert "Train Loss:" in out
    assert "Dirichlet Energy Loss:" in out
    assert "Lattice W Non-Zero Entries:" in out
    assert "Temporal causality preserved (g_t evaluated on pre-update lattice): True" in out


def test_task_4():
    out = _capture(ch13_tasks.task_4)
    assert "Multi-Epoch Training Ledger:" in out
    assert "Epoch 1: Loss =" in out
    assert "Epoch 2: Loss =" in out
    assert "Epoch 3: Loss =" in out
    assert "Net Loss Contraction: True" in out


def test_task_5():
    out = _capture(ch13_tasks.task_5)
    assert "Training Determinism Audit (Seed 42):" in out
    assert "Run 1 Losses: [2.5882, 2.6544]" in out
    assert "Run 2 Losses: [2.5882, 2.6544]" in out
    assert "Max Absolute Discrepancy: 0.000000e+00  (Bit-Exact ✔)" in out


def test_task_6():
    out = _capture(ch13_tasks.task_6)
    assert "SDK Parity Audit:" in out
    assert "SDK Param:  1.498750" in out
    assert "Book Param: 1.498750" in out
    assert "Bit-exact optimizer trajectory agreement: True" in out
