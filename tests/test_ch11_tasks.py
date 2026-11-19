# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch11_tasks.py — pin stdout of Chapter 11 tasks against exact book expectations.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from gftac import ch11_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch11_tasks.task_1)
    assert "BookGeoFormerTACModel: 2 blocks, d_model = 32, 16 entities" in out
    assert "Logits shape: [4, 16] | Feature shape: [4, 32]" in out
    assert "Total Parameters: 29152 (5.69 KB packed at 1.58 bits/param)" in out
    assert "Emergent hops per query: [1, 1, 1, 1]" in out


def test_task_2():
    out = _capture(ch11_tasks.task_2)
    assert "Thermodynamic Loss Suite Breakdown:" in out
    assert "Primary Task CE Loss:     0.099289" in out
    assert "Metric Dirichlet Energy:  0.500000" in out
    assert "Ternary Regularization:   0.150761" in out
    assert "Composite Total Loss:     0.124304" in out


def test_task_3():
    out = _capture(ch11_tasks.task_3)
    assert "Backward Gradient Flow Audit:" in out
    assert "Embedding gradient norm:       0.786081  (non-vanishing ✔)" in out
    assert "Block 0 value proj grad:       3.164018  (passed through stack ✔)" in out
    assert "Block 0 SwiGLU gate grad:      1.182947  (unbroken residual highway ✔)" in out
    assert "Final classifier grad norm:    5.516822  (clean supervision ✔)" in out


def test_task_4():
    out = _capture(ch11_tasks.task_4)
    assert "Initial lattice W[0, 1] = 0.0000" in out
    assert "After plastic update: W[0, 1] = 0.2000 (reinforced twice)" in out
    assert "W[2, 3] = 0.1000" in out
    assert "Diagonal sum: 0.0000 (hollow diagonal invariant preserved)" in out


def test_task_5():
    out = _capture(ch11_tasks.task_5)
    assert "Two-Timescale Execution Step Complete:" in out
    assert "Slow timescale (weights): optimizer step executed successfully" in out
    assert "Fast timescale (lattice): W[0,2] = 0.0667, W[1,3] = 0.0667" in out
    assert "Substrates remained completely non-interfering: True" in out


def test_task_6():
    out = _capture(ch11_tasks.task_6)
    assert "SDK Parity Audit:" in out
    assert "Dirichlet Energy: SDK = 0.060097 | Book = 0.060097  (Agreement ✔)" in out
    assert "Shannon Entropy:  SDK = 2.317219 | Book = 2.317219  (Agreement ✔)" in out
    assert "Bit-exact mathematical loss parity certified: True" in out
