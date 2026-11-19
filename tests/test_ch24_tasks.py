# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch24_tasks.py — Tests asserting each runnable capstone task runs cleanly.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from gftac import ch24_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch24_tasks.task_1)
    assert "Stage 1 — Metric lattice:" in out
    assert "edges                        31" in out
    assert "fiedler_lambda2              0.00963038671762" in out
    assert "nodes                        32" in out
    assert "quantized                    True" in out


def test_task_2():
    out = _capture(ch24_tasks.task_2)
    assert "Stage 2 — Hebbian plasticity:" in out
    assert "contracted                   True" in out
    assert "edges_updated                31" in out
    assert "invariants                   True" in out
    assert "resistance_after             4.99999967549" in out


def test_task_3():
    out = _capture(ch24_tasks.task_3)
    assert "Stage 3 — Model forward:" in out
    assert "logits_sum                   2.62864494324" in out
    assert "params                       29152" in out


def test_task_4():
    out = _capture(ch24_tasks.task_4)
    assert "Stage 4 — Soliton propagation:" in out
    assert "final_peak_mass              0.812405288219" in out
    assert "satisfies_bound              True" in out
    assert "supercritical                True" in out


def test_task_5():
    out = _capture(ch24_tasks.task_5)
    assert "Stage 5 — Packed export:" in out
    assert "bits_per_parameter           1.6" in out
    assert "packed_bytes                 205" in out
    assert "trits                        1024" in out


def test_task_6():
    out = _capture(ch24_tasks.task_6)
    assert "Stage 6 — Code-security transfer (RESEARCH DEMO):" in out
    assert "safe_sinks_reached           0" in out
    assert "sanitizer_blocked_taint      True" in out
    assert "vulnerable_sinks_reached     2" in out


def test_task_7():
    out = _capture(ch24_tasks.task_7)
    assert "Whole-Stack Certificate:" in out
    assert "Stages executed:                6" in out
    assert "Certificate SHA-256 (first 16): a9e0b239a83fe32c" in out
    assert "All stages green:               True" in out
