# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch22_tasks.py — Tests asserting each runnable task in Chapter 22 runs cleanly.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from gftac import ch22_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1():
    out = _capture(ch22_tasks.task_1)
    assert "Claims Ledger Census:" in out
    assert "Total claims registered:        31" in out
    assert "MEASURED         claims:        25" in out
    assert "MODELED          claims:        4" in out
    assert "RESEARCH DEMO    claims:        2" in out
    assert "Every claim carries a valid label: True" in out


def test_task_2():
    out = _capture(ch22_tasks.task_2)
    assert "Artifact Existence Audit:" in out
    assert "Artifacts present on disk:      31" in out
    assert "Missing artifacts:              0" in out
    assert "All cited artifacts exist:      True" in out


def test_task_3():
    out = _capture(ch22_tasks.task_3)
    assert "Test-Linkage Audit:" in out
    assert "Distinct tests cited:           28" in out
    assert "Every row is test-backed:       True" in out


def test_task_4():
    out = _capture(ch22_tasks.task_4)
    assert "Pinned-Value Cross-Check:" in out
    assert "Rows with a pinned value:       22" in out
    assert "All pinned values are asserted: True" in out


def test_task_5():
    out = _capture(ch22_tasks.task_5)
    assert "CLAIMS.md Generation:" in out
    assert "Output file:                    CLAIMS.md" in out
    assert "Ledger rows written:            31" in out
    assert "Rows match registry:            True" in out


def test_task_6():
    out = _capture(ch22_tasks.task_6)
    assert "Full Ledger Audit:" in out
    assert "Rows passing:                   31" in out
    assert "Rows failing:                   0" in out
    assert "Live command probe exit code:   0" in out
    assert "Ledger is green:                True" in out
