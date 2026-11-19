# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch21_tasks.py — Pins every printed line of the Chapter 21 tasks.

Each task is run in-process and its stdout is compared, line for line, against
the exact transcript that appears in ``chapters/ch21_memristor.md``. Because the
model is deterministic and seeded, no line is run-dependent.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from gftac import ch21_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _assert_exact(actual: str, expected: str) -> None:
    a = [ln.rstrip() for ln in actual.strip("\n").split("\n")]
    e = [ln.rstrip() for ln in expected.strip("\n").split("\n")]
    assert a == e


TASK_1 = """\
Task 1 — First-Principles Re-Derivation of the 443.9 pJ Figure (MODELED)
  • Crossbar array:          64 x 64 = 4,096 1T1R cells
  • HRS g_min:               10.000 uS
  • LRS g_max:               150.000 uS
  • 4-bit levels / step:     16 states, delta_g = 9.3333 uS
  • Uniform W_norm = 0.5 ->  level 8 ->  cell G = 84.6667 uS
  • Total conductance sum(G):346.794667 mS
  • One-stage energy V^2*G*t:1.387179e-10 J
  • Two-stage energy:        2.774357e-10 J
  • x1.6 peripheral factor:  4.438972e-10 J
  • Total energy:            0.443897 nJ = 443.897 pJ
  • Reduction vs FP32 GPU:   9,462x"""

TASK_2 = """\
Task 2 — SDK Parity Audit: book model vs geoformer.hardware (MODELED)
  • Book crossbar energy:    0.443897173 nJ
  • SDK  crossbar energy:    0.443897173 nJ
  • Absolute difference:     0.000e+00 nJ
  • Book GPU-FP32 ratio:     9,461.6507x
  • SDK  GPU-FP32 ratio:     9,461.6507x
  • Parity verified (<1e-9): True"""

TASK_3 = """\
Task 3 — Monte-Carlo Noise-Resilience Curve (MODELED)
  Accuracy = held-out 2-hop argmax hits / queries,
  mean over 5 seeds x 5 noise realizations (1/64 chance = 1.56%).
   Noise sigma | Mean accuracy |  Std dev | Status
  --------------------------------------------------------
         0.0% |         38.4% |     5.7% | RESILIENT
         5.0% |         38.1% |     7.1% | RESILIENT
        10.0% |         37.3% |     7.1% | RESILIENT
        15.0% |         37.6% |     7.2% | RESILIENT
        20.0% |         38.0% |     6.8% | RESILIENT
  --------------------------------------------------------
  • Headline: 38.0% accuracy at 20.0% conductance noise
  • Above the 1.56% chance floor: True"""

TASK_4 = """\
Task 4 — Sensitivity Analysis: +/-10% on Each Physical Parameter (MODELED)
   Parameter |  Perturb | Energy (pJ) |  GPU ratio
  --------------------------------------------------
    baseline |    1.00x |     443.897 |     9,462x
       g_min |    0.90x |     441.450 |     9,514x
       g_min |    1.10x |     446.344 |     9,410x
       g_max |    0.90x |     401.954 |    10,449x
       g_max |    1.10x |     485.840 |     8,645x
      v_read |    0.90x |     359.557 |    11,681x
      v_read |    1.10x |     537.116 |     7,820x
     t_pulse |    0.90x |     399.507 |    10,513x
     t_pulse |    1.10x |     488.287 |     8,602x
  --------------------------------------------------
  • Baseline energy:         443.897 pJ
  • Worst-case GPU ratio:    7,820x
  • Thousands-fold conclusion invariant: True"""

TASK_5 = """\
Task 5 — Conductance Drift G(t)=G0*(t/t0)^(-nu) vs Energy (MODELED)
  Drift exponent nu = 0.045 (dimensionless)
        Time | Drift factor | Energy (pJ) | Retained
  --------------------------------------------------
      1e+00s |     1.000000 |     443.897 |   100.0%
      1e+03s |     0.732825 |     325.299 |    73.3%
      1e+06s |     0.537032 |     238.387 |    53.7%
      1e+09s |     0.393550 |     174.696 |    39.4%
  --------------------------------------------------
  • Drift lowers conductance, so energy falls with it
  • Energy is monotone in the retained factor: True"""

TASK_6 = """\
Task 6 — Baseline Energy Comparison per Inference (MODELED)
  Crossbar energy: 0.443897 nJ = 443.897 pJ
  Platform                         |     Energy |    Reduction
  --------------------------------------------------------------
  NVIDIA Server GPU (FP32)         |    4200 nJ |      9,462x
  NVIDIA Tensor Core GPU (FP16)    |    1100 nJ |      2,478x
  Google Edge TPU (INT8)           |     145 nJ |        327x
  ARM Cortex-M55 NPU (INT8)        |      65 nJ |        146x
  Q-GeoFormer Memristor (this model) |    443.9 pJ | 1x (reference)
  --------------------------------------------------------------
  • Headline: 9,462x lower energy than an FP32 server GPU
  • All baselines exceeded by >100x: True"""


def test_task_1():
    _assert_exact(_capture(ch21_tasks.task_1), TASK_1)


def test_task_2():
    _assert_exact(_capture(ch21_tasks.task_2), TASK_2)


def test_task_3():
    _assert_exact(_capture(ch21_tasks.task_3), TASK_3)


def test_task_4():
    _assert_exact(_capture(ch21_tasks.task_4), TASK_4)


def test_task_5():
    _assert_exact(_capture(ch21_tasks.task_5), TASK_5)


def test_task_6():
    _assert_exact(_capture(ch21_tasks.task_6), TASK_6)
