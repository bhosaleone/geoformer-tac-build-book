# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 5 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch05_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_row_stochastic(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "attn shape (B, heads, N, N): (1, 4, 6, 6)" in out
    assert "min entry = 1.7471e-02  (>= 0)" in out
    assert "row sums: min 0.9999999, max 1.0000001" in out
    assert "head 0, row 0: [0.1982, 0.0871, 0.3114, 0.2829, 0.1029, 0.0175]" in out


def test_task_2_geodesic_shift(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "lambda_geo = 0.5" in out
    assert "d[0, :]           = [0, 1, 2, 3, 4]" in out
    assert "logit shift row 0 = [0.0, 0.5, 1.0, 1.5, 2.0]" in out
    assert "shift == lambda*d everywhere: True" in out


def test_task_3_concentration_law(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "        0.00    1.948021        1.618854" in out
    assert "        5.00    0.020034        0.097039" in out
    assert "E[d] non-increasing in lambda_geo: True" in out


def test_task_4_distance_derivative(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "analytic  -Var_a(d)           : -1.494368" in out
    assert "finite difference d E[d]/dlam : -1.496077" in out
    assert "identity holds: True" in out


def test_task_5_finsler(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "B antisymmetric (max|B + B^T|): 0.0e+00" in out
    assert "(L_ij - L_ji) - (S_ij - S_ji) for i=0,j=3: 3.000000" in out
    assert "2 * B[0, 3]                              : 3.000000" in out
    assert "identity holds everywhere: True" in out


def test_task_6_parity(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "max |attn_book - attn_sdk| = 0.000e+00" in out
    assert "max |out_book  - out_sdk | = 0.000e+00" in out
    assert "bit-exact parity: True" in out
