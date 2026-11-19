# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 7 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch07_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_energy_descent_and_step_condition(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "lambda_max(A) = 1.236603,  stability threshold 2/lambda_max = 1.617334" in out
    assert "tau =   1   F(h_tau) = 1.5988730788" in out
    assert "tau =  40   F(h_tau) = 1.3846600056" in out
    assert "exact threshold dt* = 2||g||^2/(g^T A g) = 1.700029" in out
    assert "F at 0.99*dt* = 1.5904019058  (decrease: True)" in out
    assert "F at 1.01*dt* = 1.6075152278  (increase: True)" in out


def test_task_2_contraction_bound(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "lambda_min(A) = 0.999999,  delta_tau = 0.1" in out
    assert "||e_0|| = 1.480406" in out
    assert "    1            1.30677342              1.33236521" in out
    assert "   40            0.01054329              0.02188183" in out


def test_task_3_halting(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "schedule budget (tau_max) = 200" in out
    assert "actual_tau (halting step) = 54" in out
    assert "halting step <= budget:   True" in out
    assert "||h_tau - h*|| at halt    = 1.915e-03" in out
    assert "final K_S (S5 readout)    = 6.474590" in out


def test_task_4_permutation_invariance(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "schedule          : [0.05, 0.1, 0.2, 0.4, 0.05, 0.1]" in out
    assert "permuted          : [0.4, 0.05, 0.1, 0.05, 0.2, 0.1]" in out
    assert "max |out - out_perm|  float32 = 2.384e-07" in out
    assert "max |out - out_perm|  float64 = 4.441e-16" in out


def test_task_5_chebyshev_acceleration(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "spectrum [0.999999, 1.236603],  kappa = 1.2366" in out
    assert "   1      0.10578738      0.10578738        1.0x" in out
    assert "   6      0.00000140      0.00000004       31.5x" in out
    assert "chebyshev k=4 step sizes: 0.9911, 0.9319, 0.8594, 0.8146" in out


def test_task_6_sdk_parity(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "max |h_book - h_sdk| = 0.000e+00" in out
    assert "actual_tau: book 8, sdk 8   (equal: True)" in out
    assert "|K_S book - K_S sdk| = 0.000e+00" in out
