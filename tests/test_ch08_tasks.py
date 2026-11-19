# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 8 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""
from numeric_asserts import assert_close_in

from gftac.ch08_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_frobenius_schur_accounting(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "Burnside sum: sum(d_i^2) = 120  (|S_5| = 5! = 120)" in out
    assert "Invariant subspace dim: sum(d_i) = 26" in out
    assert "Standard                      4      16" in out
    assert "2-Hook Young                  6      36" in out


def test_task_2_projector_algebra(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "d_model = 32,  target invariant rank = 26" in out
    assert "Symmetry:    |P - P^T|  =" in out
    assert "Idempotence: |P^2 - P|  =" in out
    assert "Matrix rank(P)          = 26" in out
    assert_close_in("Matrix trace(P)         = 26.0000", out)


def test_task_3_energy_decomposition(capsys):
    task_3()
    out = capsys.readouterr().out
    assert_close_in("||h||^2                 = 28.568502", out)
    assert_close_in("||Ph||^2 (symmetric)    = 24.111300", out)
    assert_close_in("||(I-P)h||^2 (residual) = 4.457200", out)
    assert "Inner product <Ph, h_perp> =" in out


def test_task_4_kolmogorov_capacities(capsys):
    task_4()
    out = capsys.readouterr().out
    assert_close_in("Total multi-channel capacity K_total = 54.7291 bits", out)
    assert_close_in("Max single-channel capacity          = 15.5098 bits", out)
    assert_close_in("Standard                   dim=4   capacity =  8.0000 bits", out)
    assert_close_in("2-Hook Young               dim=6   capacity = 15.5098 bits", out)


def test_task_5_empirical_shannon_complexity(capsys):
    task_5()
    out = capsys.readouterr().out
    assert_close_in("Theoretical maximum K_S bound (log2(120)) = 6.906891 bits", out)
    assert_close_in("K_S for random Gaussian activation       = 6.397839 bits", out)
    assert_close_in("K_S for single-mode activation           = 5.582055 bits", out)
    assert_close_in("Random K_S <= 6.9069: True", out)
    assert "Single-mode K_S < Random K_S: True" in out


def test_task_6_sdk_parity(capsys):
    task_6()
    out = capsys.readouterr().out
    assert_close_in("max |h_sym_book - h_sym_sdk| = 0.000e+00", out)
    assert_close_in("|K_S_book - K_S_sdk|         = 0.000e+00", out)
    assert "Bit-exact projection: True" in out
    assert "Bit-exact complexity: True" in out
