# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 6 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch06_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_rmsnorm_radial(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "RMS(n(x)) = 1.000000" in out
    assert "c =   0.01  cos = 1.0000001192  rho = 0.996549  pred = 0.996549" in out
    assert "c = 100.00  cos = 1.0000000000  rho = 1.000000  pred = 1.000000" in out


def test_task_2_silu_minimum(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "global minimum at z* = -1.278464,  g(z*) = -0.278465" in out
    assert "z =   -1.278  ->  g(z) =  -0.278465" in out
    assert "z =    2.000  ->  g(z) =   1.761594" in out


def test_task_3_gate_statistics(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "E[g(z)]         = 0.206285" in out
    assert "E[|g(z)|]       = 0.399013" in out
    assert "P(g(z) < 0)     = 0.500803" in out
    assert "P(|g(z)| < 0.1) = 0.161772" in out


def test_task_4_parameter_accounting(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "d_model = 64, d_ff = 170" in out
    assert "3 * d_model * d_ff      = 32,640" in out
    assert "8 * d_model^2 (ideal)   = 32,768" in out
    assert "packed ternary footprint = ceil(32,640/5) = 6,528 B" in out


def test_task_5_rmsnorm_parity(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "max |book - sdk| = 0.000e+00" in out
    assert "bit-exact parity: True" in out


def test_task_6_swiglu_parity(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "weight-only (act_bits=None): max |book - sdk| = 0.000e+00" in out
    assert "SDK default (8-bit activations): max |book - sdk| = 0.000e+00" in out
