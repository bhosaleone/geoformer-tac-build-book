# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 3 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch03_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_landscape(capsys):
    task_1()
    out = capsys.readouterr().out
    # minima and barrier
    assert "s = -1.0:  F = 0.0000" in out
    assert "s = +0.0:  F = 0.2500" in out
    assert "s = +1.0:  F = 0.0000" in out
    # the gradient points back toward a minimum on each side
    assert "s = -0.5:  F = 0.1406   grad F = +0.3750" in out
    assert "s = +0.5:  F = 0.1406   grad F = -0.3750" in out


def test_task_2_dissipation(capsys):
    task_2()
    out = capsys.readouterr().out
    energies = [float(line.split("F = ")[1]) for line in out.splitlines() if "F = " in line]
    assert energies == sorted(energies, reverse=True)  # never rises
    assert energies[0] > energies[-1]
    assert "the energy column never rises" in out


def test_task_3_gronwall(capsys):
    task_3()
    out = capsys.readouterr().out
    # the prediction equals the measurement for every tolerance shown
    for tol, n in (("1e-03", 135), ("1e-05", 225), ("1e-06", 270)):
        assert f"tol = {tol}:  predicted {n:>3} steps, measured {n:>3}" in out


def test_task_4_basins(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "init +0.01*1  ->  final +0.99998*1" in out
    assert "init -0.01*1  ->  final -0.99998*1" in out
    assert "init +0.00*1  ->  final +0.00000*1   (0 steps)" in out


def test_task_5_dimension_independence(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "N =    10:  270 steps to the drift tolerance" in out
    assert "N =   100:  270 steps to the drift tolerance" in out
    assert "N =  1000:  270 steps to the drift tolerance" in out


def test_task_6_equipartition(capsys):
    task_6()
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("T = ")]
    for line in lines:
        measured = float(line.split("measured std = ")[1].split()[0])
        expected = float(line.split("sqrt(T/lam) = ")[1])
        assert abs(measured - expected) / expected < 0.05
