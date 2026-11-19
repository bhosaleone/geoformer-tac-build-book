# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 2 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch02_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6, task_7,
)


def test_task_1_metric_appears(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "before any fact: d_01 = 1.000000e+08" in out
    assert "after one dw write: d_01 = 14.999997" in out


def test_task_2_shadow_accumulates(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "levels: 15, dw = 0.0667" in out
    assert "write 1 of dw/5: shadow = 0.0133, stored level = 0" in out
    assert "write 2 of dw/5: shadow = 0.0267, stored level = 0" in out
    assert "write 3 of dw/5: shadow = 0.0400, stored level = 1" in out
    assert "write 5 of dw/5: shadow = 0.0667, stored level = 1" in out


def test_task_3_stochastic_unbiased(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "1000 writes of 0.37*dw through the stochastic projector" in out
    mean = float(out.split("mean stored level: ")[1].split()[0])
    assert abs(mean - 0.37) < 0.05


def test_task_4_harmonic_law(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "level  1: d_01 = 14.999997" in out
    assert "level  3: d_01 = 5.000000" in out
    assert "(15/2 = 7.500000)" in out
    assert "(15/5 = 3.000000)" in out


def test_task_5_identity_holds(capsys):
    task_5()
    out = capsys.readouterr().out
    diff_line = [l for l in out.splitlines() if l.startswith("difference")][0]
    diff = float(diff_line.split("=")[1])
    assert diff < 1e-5


def test_task_6_floor_and_boundary(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "uniform complete lattice: lambda_2 = 0.800000" in out
    assert "(tight: equal)" in out
    assert "chain world: lambda_2 = 0.381966" in out


def test_task_7_dead_ends_honest(capsys):
    task_7()
    out = capsys.readouterr().out
    assert "dead ends at nodes: [3, 4]" in out
    assert "live row 2: [0.0, 0.0, 0.0, 1.0, 0.0]" in out
    assert "dead row 4: [0.0, 0.0, 0.0, 0.0, 0.0]" in out
