# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 1 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch01_tasks import task_1, task_2, task_3


def test_task_1_world_and_kernel(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "12 entities, 9 one-hop facts, 3 chains" in out
    assert "chain 0: [0, 1, 2, 3]" in out
    assert "dead ends (all-zero rows): nodes [3, 7, 11]" in out


def test_task_2_propagation_readout(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "z = e_0 P^2 =" in out
    assert "argmax(z) = 2  (true two-hop target: 2)" in out


def test_task_3_flat_model_learns_then_echoes(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "one-hop train accuracy: 1.00" in out
    assert "model answers 1; successor is 1, true target is 2" in out
    # the loss curve is decreasing (learning happened, not underfitting)
    losses = [float(line.split("cross-entropy ")[1])
              for line in out.splitlines() if "cross-entropy" in line]
    assert losses[0] > losses[1] > losses[2]
