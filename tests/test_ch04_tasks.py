# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 4 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch04_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_quantization(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "gamma = mean|w| = 0.468750" in out
    assert "w = +0.900  ->  w/gamma = +1.9200  ->  trit +1" in out
    assert "w = -1.100  ->  w/gamma = -2.3467  ->  trit -1" in out
    assert "w = +0.200  ->  w/gamma = +0.4267  ->  trit +0" in out


def test_task_2_scale_invariance(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "quantize(w)      trits: [-1, -1, 0, -1, 1, 1, 0, -1, 0, -1, 0, 0, 0, 1, 1, 0]" in out
    assert "trit patterns identical: True" in out


def test_task_3_ste(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "parameter gradient: [-0.6167, -1.4763, -1.1502, 0.1441]" in out
    assert "dL/dw_eff expected: [-0.6167, -1.4763, -1.1502, 0.1441]" in out
    assert "STE identity holds: True" in out


def test_task_4_packing(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "packed (3 bytes): [140, 234, 4]" in out
    assert "lossless: True  (ceil(13/5) = 3 bytes)" in out


def test_task_5_footprint(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "foundation parameters: 249,665" in out
    assert "packed bytes: ceil(249,665/5) = 49,933 B" in out
    assert "footprint: 48.76 KiB  (FP32 would need 975.3 KiB)" in out
    assert "bits/param: 1.6000  (information floor log2(3) = 1.5850)" in out


def test_task_6_variance(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "init std = 0.054112   target sqrt(1.5/fan_in) = 0.054127" in out
    assert "E[w_q^2] = 0.6913" in out
    assert "Var(w_eff) = 0.001292   (2/3)gamma^2 = 0.001246" in out
