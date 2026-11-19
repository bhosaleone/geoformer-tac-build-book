# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 9 task tests — every "Run this now" transcript in the chapter is
produced by these exact functions; these tests pin the printed output.
"""

from gftac.ch09_tasks import (
    task_1, task_2, task_3, task_4, task_5, task_6,
)


def test_task_1_quasi_crystalline_encodings(capsys):
    task_1()
    out = capsys.readouterr().out
    assert "Quasi-crystalline encoding: N = 2048,  dim = 64" in out
    assert "Row norms: min = 1.000000, max = 1.000000" in out
    assert "Collision rate (rtol=1e-5) on N=2048: 0.00%" in out
    assert "Zero collisions: True" in out


def test_task_2_block_parameter_accounting(capsys):
    task_2()
    out = capsys.readouterr().out
    assert "d_model = 64,  d_ff = 170" in out
    assert "Measured parameters: 52,480" in out
    assert "Formula predicted:   52,480  (match: True)" in out
    assert "Packed footprint:    10,983 B = 10.73 KiB" in out


def test_task_3_full_model_accounting(capsys):
    task_3()
    out = capsys.readouterr().out
    assert "GeoFormer stack: 2 layers, d_model=64, vocab=256, classes=10" in out
    assert "Total model parameters:   122,048" in out
    assert "BitNet b1.58 packed size: 23.84 KiB" in out
    assert "Compression factor:       20.0x" in out


def test_task_4_forward_shape_and_finiteness(capsys):
    task_4()
    out = capsys.readouterr().out
    assert "Input token tensor shape:  [2, 16]" in out
    assert "Output logits shape:       [2, 16, 10]" in out
    assert "Output all finite:         True" in out


def test_task_5_backward_gradient_flow(capsys):
    task_5()
    out = capsys.readouterr().out
    assert "Head weight grad norm: 204.1186  (gradient flows: True)" in out
    assert "Checkpoint roundtrip match: True" in out


def test_task_6_sdk_parity(capsys):
    task_6()
    out = capsys.readouterr().out
    assert "max |h_book - h_sdk| = 0.000e+00" in out
    assert "Bit-exact equivalence: True" in out
