# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch24_capstone.py — Unit tests for the Chapter 24 whole-stack capstone.
"""

from __future__ import annotations

import math

from numeric_asserts import assert_close_in

from gftac.ch24_capstone import (
    _all_green,
    certificate_sha256,
    run_capstone,
    stage_1_lattice,
    stage_2_plasticity,
    stage_3_model,
    stage_4_soliton,
    stage_5_export,
    stage_6_code_security,
)


def test_stage_1_lattice_builds_a_quantized_chain():
    r = stage_1_lattice(32)
    assert r.name == "lattice"
    assert r.metrics["nodes"] == 32
    assert r.metrics["edges"] == 31
    assert r.metrics["quantized"] is True
    # Chain Laplacian: lambda_2 = 2 - 2 cos(pi / 32).
    assert math.isclose(r.metrics["fiedler_lambda2"], 0.0096303867, rel_tol=1e-4, abs_tol=1e-4)


def test_stage_2_plasticity_contracts_resistance_and_keeps_invariants():
    r = stage_2_plasticity(32)
    assert r.metrics["edges_updated"] == 31
    assert r.metrics["contracted"] is True
    assert r.metrics["invariants"] is True
    assert r.metrics["resistance_after"] < r.metrics["resistance_before"]


def test_stage_3_model_counts_parameters():
    r = stage_3_model(32)
    assert r.metrics["params"] == 29152
    assert r.metrics["batch"] == 3
    assert r.metrics["num_classes"] == 4
    assert math.isclose(r.metrics["logits_sum"], 2.62864494324, rel_tol=1e-4, abs_tol=1e-4)


def test_stage_4_soliton_is_supercritical_and_meets_its_bound():
    r = stage_4_soliton(32)
    assert r.metrics["hops"] == 32
    assert r.metrics["supercritical"] is True
    assert r.metrics["satisfies_bound"] is True
    assert math.isclose(r.metrics["final_peak_mass"], 0.812405288219, rel_tol=1e-4, abs_tol=1e-4)


def test_stage_5_export_packs_five_trits_per_byte():
    r = stage_5_export(1024)
    assert r.metrics["trits"] == 1024
    assert r.metrics["packed_bytes"] == 205
    assert r.metrics["packed_bytes"] == r.metrics["theoretical_bytes"]
    assert r.metrics["bits_per_parameter"] == 1.6


def test_stage_6_code_security_reaches_sinks_and_blocks_on_sanitizer():
    r = stage_6_code_security()
    assert r.metrics["vulnerable_sinks_reached"] == 2
    assert r.metrics["safe_sinks_reached"] == 0
    assert r.metrics["taint_reached_sink"] is True
    assert r.metrics["sanitizer_blocked_taint"] is True


def test_run_capstone_has_six_stages_in_order():
    results = run_capstone()
    assert [r.name for r in results] == [
        "lattice", "plasticity", "model", "soliton", "export", "code_security",
    ]


def test_certificate_is_deterministic_and_64_hex():
    a = certificate_sha256(run_capstone())
    b = certificate_sha256(run_capstone())
    assert a == b
    assert len(a) == 64
    int(a, 16)  # valid hex


def test_whole_stack_is_green():
    assert _all_green(run_capstone()) is True
