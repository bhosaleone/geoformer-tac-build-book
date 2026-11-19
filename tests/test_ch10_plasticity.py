# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch10_plasticity.py — rigorous unit, invariant, and property tests
for Chapter 10: Two-Timescale Learning: Hebbian Plasticity with Invariants.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch10_plasticity import (
    ConsolidationReport,
    PlasticConductanceEngine,
    PlasticityConfig,
    demo_ledger,
)


def test_hebbian_resistance_contraction_b10_1():
    """Theorem B.10.1: Resistance contracts monotonically under Hebbian updates."""
    engine = PlasticConductanceEngine(PlasticityConfig(num_entities=4, bits=4))
    d_prev = engine.resistance(0, 1)
    assert d_prev == pytest.approx(1e8, rel=1e-3)

    for step in range(1, 16):
        engine.update_conductance(0, 1, weight_inc=1.0)
        d_curr = engine.resistance(0, 1)
        w_curr = float(engine.W[0, 1])

        # Strictly monotone contraction
        assert d_curr < d_prev, f"Step {step}: resistance did not strictly decrease ({d_curr} >= {d_prev})"
        # Analytical formula check
        expected_d = 1.0 / (w_curr + engine.epsilon)
        assert d_curr == pytest.approx(expected_d, rel=1e-5)
        d_prev = d_curr

    # At full saturation W = 1.0, d = 1.0
    assert engine.resistance(0, 1) == pytest.approx(1.0, rel=1e-4)


def test_invariant_battery_b10_2():
    """Theorem B.10.2: Full invariant battery preservation under batched updates."""
    torch.manual_seed(42)
    engine = PlasticConductanceEngine(PlasticityConfig(num_entities=10, bits=4, seed=42))

    # Run 500 batched random updates
    for _ in range(500):
        src = torch.randint(0, 10, (6,))
        tgt = torch.randint(0, 10, (6,))
        w_inc = torch.rand(1).item() * 3.0
        engine.update_conductance(src, tgt, weight_inc=w_inc)

    # Invariants must hold
    assert engine.assert_invariants() is True
    assert (engine.W >= 0.0).all()
    assert (engine.W <= 1.0).all()
    assert torch.all(torch.diag(engine.W) == 0.0)


def test_subthreshold_shadow_accumulation():
    """Sub-threshold Hebbian writes accumulate in W_raw without loss."""
    engine = PlasticConductanceEngine(PlasticityConfig(num_entities=4, bits=4))
    dw = engine.delta_w

    # Write 0.2 * dw five times: sum is 1.0 * dw
    for step in range(1, 6):
        engine.update_conductance(0, 1, weight_inc=0.2)
        expected_raw = step * 0.2 * dw
        assert float(engine.W_raw[0, 1]) == pytest.approx(expected_raw, rel=1e-5)

    # After 5 steps, quantized store W should be exactly at level 1 (dw)
    assert float(engine.W[0, 1]) == pytest.approx(dw, rel=1e-5)


def test_stochastic_rounding_unbiased():
    """Stochastic rounding produces an unbiased expectation across writes."""
    torch.manual_seed(0)
    dw = 1.0 / 15.0
    val = 0.40 * dw

    # Sample 1000 independent single-write engines
    sampled_levels = []
    for _ in range(1000):
        eng = PlasticConductanceEngine(PlasticityConfig(num_entities=2, bits=4))
        eng.update_conductance(0, 1, weight_inc=0.40, stochastic=True)
        sampled_levels.append(float(eng.W[0, 1]))

    mean_val = sum(sampled_levels) / len(sampled_levels)
    # Target is 0.40 * dw ≈ 0.02667
    assert mean_val == pytest.approx(val, abs=0.005)


def test_endogenous_weyl_energy_stability_b10_3():
    """Theorem B.10.3: Endogenous Weyl consolidation synthesizes transitive shortcuts."""
    N = 4
    cfg = PlasticityConfig(num_entities=N, bits=4, max_probes=64, max_delta_w=0.05, max_hop_depth=4, seed=42)
    engine = PlasticConductanceEngine(cfg)

    # Linear chain: 0 -> 1 -> 2 -> 3
    engine.W[0, 1] = 0.8
    engine.W[1, 2] = 0.8
    engine.W[2, 3] = 0.8

    assert engine.W[0, 2].item() == 0.0
    assert engine.W[0, 3].item() == 0.0

    rep = engine.endogenous_step(num_steps=3)

    # Shortcuts must be synthesized
    assert engine.W[0, 2].item() > 0.0, "2-hop shortcut 0->2 was not synthesized"
    assert engine.W[0, 3].item() > 0.0, "3-hop shortcut 0->3 was not synthesized"

    # Energy is bounded and stable
    assert math.isfinite(rep.energy_after)
    assert rep.energy_after >= rep.energy_before
    assert rep.mean_kappa <= 1.0


def test_sdk_parity_endogenous_invariants():
    """Parity check against SDK EndogenousConsolidator."""
    from geoformer.autonomous.endogenous_engine import ConsolidationConfig as SDKConfig
    from geoformer.autonomous.endogenous_engine import EndogenousConsolidator as SDKConsolidator

    N = 4
    W_init = torch.zeros(N, N)
    W_init[0, 1] = 0.8
    W_init[1, 2] = 0.8
    W_init[2, 3] = 0.8

    sdk_cfg = SDKConfig(max_probes=64, max_delta_w=0.05, max_hop_depth=4, seed=42)
    sdk_engine = SDKConsolidator(sdk_cfg)
    W_sdk, sdk_rep = sdk_engine.step(W_init.clone(), num_steps=3)

    book_cfg = PlasticityConfig(num_entities=N, bits=4, max_probes=64, max_delta_w=0.05, max_hop_depth=4, seed=42)
    book_engine = PlasticConductanceEngine(book_cfg)
    book_engine.W = W_init.clone()
    book_rep = book_engine.endogenous_step(num_steps=3)

    # Invariants match
    assert (W_sdk >= 0.0).all()
    assert (book_engine.W >= 0.0).all()
    assert torch.all(torch.diag(W_sdk) == 0.0)
    assert torch.all(torch.diag(book_engine.W) == 0.0)
    assert W_sdk[0, 2].item() > 0.0
    assert book_engine.W[0, 2].item() > 0.0


def test_demo_ledger_contract():
    """Demo ledger executes completely and certifies invariant pass."""
    res = demo_ledger()
    assert res["invariants_passed"] is True
    assert res["initial_resistance"] == pytest.approx(1e8, rel=1e-3)
    assert res["final_resistance"] <= res["initial_resistance"]
