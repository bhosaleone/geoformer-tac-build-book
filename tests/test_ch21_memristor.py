# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch21_memristor.py — Unit tests for the Chapter 21 memristor model.

Every assertion here is a MODELED statement about the analytical 1T1R crossbar
model; the SDK parity tests pin the book model against
``geoformer.hardware.crossbar`` and ``geoformer.hardware.energy``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gftac.ch21_memristor import (
    CrossbarConfig,
    CrossbarEnergyAuditor,
    MemristorCrossbar,
    NoiseResilienceSimulator,
    drift_factor,
    energy_law_nj,
    generate_tier4_chains,
    sensitivity_table,
)


# ---------------------------------------------------------------------------
# The energy law
# ---------------------------------------------------------------------------

def test_config_delta_g():
    cfg = CrossbarConfig()
    assert cfg.delta_g == pytest.approx((150e-6 - 10e-6) / 15)
    assert cfg.delta_g == pytest.approx(9.333333333333333e-06)


def test_energy_law_value():
    g_total = 64 * 64 * (10e-6 + 8 * CrossbarConfig().delta_g)
    e_nj = energy_law_nj(g_total, 0.2, 10e-9)
    assert e_nj == pytest.approx(0.4438971733333335, abs=1e-9)
    # The law is E = 2 * 1.6 * V^2 * G * t, in nJ.
    assert e_nj == pytest.approx(2.0 * 1.6 * (0.2 ** 2) * g_total * 10e-9 * 1e9, abs=1e-15)


def test_energy_quadratic_in_voltage():
    g = 1e-3
    e1 = energy_law_nj(g, 0.1, 10e-9)
    e2 = energy_law_nj(g, 0.2, 10e-9)
    assert e2 == pytest.approx(4.0 * e1, rel=1e-12)
    assert e2 > e1


def test_energy_linear_in_pulse_and_conductance():
    base = energy_law_nj(1e-3, 0.2, 10e-9)
    assert energy_law_nj(1e-3, 0.2, 20e-9) == pytest.approx(2.0 * base, rel=1e-12)
    assert energy_law_nj(2e-3, 0.2, 10e-9) == pytest.approx(2.0 * base, rel=1e-12)
    assert energy_law_nj(1e-3, 0.2, 10e-9, n_stages=4) == pytest.approx(2.0 * base, rel=1e-12)


# ---------------------------------------------------------------------------
# Programming and quantization
# ---------------------------------------------------------------------------

def test_quantization_levels():
    cb = MemristorCrossbar()
    assert int(cb.quantize_levels(0.0)) == 0
    assert int(cb.quantize_levels(0.5)) == 8      # round(7.5) -> 8 (half to even)
    assert int(cb.quantize_levels(1.0)) == 15
    assert int(cb.quantize_levels(0.1)) == 2      # round(1.5) -> 2
    assert int(cb.quantize_levels(-3.0)) == 0
    assert int(cb.quantize_levels(3.0)) == 15


def test_uniform_half_program_cell_conductance():
    cb = MemristorCrossbar()
    cb.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
    assert cb.G_ideal[0, 0] == pytest.approx(84.66666666666666e-6)
    assert cb.total_conductance() == pytest.approx(64 * 64 * 84.66666666666666e-6)
    assert cb.compute_dynamic_energy_per_inference() == pytest.approx(0.4438971733333335, abs=1e-9)
    assert cb.compute_dynamic_energy_pj() == pytest.approx(443.8971733333335, abs=1e-6)


def test_no_noise_is_exact_ideal():
    cb = MemristorCrossbar()
    cb.program_conductance_matrix(np.full((8, 8), 0.5), noise_std=0.0)
    assert np.allclose(cb.G_actual, cb.G_ideal)


def test_programming_noise_is_seeded():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    a = MemristorCrossbar()
    b = MemristorCrossbar()
    a.program_conductance_matrix(np.full((16, 16), 0.5), noise_std=0.2, rng=rng_a)
    b.program_conductance_matrix(np.full((16, 16), 0.5), noise_std=0.2, rng=rng_b)
    assert np.allclose(a.G_actual, b.G_actual)
    # Clipping window respected.
    assert np.all(a.G_actual >= CrossbarConfig().g_min * 0.5)
    assert np.all(a.G_actual <= CrossbarConfig().g_max * 1.5)


# ---------------------------------------------------------------------------
# Energy-arrangement invariance (Theorem B.21.5)
# ---------------------------------------------------------------------------

def test_energy_invariant_under_arrangement():
    cfg = CrossbarConfig(n_inputs=8, n_outputs=8)
    W = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    cb_a = MemristorCrossbar(cfg)
    cb_b = MemristorCrossbar(cfg)
    cb_a.program_conductance_matrix(W, noise_std=0.0)
    # Same multiset of conductances, permuted arrangement.
    cb_b.G_actual = cb_a.G_actual.ravel()[::-1].reshape(8, 8).copy()
    assert cb_a.compute_dynamic_energy_per_inference() == pytest.approx(
        cb_b.compute_dynamic_energy_per_inference(), rel=1e-12)


# ---------------------------------------------------------------------------
# Drift (Theorem B.21.4)
# ---------------------------------------------------------------------------

def test_drift_factor_law():
    assert drift_factor(1.0) == pytest.approx(1.0)
    assert drift_factor(1e3) == pytest.approx(1000.0 ** (-0.045), rel=1e-12)
    assert drift_factor(1e3) == pytest.approx(0.7328245, rel=1e-6)
    assert drift_factor(1e6) < drift_factor(1e3) < drift_factor(1.0)


def test_drift_lowers_energy_monotonically():
    cb = MemristorCrossbar()
    cb.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
    e0 = cb.compute_dynamic_energy_per_inference()
    f = cb.apply_drift(1e6)
    assert f < 1.0
    assert cb.compute_dynamic_energy_per_inference() == pytest.approx(e0 * f, rel=1e-12)


# ---------------------------------------------------------------------------
# SDK parity
# ---------------------------------------------------------------------------

def test_sdk_parity_energy():
    from geoformer.hardware.crossbar import PhysicalMemristorCrossbar

    W = np.full((64, 64), 0.5)
    book = MemristorCrossbar(CrossbarConfig())
    sdk = PhysicalMemristorCrossbar()
    book.program_conductance_matrix(W, noise_std=0.0)
    sdk.program_conductance_matrix(W, noise_std=0.0)
    diff = abs(book.compute_dynamic_energy_per_inference()
               - float(sdk.compute_dynamic_energy_per_inference()))
    assert diff < 1e-9
    assert diff == 0.0


def test_sdk_parity_auditor():
    from geoformer.hardware.crossbar import PhysicalMemristorCrossbar
    from geoformer.hardware.energy import NanojouleEnergyAuditor

    W = np.full((64, 64), 0.5)
    book = MemristorCrossbar(CrossbarConfig())
    sdk = PhysicalMemristorCrossbar()
    book.program_conductance_matrix(W, noise_std=0.0)
    sdk.program_conductance_matrix(W, noise_std=0.0)

    book_audit = CrossbarEnergyAuditor.audit(book)
    sdk_audit = NanojouleEnergyAuditor.audit(sdk)
    assert set(book_audit) == set(sdk_audit)
    for key in book_audit:
        assert book_audit[key]["baseline_nj"] == sdk_audit[key]["baseline_nj"]
        assert book_audit[key]["reduction_ratio"] == pytest.approx(
            sdk_audit[key]["reduction_ratio"], rel=1e-12)
    assert book_audit["nvidia_gpu_fp32"]["reduction_ratio"] == pytest.approx(9461.650698203737, rel=1e-9)


# ---------------------------------------------------------------------------
# Monte-Carlo noise resilience
# ---------------------------------------------------------------------------

def test_tier4_chain_generator_shape():
    one_hop, two_hop = generate_tier4_chains(64, 40, seed=42)
    assert len(one_hop) == 80
    assert len(two_hop) == 40
    assert len(set(one_hop)) == 80           # edges are unique
    for a, c in two_hop:
        assert a != c


def test_noise_simulator_deterministic():
    sim = NoiseResilienceSimulator(trials=2)
    a = sim.run()
    b = sim.run()
    assert a == b


def test_noise_simulator_headline():
    sim = NoiseResilienceSimulator(trials=5)
    results = sim.run()
    by_noise = {n: m for n, m, _ in results}
    assert set(by_noise) == {0.0, 0.05, 0.10, 0.15, 0.20}
    # The published headline: ~38% accuracy at 20% conductance noise.
    assert by_noise[0.20] == pytest.approx(38.0, abs=0.05)
    # Resilience: every level stays well above the 1/64 chance floor.
    assert all(m > 30.0 for m in by_noise.values())


def test_routing_margin_contraction_bounds():
    """Theorem B.21.2: a relative conductance perturbation <= sigma moves each
    output current by at most a relative sigma on a single VMM stage."""
    cfg = CrossbarConfig(n_inputs=16, n_outputs=16)
    cb = MemristorCrossbar(cfg)
    rng = np.random.default_rng(7)
    W = rng.uniform(0.0, 1.0, (16, 16))
    cb.program_conductance_matrix(W, noise_std=0.0)
    v = rng.uniform(0.0, 0.2, 16)

    i_clean = cb.compute_analog_vmm(v, apply_parasitics=False)
    sigma = 0.1
    G_pert = cb.G_actual * (1.0 + sigma * rng.uniform(-1.0, 1.0, cb.G_actual.shape))
    i_pert = G_pert @ v
    # |i'_j - i_j| <= sigma * i_j for non-negative inputs.
    assert np.all(np.abs(i_pert - i_clean) <= sigma * i_clean + 1e-15)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def test_sensitivity_invariance():
    rows = sensitivity_table(perturbations=(0.9, 1.1))
    ratios = [r[4] for r in rows]
    assert min(ratios) > 1000.0
    # The baseline row is the 443.9 pJ figure.
    assert rows[0][0] == "baseline"
    assert rows[0][3] == pytest.approx(443.8971733333335, abs=1e-6)


def test_sensitivity_voltage_is_strongest_knob():
    rows = sensitivity_table(perturbations=(0.9, 1.1))
    by_param = {}
    for param, p, e_nj, _, _ in rows:
        by_param.setdefault(param, {})[round(p, 2)] = e_nj
    v_span = by_param["v_read"][1.1] - by_param["v_read"][0.9]
    g_span = by_param["g_min"][1.1] - by_param["g_min"][0.9]
    assert v_span > g_span
