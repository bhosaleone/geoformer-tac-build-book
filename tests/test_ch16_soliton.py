# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch16_soliton.py — Unit tests for Chapter 16 soliton sharpening and Theorem Q-3.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch16_soliton import (
    adaptive_soliton_gamma,
    analytical_soliton_lower_bound,
    calculate_contraction_ratio,
    calculate_critical_gamma,
    simulate_multihop_soliton_chain,
    soliton_sharpen,
)


def test_soliton_sharpen_properties():
    z = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
    z_sharp = soliton_sharpen(z, gamma=2.0)

    # Must be row-stochastic
    assert torch.allclose(z_sharp.sum(dim=-1), torch.ones(2), atol=1e-5)
    # Peak should be amplified
    assert z_sharp[0, 0] > z[0, 0]
    assert z_sharp[1, 1] > z[1, 1]


def test_critical_gamma_and_contraction_ratio():
    alpha_0 = 2.5
    d_avg = 2.0
    gamma_crit = calculate_critical_gamma(alpha_0, d_avg)

    # ln(2.0) / ln(2.5) ≈ 0.693147 / 0.91629 ≈ 0.75647
    assert abs(gamma_crit - 0.7565) < 1e-3

    rho_super = calculate_contraction_ratio(alpha_0, gamma=2.5, d_avg=d_avg)
    assert rho_super < 1.0

    rho_sub = calculate_contraction_ratio(alpha_0, gamma=0.5, d_avg=d_avg)
    assert rho_sub > 1.0


def test_theorem_q3_32hop_preservation():
    res = simulate_multihop_soliton_chain(num_hops=32, alpha_0=2.5, gamma=2.5, d_avg=2.0)
    assert res["is_supercritical"]
    assert res["satisfies_bound"]
    assert res["final_peak_mass"] >= 0.80
    # Peak mass should remain Ω(1) and stable
    assert abs(res["empirical_peak_mass"][8] - res["empirical_peak_mass"][32]) < 1e-3


def test_entropy_adaptive_gamma_monotonicity():
    g1 = adaptive_soliton_gamma(0.2)
    g2 = adaptive_soliton_gamma(1.0)
    g3 = adaptive_soliton_gamma(2.5)
    g4 = adaptive_soliton_gamma(5.0)

    assert g1 <= g2 <= g3 <= g4
    assert g4 <= 3.5  # Clamped at gamma_max


def test_falsification_boundary_subcritical():
    res_sub = simulate_multihop_soliton_chain(num_hops=8, alpha_0=2.5, gamma=0.5, d_avg=2.0)
    assert not res_sub["is_supercritical"]
    assert res_sub["rho"] > 1.0
    # In subcritical regime, peak mass collapses under distractor leakage
    assert res_sub["final_peak_mass"] < 0.10
