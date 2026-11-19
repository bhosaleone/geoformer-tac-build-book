# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch15_diffusion_failures.py — Unit tests for Chapter 15 linear diffusion diagnostics.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch15_diffusion_failures import (
    calculate_dispersion_crossing_hop,
    simulate_eklavya_soliton_propagation,
    simulate_exponential_dispersion,
    simulate_residual_drowning,
    simulate_symmetrization_echo,
)


def test_residual_drowning_monotonicity():
    res = simulate_residual_drowning(chain_length=12, eta=0.4, num_steps=10)
    assert res["is_monotone_decreasing"]
    assert res["frontier_ratios"][0] == 1.0
    assert res["frontier_ratios"][-1] < 0.35


def test_symmetrization_echo_detected():
    res = simulate_symmetrization_echo(num_nodes=8, num_steps=4)
    # Directed return probability to origin at step 2 must be identically 0
    assert res["dir_return_prob_step2"] == 0.0
    # Symmetrized return probability reflects backwards
    assert res["sym_return_prob_step2"] >= 0.40
    assert res["echo_detected"]


def test_exponential_dispersion_crossing():
    res = simulate_exponential_dispersion(branching_degree=3.0, num_entities=64, max_hops=6)
    k_star = res["critical_crossing_depth"]
    assert k_star == 4

    # Depths below k_star must be above chance floor
    for k in range(k_star - 1):
        assert not res["below_chance_flags"][k]
    # Depths at or above k_star must be below chance floor
    assert res["below_chance_flags"][k_star - 1]


def test_calculate_dispersion_crossing_hop():
    assert calculate_dispersion_crossing_hop(64, 2.0) == 6
    assert calculate_dispersion_crossing_hop(64, 4.0) == 3
    assert calculate_dispersion_crossing_hop(1000, 10.0) == 3
    assert calculate_dispersion_crossing_hop(1000, 2.0) == 10


def test_eklavya_soliton_advantage():
    N = 8
    W = torch.zeros(N, N)
    for i in range(3):
        W[i, i + 1] = 1.0
        decoy = 4 + i
        W[i, decoy] = 0.4
        W[decoy, decoy] = 1.0
    W[3, 3] = 1.0

    res_lin = simulate_eklavya_soliton_propagation(W, 0, 3, gamma=1.0, num_steps=5)
    res_ekl = simulate_eklavya_soliton_propagation(W, 0, 3, gamma=2.5, num_steps=5)

    assert res_ekl["soliton_preserved"]
    assert res_ekl["final_target_mass"] > res_lin["final_target_mass"] * 2.0
    assert res_ekl["final_entropy"] < res_lin["final_entropy"]
