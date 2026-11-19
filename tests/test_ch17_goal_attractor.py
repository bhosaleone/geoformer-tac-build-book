# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch17_goal_attractor.py — Unit tests for Chapter 17 goal-conditioned attractor.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch17_goal_attractor import (
    BookGoalConditionedAttractor,
    build_untruncated_deep_chain,
    compute_goal_free_energy,
    simulate_certified_32hop_protocol,
)


def test_compute_goal_free_energy():
    num_nodes = 6
    W = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        W[i, i + 1] = 1.0
    W[num_nodes - 1, num_nodes - 1] = 1.0
    deg = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    L_dir = torch.eye(num_nodes) - W / deg
    goal = torch.tensor([3])

    # State at origin vs state at goal
    z_origin = torch.zeros(1, num_nodes); z_origin[0, 0] = 1.0
    z_goal = torch.zeros(1, num_nodes); z_goal[0, 3] = 1.0

    f_origin = compute_goal_free_energy(z_origin, L_dir, goal, lambda_q=2.0)
    f_goal = compute_goal_free_energy(z_goal, L_dir, goal, lambda_q=2.0)

    assert f_origin > f_goal
    assert f_goal < 1.0


def test_untruncated_deep_chain_properties():
    N = 32
    W = build_untruncated_deep_chain(num_entities=N, chain_length=20, alpha_0=2.5)

    # Invariant: No oracle sinks (out-degree > 0 everywhere)
    row_sums = W.sum(dim=-1)
    assert (row_sums > 0.0).all()
    # Hollow diagonal (except terminal self-loop)
    for i in range(N - 1):
        assert W[i, i] == 0.0


def test_goal_conditioned_attractor_exact_halting():
    for k in [4, 8]:
        N = k + 24
        W = build_untruncated_deep_chain(num_entities=N, chain_length=k + 8, alpha_0=2.5)
        attractor = BookGoalConditionedAttractor(num_entities=N, confidence_thresh=0.60)
        src = torch.tensor([0])
        goal = torch.tensor([k])

        out = attractor(src, W, goal)
        tau = int(out["emergent_hops"].item())
        pred = int(torch.argmax(out["final_z"], dim=-1).item())

        assert tau == k
        assert pred == k
        assert out["target_mass"] >= 0.70


def test_certified_32hop_protocol():
    res = simulate_certified_32hop_protocol(depth=32, alpha_0=2.5, num_seeds=3)
    assert all(acc == 1.0 for acc in res["eklavya_acc"])
    assert all(tau == 32 for tau in res["eklavya_tau"])
    assert all(mass >= 0.75 for mass in res["eklavya_mass"])
    assert all(mass < 1e-4 for mass in res["linear_mass"])
