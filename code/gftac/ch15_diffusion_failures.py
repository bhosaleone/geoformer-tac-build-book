# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 15 — How Linear Diffusion Fails: Three Diagnostics.

Implements the three canonical failure modes that doom linear diffusion
and flat attention on multi-hop transitive reasoning:
  1. Residual Drowning: The static initial embedding h_0 dilutes the advancing
     reasoning frontier, decaying relative signal as O(1 / t).
  2. Symmetrization Echo Collapse: Replacing directed flows W with symmetric Laplacians
     (W + W^T)/2 creates artificial 2-cycles that reflect probability mass back onto the
     source rather than traversing the transitive chain.
  3. Exponential Dispersion: In branching graphs with degree d_avg, linear diffusion
     dilutes target mass as (1 / d_avg)^k, crossing below the 1/N chance floor
     at critical depth k* = ceil(ln N / ln d_avg).
  4. The Eklavya Soliton Benchmark: Demonstrates how non-linear gamma-sharpening
     (Theorem Q-3 / B.16.1) defeats all three failure modes, preserving Omega(1) mass.

Theorems proved in Chapter 15:
  B.15.1  Residual Drowning Bound:
          For any additive residual chain h_{t+1} = h_t + eta Delta h_t, the ratio
          of active frontier energy to total accumulated representation energy
          decays monotonically as rho_t <= eta / (1 + eta * t) = O(1/t).
  B.15.2  Symmetrization Echo Invariant:
          For any directed chain with symmetric transition kernel P_sym = (P + P^T)/2,
          the 2-step return probability satisfies P(u -> u | 2 steps) >= 1 / deg(u) > 0,
          whereas the directed kernel satisfies P_dir(u -> u | 2 steps) = 0.
  B.15.3  Exponential Dispersion Floor Crossing:
          In a graph with uniform out-degree d >= 2 and N entities, linear diffusion
          mass on a target at depth k drops strictly below the uniform random guessing
          floor 1/N for all depths k > ln(N) / ln(d).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

__all__ = [
    "simulate_residual_drowning",
    "simulate_symmetrization_echo",
    "simulate_exponential_dispersion",
    "simulate_eklavya_soliton_propagation",
    "calculate_dispersion_crossing_hop",
]


def simulate_residual_drowning(
    chain_length: int = 16,
    eta: float = 0.5,
    num_steps: int = 16,
) -> Dict[str, Any]:
    """Simulates residual accumulation and the dilution of the advancing frontier.

    h_{t+1} = h_t + eta * delta_t, where delta_t is the new hop feature.

    Returns:
        Dict with frontier_ratios (norm of latest increment vs total norm) and step indices.
    """
    dim = 32
    torch.manual_seed(42)
    h_0 = torch.randn(dim)
    h_0 = h_0 / torch.norm(h_0)

    h_t = h_0.clone()
    frontier_ratios = [1.0]

    for step in range(1, num_steps + 1):
        # Orthogonal increment representing newly discovered relational fact
        delta_t = torch.randn(dim)
        delta_t = delta_t - (torch.dot(delta_t, h_t) / torch.dot(h_t, h_t)) * h_t
        delta_t = delta_t / torch.norm(delta_t)

        increment = eta * delta_t
        h_t = h_t + increment

        # Ratio of latest increment energy to total accumulated representation
        ratio = float((torch.norm(increment) / torch.norm(h_t)).item())
        frontier_ratios.append(ratio)

    return {
        "steps": list(range(num_steps + 1)),
        "frontier_ratios": frontier_ratios,
        "is_monotone_decreasing": all(
            frontier_ratios[i] >= frontier_ratios[i + 1] - 1e-6
            for i in range(len(frontier_ratios) - 1)
        ),
    }


def simulate_symmetrization_echo(
    num_nodes: int = 8,
    num_steps: int = 4,
) -> Dict[str, Any]:
    """Compares directed propagation vs symmetrized graph Laplacian propagation.

    Demonstrates how symmetrizing (W + W^T)/2 creates 2-cycle back-echoes
    that trap mass near the source.
    """
    # Directed line graph: 0 -> 1 -> 2 -> 3 -> ... -> N-1
    W_dir = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        W_dir[i, i + 1] = 1.0
    # Add self loop or outgoing on last to avoid zero-division
    W_dir[num_nodes - 1, num_nodes - 1] = 1.0

    # Row-stochastic transition matrix
    deg_dir = W_dir.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    P_dir = W_dir / deg_dir

    # Symmetrized graph
    W_sym = 0.5 * (W_dir + W_dir.t())
    deg_sym = W_sym.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    P_sym = W_sym / deg_sym

    # Initial condition: unit mass at node 0
    z_dir = torch.zeros(num_nodes)
    z_dir[0] = 1.0
    z_sym = z_dir.clone()

    dir_history = [z_dir.clone()]
    sym_history = [z_sym.clone()]

    for _ in range(num_steps):
        z_dir = torch.matmul(z_dir, P_dir)
        z_sym = torch.matmul(z_sym, P_sym)
        dir_history.append(z_dir.clone())
        sym_history.append(z_sym.clone())

    # In directed line, mass moves strictly forward: return to 0 at step 2 is 0.0
    # In symmetric line, 0 -> 1 -> 0 echo causes significant return mass at step 2!
    dir_return_prob_step2 = float(dir_history[2][0].item())
    sym_return_prob_step2 = float(sym_history[2][0].item())

    return {
        "dir_history": [h.tolist() for h in dir_history],
        "sym_history": [h.tolist() for h in sym_history],
        "dir_return_prob_step2": dir_return_prob_step2,
        "sym_return_prob_step2": sym_return_prob_step2,
        "echo_detected": sym_return_prob_step2 > dir_return_prob_step2 + 1e-4,
    }


def calculate_dispersion_crossing_hop(num_entities: int, branching_degree: float) -> int:
    """Calculates the theoretical hop depth k* where (1/d)^k < 1/N."""
    if branching_degree <= 1.0 or num_entities <= 1:
        return 1
    k_star = math.ceil(math.log(num_entities) / math.log(branching_degree))
    return int(k_star)


def simulate_exponential_dispersion(
    branching_degree: float = 3.0,
    num_entities: int = 64,
    max_hops: int = 8,
) -> Dict[str, Any]:
    """Simulates linear mass dilution on branching tree topologies.

    Tracks target mass P(target) = (1 / d)^k and compares against chance floor 1/N.
    """
    chance_floor = 1.0 / num_entities
    k_star = calculate_dispersion_crossing_hop(num_entities, branching_degree)

    depths = list(range(1, max_hops + 1))
    target_masses = [(1.0 / branching_degree) ** k for k in depths]
    below_chance = [mass < chance_floor for mass in target_masses]

    return {
        "depths": depths,
        "target_masses": target_masses,
        "chance_floor": chance_floor,
        "critical_crossing_depth": k_star,
        "below_chance_flags": below_chance,
    }


def simulate_eklavya_soliton_propagation(
    conductance_matrix: torch.Tensor,
    source_node: int,
    target_node: int,
    gamma: float = 2.0,
    num_steps: int = 8,
) -> Dict[str, Any]:
    """Executes Eklavya's non-linear soliton relaxation on a graph with branching decoys.

    At each step:
        z_{t+1} = Softmax(gamma * ln(z_t * P_dir + eps))
    or power-normalized:
        z_{t+1} = (z_t * P_dir)^gamma / sum((z_t * P_dir)^gamma)
    """
    N = conductance_matrix.shape[0]
    deg = conductance_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    P = conductance_matrix / deg

    # Initial condition: one-hot at source
    z = torch.zeros(N)
    z[source_node] = 1.0

    target_mass_history = [float(z[target_node].item())]
    entropy_history = [0.0]

    for step in range(num_steps):
        # Linear propagation step
        z_prop = torch.matmul(z, P)
        # Non-linear soliton power sharpening
        z_sharp = torch.pow(z_prop.clamp(min=1e-12), gamma)
        z = z_sharp / z_sharp.sum()

        target_mass = float(z[target_node].item())
        p_safe = z.clamp(min=1e-12)
        entropy = float((-torch.sum(p_safe * torch.log(p_safe))).item())

        target_mass_history.append(target_mass)
        entropy_history.append(entropy)

    return {
        "final_state": z.tolist(),
        "final_target_mass": target_mass_history[-1],
        "final_entropy": entropy_history[-1],
        "target_mass_history": target_mass_history,
        "entropy_history": entropy_history,
        "soliton_preserved": target_mass_history[-1] > 0.5,
    }
