# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 1 tests — the measured collapse must reproduce on your machine.

Every number quoted in chapters/ch01_euclidean_bottleneck.md is asserted
here. If this file passes, the chapter's claims are true on your hardware.
"""

import torch

from gftac.ch01_compositional_collapse import (
    build_chain_world,
    transition_matrix,
    propagate,
    propagate_distribution,
    run_single_seed,
    run_demo,
)

SEEDS = (0, 1, 2, 3, 4)


# --- World structure -------------------------------------------------------

def test_world_structure():
    n, edges, chains = build_chain_world(num_chains=3, chain_len=4)
    assert n == 12
    assert len(edges) == 9
    assert len(chains) == 3 and all(len(c) == 4 for c in chains)
    # pure functional chains: out-degree <= 1 everywhere
    seen_src = set()
    for i, _ in edges:
        assert i not in seen_src, "out-degree > 1 in a pure chain world"
        seen_src.add(i)
    # chains are vertex-disjoint and cover the entity set
    flattened = [x for chain in chains for x in chain]
    assert len(set(flattened)) == n


def test_transition_matrix_is_row_stochastic():
    n, edges, chains = build_chain_world()
    P = transition_matrix(n, edges)
    row_sums = P.sum(dim=1)
    live = row_sums > 0
    assert torch.allclose(row_sums[live], torch.ones(int(live.sum())))
    # chain tails are explicit dead ends (all-zero rows), not hidden oracles
    for chain in chains:
        tail = chain[-1]
        assert P[tail].abs().sum().item() == 0.0


# --- The geometric cure -----------------------------------------------------

def test_propagation_is_exact_on_pure_chains():
    n, edges, chains = build_chain_world()
    P = transition_matrix(n, edges)
    for chain in chains:
        for hops in (1, 2, 3):
            assert propagate(P, chain[0], hops) == chain[hops]
        # unit mass lands on the true target — no dispersion on pure chains
        z = propagate_distribution(P, chain[0], 3)
        assert abs(z[chain[3]].item() - 1.0) < 1e-6
        assert z.sum().item() == 1.0 or abs(z.sum().item() - 1.0) < 1e-6


# --- The flat model's failure, controlled and measured ----------------------

def test_flat_model_learns_one_hop_facts():
    """Control: the collapse is architectural, not underfitting.

    The flat model must achieve perfect accuracy on the facts it was shown.
    Otherwise the two-hop failure below would merely be 'did not train'.
    """
    for seed in (0, 1, 2):
        r = run_single_seed(seed)
        assert r["one_hop_train_acc"] == 1.0


def test_two_hop_collapse_and_echo():
    """Headline measurement: 2-hop zero-shot collapse with successor echo."""
    res = run_demo(seeds=SEEDS)
    d = res["summary"]["depths"][2]
    assert d["hits_memorizer"]["mean"] <= 0.20        # collapse ...
    assert d["hits_memorizer"]["mean"] < res["summary"]["chance"]  # ... below chance
    assert d["echo_rate"]["mean"] >= 0.50            # mechanism: successor echo
    assert d["hits_propagation"]["mean"] == 1.0      # cure: perfect routing


def test_three_hop_collapse_and_echo():
    """The collapse persists at 3 hops; propagation still perfect."""
    res = run_demo(seeds=SEEDS)
    d = res["summary"]["depths"][3]
    assert d["hits_memorizer"]["mean"] <= 0.20
    assert d["echo_rate"]["mean"] >= 0.50
    assert d["hits_propagation"]["mean"] == 1.0
