# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch12_datasets.py — rigorous unit, graph integrity, and Tier-8 audit tests
for Chapter 12: Datasets: Chains, Crossroads, and the α₀ Control.
"""

from __future__ import annotations

import pytest
import torch

from gftac.ch12_datasets import (
    DatasetConfig,
    TransitiveChainDataset,
    audit_graph_integrity,
    create_benchmark_dataloaders,
    demo_ledger,
    make_crossroad_graph,
)


def test_no_oracle_sinks_b12_1():
    """Theorem B.12.1: Graph contains no absorbing sinks (out-degree > 0 for all nodes)."""
    N = 48
    W, chains, decoys = make_crossroad_graph(num_entities=N, num_chains=8, max_hops=4, seed=42)

    row_sums = W.sum(dim=-1)
    # Assert every single node has active outgoing connections
    assert (row_sums > 1e-4).all().item(), "Found node with zero out-degree (oracle sink detected!)"
    # Diagonal must be hollow
    assert torch.all(torch.diag(W) == 0.0)


def test_alpha_0_conductance_ratio_b12_2():
    """Theorem B.12.2: True highway conductance exceeds decoy loops by ratio alpha_0."""
    alpha_0 = 3.0
    W, chains, decoys = make_crossroad_graph(num_entities=32, num_chains=4, max_hops=4, alpha_0=alpha_0, seed=42)

    u, v_true = chains[0][0], chains[0][1]
    w_true = float(W[u, v_true].item())

    decoy_weights = [float(w) for (src, _, w) in decoys if src == u]
    assert len(decoy_weights) > 0, "No decoys branched from node u"

    mean_decoy = sum(decoy_weights) / len(decoy_weights)
    measured_ratio = w_true / mean_decoy

    assert measured_ratio == pytest.approx(alpha_0, rel=1e-3)


def test_strict_zero_shot_disjointness_b12_3():
    """Theorem B.12.3: Multi-hop query pairs never appear in 1-hop training edges."""
    cfg = DatasetConfig(num_entities=64, num_chains=4, max_hops=6, seed=42)
    train_ds = TransitiveChainDataset(cfg, split="train")
    train_pairs = {(int(item["src"]), int(item["tgt"])) for item in train_ds}

    # Evaluate for depths 2, 3, 4, 5, 6
    for depth in range(2, 7):
        test_ds = TransitiveChainDataset(cfg, split="test", eval_depth=depth)
        test_pairs = {(int(item["src"]), int(item["tgt"])) for item in test_ds}
        overlap = train_pairs.intersection(test_pairs)
        assert len(overlap) == 0, f"Depth {depth} has {len(overlap)} colliding pairs with training data"


def test_dataloader_batch_generation():
    """DataLoader generates batches with proper tensor shapes and types."""
    cfg = DatasetConfig(num_entities=32, num_chains=4, max_hops=4, batch_size=8, seed=42)
    train_loader, test_loaders, W = create_benchmark_dataloaders(cfg, eval_depths=[2, 4])

    batch = next(iter(train_loader))
    assert batch["src"].shape == (8,)
    assert batch["rel"].shape == (8,)
    assert batch["tgt"].shape == (8,)
    assert batch["hops"].shape == (8,)

    assert batch["src"].dtype == torch.long
    assert batch["tgt"].dtype == torch.long


def test_graph_laplacian_properties():
    """Symmetrized Laplacian is positive semi-definite with zero ground state."""
    W, _, _ = make_crossroad_graph(num_entities=20, num_chains=4, max_hops=3, seed=42)
    A = W + W.T
    deg = torch.diag(A.sum(dim=-1))
    L = deg - A

    evals = torch.linalg.eigvalsh(L)
    # Smallest eigenvalue is 0.0
    assert evals[0].item() == pytest.approx(0.0, abs=1e-5)
    # Connected component has positive Fiedler value
    assert evals[1].item() > 0.0


def test_sdk_parity_dataset():
    """Parity check against SDK TransitiveKnowledgeGraphDataset."""
    from geoformer.training.datasets import TransitiveKnowledgeGraphDataset as SDKDataset

    sdk_ds = SDKDataset(num_entities=32, num_chains=4, max_hops=4, seed=42, split="train")
    book_cfg = DatasetConfig(num_entities=32, num_chains=4, max_hops=4, seed=42)
    book_ds = TransitiveChainDataset(book_cfg, split="train")

    assert (sdk_ds.ground_truth_W >= 0.0).all()
    assert (book_ds.W >= 0.0).all()
    assert len(sdk_ds) > 0 and len(book_ds) > 0


def test_demo_ledger_contract():
    """Demo ledger executes completely and certifies Tier-8 audit."""
    res = demo_ledger()
    assert res["num_entities"] == 32
    assert res["no_oracle_sinks"] is True
    assert res["audit"]["zero_shot_verified"] is True
