# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch18_sparse_memory.py — Unit tests for Chapter 18 sparse hippocampal memory.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch18_sparse_memory import (
    SparseHippocampalMemory,
    SparseMemoryConfig,
)


def test_sparse_memory_init_and_invariants():
    cfg = SparseMemoryConfig(num_entities=1000, mode="fp32")
    mem = SparseHippocampalMemory(cfg)
    assert mem.num_entities == 1000
    assert mem.assert_invariants() is True


def test_sparse_relation_registration_and_normalization():
    cfg = SparseMemoryConfig(num_entities=10, mode="fp32")
    mem = SparseHippocampalMemory(cfg)

    # 3 outgoing edges from source node 0 to targets 1, 2, 3
    src = torch.tensor([0, 0, 0], dtype=torch.long)
    tgt = torch.tensor([1, 2, 3], dtype=torch.long)
    edge_index = torch.stack([tgt, src], dim=0)

    mem.register_sparse_relation(0, edge_index)
    W_sp = mem.sparse_edges[0]
    vals = W_sp.values()

    # Outgoing weights from node 0 should sum to 1.0 (each is 1/3)
    assert (vals >= 0.0).all()
    assert abs(vals.sum().item() - 1.0) < 1e-6
    assert mem.assert_invariants() is True


def test_sparse_vs_dense_propagation_exact():
    torch.manual_seed(42)
    N = 60
    E = 200
    src = torch.randint(0, N, (E,), dtype=torch.long)
    tgt = torch.randint(0, N, (E,), dtype=torch.long)
    weights = torch.rand(E) + 0.1

    # Dense reference
    W_dense = torch.zeros(N, N)
    for s, t, w in zip(src, tgt, weights):
        W_dense[t, s] += w
    col_sums = W_dense.sum(dim=0, keepdim=True)
    W_dense = torch.where(col_sums > 0, W_dense / col_sums, W_dense)

    # Sparse memory
    cfg = SparseMemoryConfig(num_entities=N, mode="fp32")
    mem = SparseHippocampalMemory(cfg)
    edge_index = torch.stack([tgt, src], dim=0)
    mem.register_sparse_relation(0, edge_index, edge_weight=weights)

    # Test propagation from multiple origin nodes
    for origin in [0, 15, 42]:
        h_dense = torch.zeros(N)
        h_dense[origin] = 1.0
        h_sparse = torch.zeros(N)
        h_sparse[origin] = 1.0

        for _ in range(5):
            h_dense = W_dense @ h_dense
            h_sparse = mem.propagate_sparse_relation(0, h_sparse)

        max_err = torch.max(torch.abs(h_sparse - h_dense)).item()
        assert max_err < 1e-6


def test_forward_multi_hop_chain():
    cfg = SparseMemoryConfig(num_entities=100, mode="fp32")
    mem = SparseHippocampalMemory(cfg)

    # 4-hop chain: 10 -> 20 -> 30 -> 40 -> 50 across relation 0
    src = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    tgt = torch.tensor([20, 30, 40, 50], dtype=torch.long)
    edge_index = torch.stack([tgt, src], dim=0)
    mem.register_sparse_relation(0, edge_index)

    out = mem.forward_multi_hop_chain(start_entity=10, relation_sequence=[0, 0, 0, 0])
    top_node = torch.argmax(out).item()
    assert top_node == 50
    assert abs(out[50].item() - 1.0) < 1e-6


def test_memory_footprint_scaling():
    cfg = SparseMemoryConfig(num_entities=100_000, mode="fp32")
    mem = SparseHippocampalMemory(cfg)

    # Ingest 10,000 edges
    src = torch.arange(0, 10_000, dtype=torch.long)
    tgt = src + 1
    edge_index = torch.stack([tgt, src], dim=0)
    mem.register_sparse_relation(0, edge_index)

    ram_bytes = mem.memory_footprint_bytes()
    dense_bytes = mem.dense_footprint_bytes()

    assert ram_bytes == 10_000 * 20  # 200,000 bytes (< 0.2 MB)
    assert dense_bytes == 100_000 * 100_000 * 4  # 40 GB
    assert dense_bytes / ram_bytes == 200_000.0


def test_emergent_multihop_halting():
    cfg = SparseMemoryConfig(num_entities=1000, mode="fp32")
    mem = SparseHippocampalMemory(cfg)

    # 3-hop chain ending in sink node 4
    src = torch.tensor([1, 2, 3], dtype=torch.long)
    tgt = torch.tensor([2, 3, 4], dtype=torch.long)
    edge_index = torch.stack([tgt, src], dim=0)
    mem.register_sparse_relation(0, edge_index)

    out, hops = mem.forward_emergent_multihop(start_entity=1, relation_id=0, max_hops=10, gamma_sharpen=2.0)
    assert hops == 3
    assert torch.argmax(out).item() == 4
    assert out[4].item() >= 0.99


def test_sdk_parity():
    from geoformer.layers.sparse_memory import SparseHippocampalRelationalMemory as SDKMemory

    N = 500
    sdk_mem = SDKMemory(d_model=64, num_entities=N, mode="fp32")
    book_cfg = SparseMemoryConfig(num_entities=N, mode="fp32")
    book_mem = SparseHippocampalMemory(book_cfg)

    src = torch.tensor([2, 4, 8], dtype=torch.long)
    tgt = torch.tensor([4, 8, 16], dtype=torch.long)
    edge_index = torch.stack([tgt, src], dim=0)

    sdk_mem.register_sparse_relation(0, edge_index)
    book_mem.register_sparse_relation(0, edge_index)

    h_sdk = sdk_mem.forward_multi_hop_chain(start_entity=2, relation_sequence=[0, 0, 0])
    h_book = book_mem.forward_multi_hop_chain(start_entity=2, relation_sequence=[0, 0, 0])

    assert torch.allclose(h_sdk, h_book, atol=1e-6)
    assert torch.argmax(h_sdk).item() == 16
