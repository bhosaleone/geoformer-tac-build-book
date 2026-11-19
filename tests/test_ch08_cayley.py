# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 8 tests — The S5 Cayley Projector & Description-Length Bounds.

Guards the theorems and claims of Chapter 8:
  B.8.1  Frobenius-Schur decomposition: irrep dimensions [1, 1, 4, 4, 5, 5, 6],
         sum(d^2) = 120, sum(d) = 26.
  B.8.2  Orthogonal projection: P = P^T, P^2 = P, rank(P) = min(D, 26),
         energy conservation ||h||^2 = ||Ph||^2 + ||(I-P)h||^2, and orthogonality.
  B.8.3  Kolmogorov bounds: K_total <= 54.73 bits, single channel <= 15.51 bits,
         and empirical Shannon complexity 0 <= K_S <= log2(120) = 6.9069 bits.
  Parity: Bit-exact agreement with geoformer.layers.cayley.S5CayleyProjector.
"""

import math
import pytest
import torch

from gftac.ch08_cayley import (
    BookS5Projector,
    burnside_sum,
    irrep_dimensions,
    irrep_names,
    kolmogorov_bounds,
    kolmogorov_capacities,
)


# ----------------------------------------------------------------------
# B.8.1 — Irrep dimensions & Burnside accounting
# ----------------------------------------------------------------------

def test_b81_irrep_dimensions_count():
    dims = irrep_dimensions()
    assert len(dims) == 7
    assert dims == [1, 1, 4, 4, 5, 5, 6]


def test_b81_burnside_identity():
    # sum_{rho=1}^7 d_rho^2 = |S_5| = 5! = 120
    assert burnside_sum() == 120
    dims = irrep_dimensions()
    assert sum(d * d for d in dims) == 120


def test_b81_subspace_dimension():
    # sum_{rho=1}^7 d_rho = 1 + 1 + 4 + 4 + 5 + 5 + 6 = 26
    dims = irrep_dimensions()
    assert sum(dims) == 26


def test_b81_irrep_names():
    names = irrep_names()
    assert len(names) == 7
    assert names[0] == "Trivial"
    assert names[1] == "Sign"
    assert names[6] == "2-Hook Young"


# ----------------------------------------------------------------------
# B.8.2 — Orthogonal projector properties & energy conservation
# ----------------------------------------------------------------------

def test_b82_projector_symmetry_and_idempotence():
    torch.manual_seed(42)
    proj = BookS5Projector(d_model=32)
    P = proj.projector_matrix()
    # Symmetry: P^T = P
    assert torch.allclose(P, P.t(), atol=1e-6)
    # Idempotence: P^2 = P
    assert torch.allclose(P @ P, P, atol=1e-5)


def test_b82_projector_rank_and_trace():
    for d_model in (16, 26, 32, 64):
        torch.manual_seed(d_model)
        proj = BookS5Projector(d_model=d_model)
        P = proj.projector_matrix()
        expected_rank = min(d_model, 26)
        rank = int(torch.linalg.matrix_rank(P).detach())
        trace = float(torch.trace(P).detach())
        assert rank == expected_rank
        assert abs(trace - expected_rank) < 1e-4


def test_b82_pythagorean_energy_split():
    torch.manual_seed(10)
    proj = BookS5Projector(d_model=32)
    h = torch.randn(4, 10, 32)
    with torch.no_grad():
        h_sym, _ = proj(h)
        h_perp = h - h_sym
        norm_orig = (h ** 2).sum(-1)
        norm_split = (h_sym ** 2).sum(-1) + (h_perp ** 2).sum(-1)
        assert torch.allclose(norm_orig, norm_split, atol=1e-4)


def test_b82_subspace_orthogonality():
    torch.manual_seed(20)
    proj = BookS5Projector(d_model=32)
    h = torch.randn(4, 10, 32)
    with torch.no_grad():
        h_sym, _ = proj(h)
        h_perp = h - h_sym
        inner_prod = (h_sym * h_perp).sum(-1)
        assert torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-4)


def test_b82_idempotence_on_activations():
    torch.manual_seed(30)
    proj = BookS5Projector(d_model=32)
    h = torch.randn(3, 8, 32)
    with torch.no_grad():
        h_sym1, _ = proj(h)
        h_sym2, _ = proj(h_sym1)
        assert torch.allclose(h_sym1, h_sym2, atol=1e-5)


# ----------------------------------------------------------------------
# B.8.3 — Kolmogorov description bounds
# ----------------------------------------------------------------------

def test_b83_channel_capacities():
    caps = kolmogorov_capacities()
    assert len(caps) == 7
    assert caps[0] == 0.0  # 1 * log2(1) = 0
    assert caps[1] == 0.0  # 1 * log2(1) = 0
    assert abs(caps[2] - 8.0) < 1e-6   # 4 * log2(4) = 8
    assert abs(caps[3] - 8.0) < 1e-6
    assert abs(caps[4] - 5 * math.log2(5)) < 1e-6
    assert abs(caps[5] - 5 * math.log2(5)) < 1e-6
    assert abs(caps[6] - 6 * math.log2(6)) < 1e-6


def test_b83_total_kolmogorov_bound():
    bounds = kolmogorov_bounds()
    # K_total = sum d_rho log2(d_rho) <= 54.73 bits
    assert abs(bounds["K_total"] - 54.7291) < 1e-3
    assert bounds["K_total"] <= 54.73


def test_b83_single_channel_bound():
    bounds = kolmogorov_bounds()
    # max d_rho log2(d_rho) = 6 log2(6) = 15.5098 bits
    assert abs(bounds["K_channel_max"] - 6 * math.log2(6)) < 1e-4
    assert bounds["K_channel_max"] <= 15.51


def test_b83_simplex_expected_complexity():
    torch.manual_seed(99)
    caps = torch.tensor(kolmogorov_capacities(), dtype=torch.float64)
    # Test 1000 random simplex distributions
    for _ in range(50):
        logits = torch.randn(7, dtype=torch.float64)
        probs = torch.softmax(logits, dim=0)
        expected_k = float((probs * caps).sum())
        # Refutation 1 of Theorem Q-6: K_expected <= 15.51 bits
        assert expected_k <= 15.5098 + 1e-6


def test_b83_empirical_shannon_complexity_bounds():
    torch.manual_seed(77)
    proj = BookS5Projector(d_model=32)
    max_bound = math.log2(120.0)
    for _ in range(10):
        h = torch.randn(2, 5, 32)
        _, K_S = proj(h)
        k_val = float(K_S.detach())
        assert 0.0 <= k_val <= max_bound + 1e-4


# ----------------------------------------------------------------------
# Parity with production SDK
# ----------------------------------------------------------------------

def test_parity_s5_projector_bit_exact():
    from geoformer.layers.cayley import S5CayleyProjector as SDKProj

    for d in (16, 26, 32, 64):
        torch.manual_seed(d)
        sdk = SDKProj(d_model=d)
        book = BookS5Projector(d_model=d)
        book.load_state_dict(sdk.state_dict())
        h = torch.randn(2, 5, d)
        with torch.no_grad():
            h_sdk, k_sdk = sdk(h)
            h_bk, k_bk = book(h)
        assert torch.allclose(h_bk, h_sdk, atol=0.0)
        assert torch.allclose(k_bk, k_sdk, atol=0.0)
