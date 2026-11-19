# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 2 tests — the lattice's invariants are the chapter's claims.

Each test pins a statement made in chapters/ch02_metric_lattice.md:
the quantization invariant battery, Theorem B.2.1 (contraction),
shadow-buffer accumulation, stochastic unbiasedness, Theorem B.2.2 (PSD
and the pairwise identity the proof is made of), connectivity via the
Fiedler value, Theorem B.2.3 (the floor and its tightness), and the
dead-end policy of the directed kernel.
"""

import statistics

import torch

from gftac.ch02_metric_lattice import MetricLattice


# --- Storage invariant ------------------------------------------------------

def test_lattice_is_quantized_after_arbitrary_writes():
    """Invariant battery: W stays on the lattice under any write pattern."""
    lat = MetricLattice(num_entities=8, w_max=1.0, bits=4)
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        i = int(torch.randint(0, 8, (1,), generator=g).item())
        j = int(torch.randint(0, 8, (1,), generator=g).item())
        if i == j:
            continue
        w = float(torch.rand((1,), generator=g).item()) * 0.4
        stochastic = bool(torch.rand((1,), generator=g).item() < 0.5)
        lat.observe_edge(i, j, weight=w, stochastic=stochastic)
    assert lat.is_quantized()


# --- Theorem B.2.1: contraction ----------------------------------------------

def test_hebbian_contraction_strict_until_saturation():
    """Every dw-write strictly lowers resistance until w_max is reached."""
    lat = MetricLattice(num_entities=4, w_max=1.0, bits=4)
    assert lat.level(0, 1) == 0
    resistances = [lat.resistance(0, 1)]
    for _ in range(lat.levels):
        lat.observe_edge(0, 1, weight=lat.delta_w)
        resistances.append(lat.resistance(0, 1))
    for a, b in zip(resistances, resistances[1:]):
        assert b < a                       # strictly decreasing, every write
    assert lat.level(0, 1) == lat.levels   # saturated at the top level
    d = lat.resistance(0, 1)
    assert abs(d - 1.0 / (lat.w_max + lat.epsilon)) < 1e-9
    lat.observe_edge(0, 1, weight=lat.delta_w)   # further writes change nothing
    assert lat.resistance(0, 1) == d


def test_subthreshold_increments_accumulate():
    """The shadow buffer: writes smaller than dw/2 are not lost forever."""
    lat = MetricLattice(num_entities=2, w_max=1.0, bits=4)
    dw = lat.delta_w
    for _ in range(75):
        lat.observe_edge(0, 1, weight=dw / 10.0)
    # 75 x dw/10 = 7.5 dw of accumulated Hebbian mass: level 7 or 8
    assert 7 * dw <= lat.W[0, 1].item() <= 8 * dw
    assert lat.level(0, 1) in (7, 8)


def test_stochastic_rounding_is_unbiased():
    """E[stochastic projection] == the exact accumulated weight."""
    torch.manual_seed(0)
    levels_observed = []
    for _ in range(4000):
        lat = MetricLattice(num_entities=2, w_max=1.0, bits=4)
        lat.observe_edge(0, 1, weight=0.37 * lat.delta_w, stochastic=True)
        levels_observed.append(lat.W[0, 1].item() / lat.delta_w)
    mean = statistics.fmean(levels_observed)
    assert abs(mean - 0.37) < 0.05


# --- Theorem B.2.2: the Laplacian --------------------------------------------

def test_laplacian_psd_and_pairwise_identity():
    """L is PSD, and the quadratic form equals the pairwise sum of the proof."""
    lat = MetricLattice(num_entities=6, w_max=1.0, bits=4)
    g = torch.Generator().manual_seed(11)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (0, 4), (2, 5)]
    for i, j in edges:
        lat.observe_edge(i, j, weight=float(torch.rand((1,), generator=g).item()))
    L = lat.laplacian()
    evals = torch.linalg.eigvalsh(L)
    assert evals.min().item() >= -1e-6                      # B.2.2: PSD

    # The identity the proof is made of, verified numerically:
    x = torch.randn(6, generator=g)
    qf = (x @ L @ x).item()
    A = lat.W + lat.W.T
    diff2 = (x[:, None] - x[None, :]) ** 2
    pairwise = (A * diff2).sum().item() / 2.0
    assert abs(qf - pairwise) < 1e-5 * max(1.0, abs(qf))


def test_fiedler_zero_iff_disconnected():
    """lambda_2 > 0 exactly when the symmetrized store is connected."""
    connected = MetricLattice(num_entities=4)
    connected.observe_edges([(0, 1), (1, 2), (2, 3)])
    assert connected.fiedler_value() > 1e-6
    split = MetricLattice(num_entities=4)
    split.observe_edges([(0, 1), (2, 3)])
    assert abs(split.fiedler_value()) < 1e-6


# --- Theorem B.2.3: the Fiedler floor ----------------------------------------

def test_fiedler_bound_on_dense_lattice():
    """Complete min-conductance lattices: lambda_2 >= N * w_min."""
    lat = MetricLattice(num_entities=16, w_max=1.0, bits=4)
    g = torch.Generator().manual_seed(3)
    for i in range(16):
        for j in range(16):
            if i != j:
                lv = int(torch.randint(1, lat.levels + 1, (1,), generator=g).item())
                lat.W[i, j] = lv * lat.delta_w
    w_min = 2 * lat.delta_w   # every unordered pair carries >= 2*dw
    assert lat.fiedler_value() >= lat.n * w_min - 1e-6


def test_fiedler_bound_is_tight():
    """The uniform complete lattice attains the floor exactly."""
    lat = MetricLattice(num_entities=6, w_max=1.0, bits=4)
    for i in range(6):
        for j in range(6):
            if i != j:
                lat.W[i, j] = lat.delta_w
    w_min = 2 * lat.delta_w
    assert abs(lat.fiedler_value() - 6 * w_min) < 1e-6


# --- The directed kernel ------------------------------------------------------

def test_directed_kernel_rows_and_dead_ends():
    """Live rows are stochastic; dead ends are explicit, not hidden."""
    lat = MetricLattice(num_entities=5, w_max=1.0, bits=4)
    lat.observe_edges([(0, 1), (1, 2), (2, 3)])
    P, dead = lat.directed_kernel()
    live = ~dead
    assert torch.allclose(P[live].sum(dim=1), torch.ones(int(live.sum())))
    assert P[dead].abs().sum() == 0.0
    assert dead.tolist() == [False, False, False, True, True]
    assert P.diagonal().abs().sum() == 0.0     # no self-loops
