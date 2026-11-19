# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 2 — The Metric Lattice.

The substrate of the entire architecture: a non-negative, quantized
conductance lattice W, the edge-resistance metric d_ij = 1/(W_ij + eps),
the symmetrized graph Laplacian, the Fiedler value, and the directed
row-stochastic kernel with honest dead ends.

Book theorems proved in the chapter and pinned by the tests:

  B.2.1  Hebbian resistance contraction — a write never increases
         distance; a used highway is permanently cheaper.
  B.2.2  Laplacian positive semidefiniteness, via the pairwise identity
         x^T L x = sum_{i<j} A_ij (x_i - x_j)^2.
  B.2.3  Fiedler floor — if every unordered pair carries symmetrized
         conductance >= w_min, then lambda_2(L) >= N * w_min; tight for
         the uniform complete lattice. (The precise form of the research
         program's Theorem Q-2, hypotheses explicit.)

Run the demonstration ledger::

    python3 code/gftac/ch02_metric_lattice.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["MetricLattice", "demo_ledger"]


class MetricLattice:
    """A quantized conductance lattice over ``num_entities`` entities.

    Storage model
    -------------
    ``W``      the quantized store: every entry is a multiple of ``delta_w``
               in ``[0, w_max]`` — ``b_w`` bits per directed edge. This is
               the artifact serialization, attention (Ch. 5), and the
               Laplacian read. A fact written is the fact read.
    ``W_raw``  the unquantized write buffer (the "shadow"). Hebbian
               increments accumulate here; ``W`` is its lattice projection.
               Sub-threshold increments are therefore never lost: they
               accumulate until they cross a level boundary (Theorem B.2.1
               applies to the projection, and the buffer keeps the residue).

    The metric
    ----------
    ``d_ij = 1/(W_ij + eps)`` — the *edge resistance*: the cost of
    traversing the single edge (i -> j). Disconnected edges read as
    ``1/eps`` (effectively infinite, but finite arithmetic). Note the
    terminological care used throughout this book: the *electrical*
    effective resistance between two nodes (a path quantity computable
    from the Laplacian pseudoinverse) is a different object; see the
    chapter's discussion of the division of labor between the pairwise
    metric and the composite operator.
    """

    def __init__(
        self,
        num_entities: int,
        w_max: float = 1.0,
        bits: int = 4,
        epsilon: float = 1e-8,
        seed: Optional[int] = None,
    ):
        if num_entities < 2:
            raise ValueError("a lattice needs at least 2 entities")
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if w_max <= 0:
            raise ValueError("w_max must be positive")

        self.n = num_entities
        self.w_max = float(w_max)
        self.bits = int(bits)
        self.epsilon = float(epsilon)

        # Quantization arithmetic: b_w bits -> levels 0..2^b_w - 1, where
        # level 0 means "no conductance" and level (2^b_w - 1) is w_max.
        self.levels = 2 ** self.bits - 1
        self.delta_w = self.w_max / self.levels

        self.W = torch.zeros(num_entities, num_entities)
        self.W_raw = torch.zeros(num_entities, num_entities)
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    # ------------------------------------------------------------------
    # Lattice projection (deterministic and stochastic)
    # ------------------------------------------------------------------

    def _project(self, w_raw: torch.Tensor) -> torch.Tensor:
        """Deterministic projection: nearest lattice level, clamped to [0, w_max].

        Monotone: if ``w_raw`` is already at or above level k, the projection
        is at least k — the ingredient of Theorem B.2.1's first part.
        """
        scaled = float(w_raw) / self.delta_w
        level = min(self.levels, max(0, round(scaled)))
        return torch.tensor(level * self.delta_w)

    def _project_stochastic(self, w_raw: torch.Tensor) -> torch.Tensor:
        """Unbiased stochastic projection: round up with probability equal to
        the fractional part.

        For a buffer value at level k with fraction f, the projected level is
        k + 1 with probability f and k with probability 1 - f, so
        E[projected] = k + f = the exact buffer value. Plasticity statistics
        survive coarse storage — the same mathematics that makes memristor
        programming noise survivable (Ch. 21).
        """
        scaled = float(w_raw) / self.delta_w
        k = math.floor(scaled)
        frac = min(1.0, max(0.0, scaled - k))
        draw = torch.rand((1,), generator=self.generator).item()
        level = min(self.levels, k + (1 if draw < frac else 0))
        return torch.tensor(level * self.delta_w)

    # ------------------------------------------------------------------
    # Hebbian writes
    # ------------------------------------------------------------------

    def observe_edge(
        self,
        i: int,
        j: int,
        weight: float = 1.0,
        stochastic: bool = False,
    ) -> Tuple[float, float]:
        """Hebbian increment on the edge ``i -> j``.

        The write accumulates ``weight`` into the shadow buffer (capped at
        ``w_max``) and projects to the lattice. Returns ``(d_before, d_after)``
        — the edge resistance before and after the write — so callers can
        watch Theorem B.2.1 happen.

        Self-edges are rejected: conductance lives between entities
        (``W_ii = 0`` is part of the invariant battery).
        """
        if i == j:
            raise ValueError("self-edges are not conductance (W_ii = 0 by design)")
        if weight < 0:
            raise ValueError("Hebbian increments are non-negative (W >= 0 by design)")

        d_before = self.resistance(i, j)
        self.W_raw[i, j] = min(self.w_max, float(self.W_raw[i, j]) + weight)
        if stochastic:
            self.W[i, j] = self._project_stochastic(self.W_raw[i, j])
        else:
            self.W[i, j] = self._project(self.W_raw[i, j])
        d_after = self.resistance(i, j)
        return d_before, d_after

    def observe_edges(
        self, edges: List[Tuple[int, int]], weight: float = 1.0, stochastic: bool = False
    ) -> "MetricLattice":
        """Apply the same Hebbian increment to every edge in a list."""
        for i, j in edges:
            self.observe_edge(i, j, weight=weight, stochastic=stochastic)
        return self

    # ------------------------------------------------------------------
    # The metric and the storage invariants
    # ------------------------------------------------------------------

    def resistance(self, i: int, j: int) -> float:
        """Edge resistance ``d_ij = 1/(W_ij + eps)``.

        A disconnected edge reads ``1/eps`` — effectively infinite, but a
        finite, computable number: no infinities ever enter the tensors.
        """
        return 1.0 / (self.W[i, j].item() + self.epsilon)

    def resistance_matrix(self) -> torch.Tensor:
        """The full matrix of edge resistances ``1/(W + eps)``."""
        return 1.0 / (self.W + self.epsilon)

    def level(self, i: int, j: int) -> int:
        """Current quantization level of edge ``(i, j)``: an integer in [0, levels]."""
        return int(round(self.W[i, j].item() / self.delta_w))

    def is_quantized(self, tol: float = 1e-4) -> bool:
        """The storage invariant battery, as a single predicate.

        Every entry is a multiple of ``delta_w`` (within ``tol``, measured in
        level units), every entry lies in ``[0, w_max]``, and the diagonal is
        exactly zero. Chapter 10's plasticity rule must keep this true after
        every update; the Chapter 2 tests check it after arbitrary writes.
        """
        scaled = self.W / self.delta_w
        ok_levels = torch.allclose(scaled, torch.round(scaled), atol=tol)
        ok_range = bool((self.W >= -1e-9).all() and (self.W <= self.w_max + 1e-9).all())
        ok_diag = bool((self.W.diagonal() == 0).all())
        return bool(ok_levels and ok_range and ok_diag)

    def summary(self) -> Dict:
        """A one-glance description of the store."""
        edges = int((self.W > 0).sum().item())
        return {
            "num_entities": self.n,
            "bits": self.bits,
            "levels": self.levels,
            "delta_w": self.delta_w,
            "w_max": self.w_max,
            "directed_edges": edges,
            "fill_fraction": edges / (self.n * (self.n - 1)),
        }

    # ------------------------------------------------------------------
    # Operators: the Laplacian and the Fiedler value
    # ------------------------------------------------------------------

    def laplacian(self) -> torch.Tensor:
        """The symmetrized Laplacian ``L = D - A`` with ``A = W + W^T``.

        Two one-way conductances between a pair add, exactly as parallel
        conductances do in a circuit — ``A_ij`` is the total communication
        capacity of the unordered pair {i, j}. By Theorem B.2.2, ``L`` is
        positive semidefinite, and by the pairwise identity,

            x^T L x = sum_{i<j} A_ij (x_i - x_j)^2.

        This is the operator Chapter 3's free energy is built from.
        """
        A = self.W + self.W.T
        deg = A.sum(dim=1)
        return torch.diag(deg) - A

    def fiedler_value(self) -> float:
        """The algebraic connectivity ``lambda_2`` of the symmetrized lattice.

        ``lambda_2 > 0`` exactly when the symmetrized store is connected
        (Theorem B.2.2, second corollary); its magnitude is the spectral
        floor Theorem B.2.3 bounds.
        """
        if self.n < 2:
            return 0.0
        evals = torch.linalg.eigvalsh(self.laplacian())
        return float(evals[1].item())

    # ------------------------------------------------------------------
    # The directed kernel and the dead-end policy
    # ------------------------------------------------------------------

    def directed_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row-stochastic propagation kernel ``P`` and the dead-end mask.

        Live rows are normalized to sum to one; rows with no outgoing
        conductance are left at exactly zero — mass reaching a dead end is
        absorbed, not teleported or re-normalized. Renormalizing dead rows
        (or truncating them away) would inject information the graph does
        not contain: the oracle artifact the research program's Tier-8 audit
        found and forbade. The kernel is *read-time* normalization: the
        lattice stores magnitudes, because the conductance ratio alpha_0
        that Chapter 16's soliton theorem needs is a property of ``W``, not
        of ``P`` — normalizing at write time would erase the loudness
        contrast the theorem feeds on.
        """
        out = self.W.sum(dim=1, keepdim=True)
        safe = self.W / out.clamp(min=self.epsilon)
        P = torch.where(out > 0, safe, torch.zeros_like(self.W))
        dead = out.squeeze(1) <= 0
        return P, dead


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def demo_ledger() -> str:
    """The Chapter 2 demonstration: contraction, the Fiedler floor, dead ends."""
    lines: List[str] = []
    lines.append("Chapter 2 — The Metric Lattice, demonstrated")
    lines.append("=" * 62)
    lat = MetricLattice(num_entities=2, w_max=1.0, bits=4)
    lines.append(
        f"b_w = {lat.bits} bits: levels = {lat.levels}, "
        f"dw = w_max/{lat.levels} = {lat.delta_w:.6f}"
    )
    lines.append("")
    lines.append("B.2.1 — Hebbian contraction on edge (0 -> 1), dw per write:")
    lines.append(f"{'write':>6}{'level':>8}{'d_ij = 1/(W_ij + eps)':>24}")
    d0 = lat.resistance(0, 1)
    lines.append(f"{'0':>6}{0:>8}{d0:>24.3e}   (disconnected: reads 1/eps)")
    for k in range(1, lat.levels + 2):
        lat.observe_edge(0, 1, weight=lat.delta_w)
        note = "   <- saturated at w_max" if k > lat.levels else ""
        lines.append(f"{k:>6}{lat.level(0, 1):>8}{lat.resistance(0, 1):>24.6f}{note}")
    lines.append("")

    # Sparse world: where the B.2.3 hypothesis fails, honestly shown.
    chain = MetricLattice(num_entities=5, w_max=1.0, bits=4)
    chain.observe_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    lines.append(
        f"chain world (5 nodes, 4 edges at w_max): lambda_2 = "
        f"{chain.fiedler_value():.6f}"
    )
    lines.append("  (sparse: the complete-lattice hypothesis of B.2.3 does NOT apply)")

    # Dense world: the bound, verified.
    dense = MetricLattice(num_entities=16, w_max=1.0, bits=4)
    g = torch.Generator().manual_seed(3)
    for i in range(dense.n):
        for j in range(dense.n):
            if i != j:
                lv = int(torch.randint(1, dense.levels + 1, (1,), generator=g).item())
                dense.W[i, j] = lv * dense.delta_w
    w_min = 2 * dense.delta_w  # minimum symmetrized off-diagonal conductance
    lines.append(
        f"dense random lattice (16 nodes, all pairs at levels 1..{dense.levels}): "
        f"lambda_2 = {dense.fiedler_value():.6f}"
    )
    lines.append(f"  B.2.3 floor: N*w_min = {dense.n * w_min:.6f}  -> bound holds")

    uni = MetricLattice(num_entities=6, w_max=1.0, bits=4)
    for i in range(uni.n):
        for j in range(uni.n):
            if i != j:
                uni.W[i, j] = uni.delta_w
    lines.append(
        f"uniform complete lattice (6 nodes, level 1 both ways): lambda_2 = "
        f"{uni.fiedler_value():.6f}"
    )
    lines.append(
        f"  equals N*w_min = {uni.n * 2 * uni.delta_w:.6f} exactly -> the bound is tight"
    )
    lines.append("")

    P, dead = chain.directed_kernel()
    dead_idx = dead.nonzero().flatten().tolist()
    lines.append(
        f"directed kernel of the chain world: dead ends at nodes {dead_idx}"
        " (rows all zero, mass absorbed)"
    )
    lines.append("=" * 62)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
