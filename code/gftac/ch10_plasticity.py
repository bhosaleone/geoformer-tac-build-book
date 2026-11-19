# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 10 — Two-Timescale Learning: Hebbian Plasticity with Invariants.

This module implements the fast-timescale relational learning substrate of GeoFormer:
the PlasticConductanceEngine, which governs fast Hebbian outer-product updates into the
quantized conductance lattice W while enforcing the full invariant battery of the
EndogenousConsolidator.

Theorems proved in Chapter 10:
  B.10.1  Hebbian Resistance Contraction & Monotone Metric Warping:
          An observed edge reinforcement W_{ij} -> W_{ij} + ΔW strictly contracts the
          geodesic resistance d_{ij} = 1/(W_{ij} + eps) monotonically, with contraction
          derivative ∂d_{ij}/∂W_{ij} = -1/(W_{ij} + eps)^2 < 0.
  B.10.2  Invariant Battery Preservation:
          Under arbitrary batched Hebbian updates, the lattice state W strictly satisfies:
          (1) W >= 0 (non-negativity),
          (2) diag(W) = 0 (hollow diagonal),
          (3) ||ΔW||_max <= ΔW_max (bounded step),
          (4) W_{ij} in {0, Δw, 2Δw, ..., w_max} (quantized level membership).
  B.10.3  Endogenous Weyl Consolidation & Bounded Dirichlet Energy:
          Background Weyl exploration over irrational golden driver alpha = (sqrt(5)-1)/2
          synthesizes higher-order transitive shortcut highways (W^m)_{ik} while keeping
          total Frobenius energy E = 0.5 * sum(W^2) strictly bounded and non-explosive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

__all__ = [
    "PlasticityConfig",
    "ConsolidationReport",
    "PlasticConductanceEngine",
    "demo_ledger",
]


@dataclass
class PlasticityConfig:
    """Configuration for two-timescale Hebbian plasticity and endogenous consolidation."""
    num_entities: int = 16
    w_max: float = 1.0
    bits: int = 4                       # 4-bit lattice: 15 levels
    epsilon: float = 1e-8
    max_delta_w: float = 0.2           # hard bound on |ΔW| per update event
    plastic_rate: float = 1.0          # Hebbian scaling factor
    decay: float = 1.0                 # 1.0 = no decay; < 1.0 = passive forgetting
    alpha: float = (5.0**0.5 - 1.0) / 2.0  # irrational golden driver ≈ 0.6180339887
    kappa_closed: float = 0.85        # trajectory closure threshold
    max_probes: int = 64              # Weyl probe budget per consolidation step
    max_hop_depth: int = 4            # maximum transitive exploration depth
    seed: Optional[int] = 42


@dataclass
class ConsolidationReport:
    """Diagnostic report emitted after background endogenous consolidation."""
    mean_kappa: float
    num_probes: int
    edges_updated: int
    max_abs_delta: float
    energy_before: float
    energy_after: float


class PlasticConductanceEngine:
    """The fast-timescale plastic conductance engine for GeoFormer.

    Separates fast, episodic relational plasticity (Hebbian outer products onto
    the metric lattice W >= 0) from slow gradient-based parametric learning.
    Enforces the complete invariant battery on every single modification.
    """

    def __init__(self, config: Optional[PlasticityConfig] = None, **kwargs):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = PlasticityConfig(**kwargs)

        if self.cfg.num_entities < 2:
            raise ValueError("Lattice requires at least 2 entities")
        if self.cfg.bits < 1:
            raise ValueError("Lattice quantization bits must be >= 1")
        if self.cfg.w_max <= 0:
            raise ValueError("w_max must be positive")

        self.n = self.cfg.num_entities
        self.w_max = float(self.cfg.w_max)
        self.bits = int(self.cfg.bits)
        self.epsilon = float(self.cfg.epsilon)
        self.max_delta_w = float(self.cfg.max_delta_w)

        # Quantization levels: 2^bits - 1 steps in [0, w_max]
        self.levels = (2 ** self.bits) - 1
        self.delta_w = self.w_max / float(self.levels)

        # Storage: quantized store W and unquantized shadow write buffer W_raw
        self.W = torch.zeros(self.n, self.n, dtype=torch.float32)
        self.W_raw = torch.zeros(self.n, self.n, dtype=torch.float32)

        # Weyl driver state for endogenous consolidation
        seed_val = self.cfg.seed if self.cfg.seed is not None else 0
        self._z = float(seed_val % 1000) * 0.00137
        if self._z <= 0:
            self._z = 0.123456789

        self.history: List[Dict[str, Any]] = []

    def quantize(self, raw_val: float, stochastic: bool = False) -> float:
        """Projects a continuous non-negative value onto the discrete lattice levels."""
        clamped = max(0.0, min(self.w_max, raw_val))
        step = clamped / self.delta_w
        if stochastic:
            base = math.floor(step)
            frac = step - base
            # Stochastic unbiased projection
            rand_val = torch.rand(1).item()
            k = base + 1 if rand_val < frac else base
            k = min(self.levels, max(0, k))
            return float(k * self.delta_w)
        else:
            k = round(step)
            k = min(self.levels, max(0, k))
            return float(k * self.delta_w)

    def update_conductance(
        self,
        src_indices: Union[int, List[int], torch.Tensor],
        tgt_indices: Union[int, List[int], torch.Tensor],
        weight_inc: float = 1.0,
        stochastic: bool = False,
    ) -> int:
        """Applies fast Hebbian plastic updates along observed edges.

        Args:
            src_indices: source entity index or batch of indices
            tgt_indices: target entity index or batch of indices
            weight_inc: scalar strength multiplier (default 1.0)
            stochastic: whether to use unbiased stochastic rounding

        Returns:
            Number of distinct edges updated.
        """
        if isinstance(src_indices, int):
            src_list = [src_indices]
        elif isinstance(src_indices, torch.Tensor):
            src_list = src_indices.detach().cpu().view(-1).tolist()
        else:
            src_list = list(src_indices)

        if isinstance(tgt_indices, int):
            tgt_list = [tgt_indices]
        elif isinstance(tgt_indices, torch.Tensor):
            tgt_list = tgt_indices.detach().cpu().view(-1).tolist()
        else:
            tgt_list = list(tgt_indices)

        if len(src_list) != len(tgt_list):
            raise ValueError(f"Source length {len(src_list)} != Target length {len(tgt_list)}")

        # Step increment: scaled by delta_w and plastic_rate, clamped by max_delta_w
        raw_inc = self.delta_w * self.cfg.plastic_rate * float(weight_inc)
        bounded_inc = min(self.max_delta_w, max(-self.max_delta_w, raw_inc))

        updated_count = 0
        with torch.no_grad():
            for s, t in zip(src_list, tgt_list):
                s_idx, t_idx = int(s), int(t)
                if s_idx == t_idx:
                    continue  # Invariant: hollow diagonal (W_ii = 0)
                if not (0 <= s_idx < self.n and 0 <= t_idx < self.n):
                    raise IndexError(f"Entity indices ({s_idx}, {t_idx}) out of bounds [0, {self.n})")

                cur_raw = float(self.W_raw[s_idx, t_idx].item())
                new_raw = max(0.0, min(self.w_max, cur_raw + bounded_inc))
                self.W_raw[s_idx, t_idx] = new_raw

                new_quant = self.quantize(new_raw, stochastic=stochastic)
                if abs(new_quant - float(self.W[s_idx, t_idx].item())) > 1e-7:
                    updated_count += 1
                self.W[s_idx, t_idx] = new_quant

            # Guarantee diagonal is strictly zero
            self.W.fill_diagonal_(0.0)
            self.W_raw.fill_diagonal_(0.0)

        return updated_count

    def compute_kappa(self, trajectory: torch.Tensor) -> float:
        """Computes the topological trajectory closure index kappa in [0, 1].

        kappa = 1 - (final_dist / max_excursion).
        When the trajectory forms a closed loop, final_dist -> 0 and kappa -> 1.
        """
        if trajectory.dim() == 1:
            trajectory = trajectory.unsqueeze(1)
        T = trajectory.shape[0]
        if T < 2:
            return 1.0

        p0 = trajectory[0]
        diffs = trajectory - p0
        dists = torch.norm(diffs, p=2, dim=-1)

        max_excursion = dists.max().item()
        final_dist = dists[-1].item()

        if max_excursion < 1e-12:
            return 1.0

        kappa = 1.0 - (final_dist / max_excursion)
        return float(max(0.0, min(1.0, kappa)))

    def endogenous_step(
        self,
        num_steps: int = 1,
        max_probes: Optional[int] = None,
        max_hop_depth: Optional[int] = None,
        kappa_closed: Optional[float] = None,
    ) -> ConsolidationReport:
        """Performs background endogenous consolidation using Weyl irrational probes.

        Discovers open multi-hop relational paths and reinforces direct shortcut
        conductances, transforming deep multi-hop reasoning into direct 1-hop highways.
        """
        probes_limit = max_probes if max_probes is not None else self.cfg.max_probes
        hop_limit = max_hop_depth if max_hop_depth is not None else self.cfg.max_hop_depth
        k_closed = kappa_closed if kappa_closed is not None else self.cfg.kappa_closed

        energy_before = 0.5 * float(torch.sum(self.W ** 2).item())
        initial_W = self.W.clone()

        total_probes = 0
        edges_updated = 0
        kappa_history: List[float] = []
        max_abs_delta = 0.0

        with torch.no_grad():
            for s in range(num_steps):
                step_probes = 0
                while step_probes < probes_limit and self.n > 2:
                    step_probes += 1
                    total_probes += 1

                    # 1. Weyl rotation to select candidate entity pair (i, k)
                    self._z = (self._z + self.cfg.alpha) % 1.0
                    i = int(math.floor(self._z * self.n)) % self.n

                    self._z = (self._z + self.cfg.alpha) % 1.0
                    k = int(math.floor(self._z * (self.n - 1)))
                    if k >= i:
                        k += 1

                    # 2. Check 2-hop bridge strength
                    w_row_i = self.W[i, :]
                    w_col_k = self.W[:, k]
                    two_hop = w_row_i * w_col_k
                    two_hop[i] = 0.0
                    two_hop[k] = 0.0

                    best_val, best_j = torch.max(two_hop, dim=0)
                    s_bridge = float(best_val.item())
                    j = int(best_j.item())
                    best_hop = 2

                    # Multi-hop search if 2-hop is absent
                    if s_bridge < 1e-4 and hop_limit > 2:
                        v = torch.zeros(self.n)
                        v[i] = 1.0
                        for m in range(2, min(hop_limit + 1, 9)):
                            v = torch.matmul(self.W.T, v)
                            max_v = v.max().item()
                            if max_v > 1e-5:
                                v = v / max_v
                            cand_s = float(self.W[:, k].dot(v).item())
                            if cand_s > s_bridge:
                                s_bridge = cand_s
                                best_hop = m

                    # 3. Trajectory closure
                    w_ij = float(self.W[i, j].item())
                    w_jk = float(self.W[j, k].item())
                    w_ik = float(self.W[i, k].item())
                    traj = torch.tensor([
                        [0.0, 0.0],
                        [w_ij, 0.0],
                        [w_ij, w_jk],
                        [w_ik, 0.0],
                    ], dtype=torch.float32)
                    k_val = self.compute_kappa(traj)
                    kappa_history.append(k_val)

                    # 4. If path is open and bridge is stronger than direct edge
                    if s_bridge > 1e-4 and w_ik < s_bridge:
                        raw_delta = (s_bridge - w_ik) * (0.1 / math.sqrt(best_hop - 1))
                        b_delta = min(self.max_delta_w, max(-self.max_delta_w, raw_delta))
                        new_raw = max(0.0, min(self.w_max, w_ik + b_delta))
                        new_quant = self.quantize(new_raw)
                        delta_eff = abs(new_quant - w_ik)
                        if delta_eff > 1e-7:
                            self.W[i, k] = new_quant
                            self.W_raw[i, k] = new_raw
                            edges_updated += 1
                            max_abs_delta = max(max_abs_delta, delta_eff)

                # Mild decay if configured
                if self.cfg.decay < 1.0:
                    self.W = self.W * self.cfg.decay
                    self.W_raw = self.W_raw * self.cfg.decay
                    # Re-quantize
                    for r in range(self.n):
                        for c in range(self.n):
                            if r != c:
                                self.W[r, c] = self.quantize(float(self.W_raw[r, c].item()))

                self.W.fill_diagonal_(0.0)
                self.W_raw.fill_diagonal_(0.0)

        energy_after = 0.5 * float(torch.sum(self.W ** 2).item())
        mean_k = float(sum(kappa_history) / len(kappa_history)) if kappa_history else 1.0

        return ConsolidationReport(
            mean_kappa=mean_k,
            num_probes=total_probes,
            edges_updated=edges_updated,
            max_abs_delta=max_abs_delta,
            energy_before=energy_before,
            energy_after=energy_after,
        )

    def resistance_matrix(self) -> torch.Tensor:
        """Computes pairwise edge resistance matrix d_{ij} = 1 / (W_{ij} + eps)."""
        return 1.0 / (self.W + self.epsilon)

    def resistance(self, i: int, j: int) -> float:
        """Effective resistance of directed edge (i -> j)."""
        return float(1.0 / (self.W[i, j].item() + self.epsilon))

    def laplacian(self) -> torch.Tensor:
        """Symmetrized graph Laplacian L = D - (W + W^T)."""
        A = self.W + self.W.T
        deg = torch.diag(torch.sum(A, dim=1))
        return deg - A

    def dirichlet_energy(self, x: torch.Tensor) -> float:
        """Calculates graph Dirichlet energy x^T L x."""
        L = self.laplacian()
        return float((x.T @ L @ x).item())

    def assert_invariants(self) -> bool:
        """Audits the full mathematical invariant battery of the conductance engine.

        Invariants checked:
          1. Non-negativity: W >= 0 entrywise.
          2. Hollow diagonal: W_{ii} == 0 for all i.
          3. Bounded range: W_{ij} <= w_max for all i, j.
          4. Discrete lattice levels: W_{ij} in {0, delta_w, ..., w_max} to tolerance 1e-5.
          5. Energy finiteness: 0.5 * sum(W^2) is finite and non-negative.
        """
        # 1. Non-negativity
        assert (self.W >= 0.0).all().item(), "Invariant violation: W contains negative values"
        assert (self.W_raw >= 0.0).all().item(), "Invariant violation: W_raw contains negative values"

        # 2. Hollow diagonal
        diag_w = torch.diag(self.W)
        assert torch.all(diag_w == 0.0).item(), f"Invariant violation: non-zero diagonal in W: {diag_w}"
        diag_raw = torch.diag(self.W_raw)
        assert torch.all(diag_raw == 0.0).item(), f"Invariant violation: non-zero diagonal in W_raw: {diag_raw}"

        # 3. Bounded range
        assert (self.W <= self.w_max + 1e-6).all().item(), "Invariant violation: W exceeds w_max"
        assert (self.W_raw <= self.w_max + 1e-6).all().item(), "Invariant violation: W_raw exceeds w_max"

        # 4. Discrete lattice levels
        scaled = self.W / self.delta_w
        rounded = torch.round(scaled)
        diff = torch.abs(scaled - rounded)
        assert (diff < 1e-4).all().item(), "Invariant violation: W has non-discrete lattice levels"

        # 5. Energy finiteness
        energy = 0.5 * torch.sum(self.W ** 2).item()
        assert math.isfinite(energy) and energy >= 0.0, "Invariant violation: lattice energy not finite"

        return True


def demo_ledger() -> Dict[str, Any]:
    """Generates the verified experimental ledger for Chapter 10."""
    torch.manual_seed(42)
    engine = PlasticConductanceEngine(PlasticityConfig(num_entities=8, bits=4, seed=42))

    # Initial state
    d_initial = engine.resistance(0, 1)

    # Apply 5 successive Hebbian updates along edge (0 -> 1)
    d_trajectory = [d_initial]
    w_trajectory = [float(engine.W[0, 1].item())]

    for step in range(5):
        engine.update_conductance(0, 1, weight_inc=1.0)
        engine.assert_invariants()
        d_trajectory.append(engine.resistance(0, 1))
        w_trajectory.append(float(engine.W[0, 1].item()))

    # Verify resistance contraction
    for k in range(len(d_trajectory) - 1):
        assert d_trajectory[k+1] <= d_trajectory[k], "Monotone contraction violated"

    # Endogenous consolidation on a 2-hop chain: 1 -> 2
    engine.update_conductance(1, 2, weight_inc=3.0)
    report = engine.endogenous_step(num_steps=3, max_probes=32)
    engine.assert_invariants()

    return {
        "initial_resistance": d_initial,
        "final_resistance": d_trajectory[-1],
        "resistance_trajectory": d_trajectory,
        "conductance_trajectory": w_trajectory,
        "consolidation_report": report,
        "invariants_passed": True,
    }


if __name__ == "__main__":
    print("=" * 68)
    print("Chapter 10 — Two-Timescale Learning: Hebbian Plasticity Ledger")
    print("=" * 68)
    res = demo_ledger()
    print(f"Initial resistance (0->1): {res['initial_resistance']:.2e}")
    print("Hebbian Resistance Contraction Trajectory:")
    for step, (w, d) in enumerate(zip(res['conductance_trajectory'], res['resistance_trajectory'])):
        print(f"  Step {step}: W[0,1] = {w:.4f}  -->  d_01 = {d:8.4f}")
    rep = res["consolidation_report"]
    print("\nEndogenous Consolidation:")
    print(f"  Probes: {rep.num_probes} | Edges Updated: {rep.edges_updated} | Mean Kappa: {rep.mean_kappa:.4f}")
    print(f"  Energy before: {rep.energy_before:.6f} | Energy after: {rep.energy_after:.6f}")
    print(f"  Invariant Battery Audit: {'PASSED' if res['invariants_passed'] else 'FAILED'}")
    print("=" * 68)
