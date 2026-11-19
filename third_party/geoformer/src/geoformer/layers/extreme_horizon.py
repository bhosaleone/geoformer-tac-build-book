"""
Extreme-Horizon TAC: Sparse Relaxation and the Companion Shortcut
=================================================================
This module pushes the goal-conditioned TAC engine beyond the dense-matrix
wall and then beyond the *time-integration* wall.

Two regimes
-----------
1. **Thermodynamic regime** (``GoalConditionedSparseTAC``).  The continuous
   bidirectional relaxation is run on an edge list, so memory is ``O(N + |E|)``
   instead of ``O(N^2)``.  This makes 16,384-hop continuous relaxation feasible
   on graphs where the dense conductance matrix would require tens of GB.
   The backward field is the exact adjoint (reverse-BFS) potential

       b_j = beta^{-d(j, q) / scale}   if j can reach q, else 0,

   which preserves the annihilation guarantee of Theorem Q-4 and is computed
   in ``O(N + |E|)`` rather than ``O(D |E|)``.  A diffusion mode reproduces the
   soft accumulated field of ``bidirectional_tac`` for comparison.

2. **Companion regime** (``CompanionJumpPlanner``).  If b is exact, the
   backward potential is a **gradient field** pointing one step closer to the
   goal.  Following it from the source reconstructs a geodesic without
   integrating the forward relaxation.  Binary lifting on that path compresses
   the readout to ``O(log D)`` jumps.  This is the companion shortcut:

       D-step relaxation        ->  O(D |E|) time
       exact adjoint + lifting  ->  O(N + |E| + log D) time

   The shortcut answers *reachability and the path*.  It does not reproduce the
   continuous probability distribution, which still costs ``O(D |E|)``.

Terminology
-----------
"Companion" is used in the physical/adjoint sense: the reverse field is the
dual companion of the forward wave, and dyadic lifting is the companion
(power-of-two) decomposition of its flow.  No unproven claims are attached to
the word; the guarantees are the ones proved for Q-4 and Q-5.

Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs | SetMD | ISL Research
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .bidirectional_tac import (
    dense_to_edges,
    edge_degrees,
    shannon_entropy,
    soliton_sharpen,
    sparse_step,
)


TensorLike = Union[torch.Tensor, Sequence[int], np.ndarray]


# ---------------------------------------------------------------------------
# Sparse lattice
# ---------------------------------------------------------------------------

@dataclass
class SparseLattice:
    """
    Directed non-negative conductance lattice stored as an edge list.

    ``n`` may be larger than ``max(src, dst) + 1`` to preserve entity indices
    that participate only as isolated nodes.
    """

    src: torch.Tensor
    dst: torch.Tensor
    w: torch.Tensor
    n: int
    inv_out: torch.Tensor
    inv_in: torch.Tensor

    @classmethod
    def from_edges(
        cls,
        src: TensorLike,
        dst: TensorLike,
        w: TensorLike,
        num_entities: Optional[int] = None,
    ) -> "SparseLattice":
        src_t = torch.as_tensor(src, dtype=torch.long)
        dst_t = torch.as_tensor(dst, dtype=torch.long)
        w_t = torch.as_tensor(w, dtype=torch.float32)
        if src_t.numel() != dst_t.numel() or src_t.numel() != w_t.numel():
            raise ValueError("src, dst and w must have the same length")
        if num_entities is None:
            num_entities = int(max(int(src_t.max()), int(dst_t.max()))) + 1
        out_deg, in_deg = edge_degrees(src_t, dst_t, w_t, int(num_entities))
        return cls(
            src=src_t,
            dst=dst_t,
            w=w_t,
            n=int(num_entities),
            inv_out=1.0 / torch.clamp(out_deg, min=1e-30),
            inv_in=1.0 / torch.clamp(in_deg, min=1e-30),
        )

    @classmethod
    def from_dense(cls, W: torch.Tensor) -> "SparseLattice":
        src, dst, w = dense_to_edges(W)
        return cls.from_edges(src, dst, w, W.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def dense_nbytes(self) -> int:
        """Memory a float32 dense conductance matrix would require."""
        return int(self.n) * int(self.n) * 4

    def to_scipy(self):
        """Return a CSR copy of the conductance matrix (CPU)."""
        import scipy.sparse as sp

        return sp.coo_matrix(
            (self.w.detach().cpu().numpy(), (self.src.numpy(), self.dst.numpy())),
            shape=(self.n, self.n),
        ).tocsr()

    def degrees(self) -> Tuple[torch.Tensor, torch.Tensor]:
        out_deg = torch.zeros(self.n)
        in_deg = torch.zeros(self.n)
        out_deg.index_add_(0, self.src, self.w)
        in_deg.index_add_(0, self.dst, self.w)
        return out_deg, in_deg


# ---------------------------------------------------------------------------
# Exact companion potential (reverse BFS)
# ---------------------------------------------------------------------------

def reverse_bfs(
    lattice: SparseLattice,
    goal: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact adjoint field.

    Returns ``(dist, next_step)`` where ``dist[j]`` is the unweighted shortest
    distance ``j -> goal`` (``-1`` if unreachable) and ``next_step[j]`` is an
    out-neighbour of ``j`` one step closer to the goal (``-1`` at the goal or
    when unreachable).
    """
    import scipy.sparse as sp
    from scipy.sparse import csgraph

    A = lattice.to_scipy()
    # Reverse graph: edge (j -> goal) becomes (goal -> j).
    A_rev = sp.csr_matrix((A.data, A.indices, A.indptr), shape=(lattice.n, lattice.n)).T.tocsr()
    order, pred = csgraph.breadth_first_order(
        A_rev, int(goal), directed=True, return_predecessors=True
    )

    dist = np.full(lattice.n, -1, dtype=np.int32)
    dist[int(goal)] = 0
    # ``breadth_first_order`` returns a BFS order; predecessors are earlier.
    # Walk the order once to assign exact distances.
    for u in order[1:]:
        p = int(pred[u])
        if p >= 0:
            dist[u] = dist[p] + 1

    next_step = np.asarray(pred, dtype=np.int64).copy()
    next_step[int(goal)] = -1
    next_step[dist < 0] = -1
    return dist, next_step


def companion_potential(
    lattice: SparseLattice,
    goal: int,
    scale: Optional[float] = None,
    base: float = 2.0,
) -> np.ndarray:
    """
    Distance-graded adjoint potential with ``b[goal] == 1``.

    ``b_j = base^{-dist_j / scale}`` on the goal-reaching set and ``0`` off it.
    The scale defaults to ``max(1, max_distance / 20)``, so the smallest
    positive value is around ``base^{-20}`` and never underflows.
    """
    dist, _ = reverse_bfs(lattice, goal)
    reach = dist >= 0
    if scale is None:
        max_d = int(dist[reach].max()) if reach.any() else 0
        scale = max(1.0, float(max_d) / 20.0)
    b = np.zeros(lattice.n, dtype=np.float32)
    b[reach] = np.power(np.float32(base), -dist[reach].astype(np.float32) / np.float32(scale))
    b[int(goal)] = 1.0
    return b


# ---------------------------------------------------------------------------
# Continuous sparse relaxation
# ---------------------------------------------------------------------------

class GoalConditionedSparseTAC(nn.Module):
    """
    Goal-conditioned bidirectional TAC on an edge list.

    ``backward_mode``:
      * ``"exact"``      - reverse-BFS companion potential (default, O(N+|E|)).
      * ``"diffusion"``  - accumulated reverse soliton field (soft, O(D|E|)).
      * ``"none"``       - no adjoint (amplitude-only control).

    ``backend``:
      * ``"auto"``  - SciPy on CPU for ``n > 4096``, Torch otherwise.
      * ``"torch"`` - scatter-based Torch loop.
      * ``"scipy"`` - CSR matvec loop (CPU; fastest for very large graphs).
    """

    def __init__(
        self,
        num_entities: int,
        gamma_0: float = 2.5,
        beta_adaptive: float = 0.25,
        gamma_max: float = 4.5,
        gamma_backward: float = 1.5,
        eps_tac: float = 1e-4,
        eps_q: float = 0.15,
        eps_goal: float = 0.60,
        max_steps: int = 2048,
        backward_mode: str = "exact",
        potential_scale_div: float = 20.0,
        potential_base: float = 2.0,
        backend: str = "auto",
        action_quantum: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.gamma_0 = gamma_0
        self.beta_adaptive = beta_adaptive
        self.gamma_max = gamma_max
        self.gamma_backward = gamma_backward
        self.eps_tac = eps_tac
        self.eps_q = eps_q
        self.eps_goal = eps_goal
        self.max_steps = max_steps
        self.backward_mode = backward_mode
        self.potential_scale_div = potential_scale_div
        self.potential_base = potential_base
        self.backend = backend
        self.action_quantum = action_quantum

    # -- adjoint field ------------------------------------------------------

    def _exact_potential(self, lattice: SparseLattice, goal: int, max_steps: int) -> torch.Tensor:
        dist, _ = reverse_bfs(lattice, goal)
        reach = dist >= 0
        max_d = int(dist[reach].max()) if reach.any() else 0
        scale = max(1.0, float(max_d) / float(self.potential_scale_div))
        b = np.zeros(lattice.n, dtype=np.float32)
        b[reach] = np.power(
            np.float32(self.potential_base),
            -dist[reach].astype(np.float32) / np.float32(scale),
        )
        b[int(goal)] = 1.0
        return torch.from_numpy(b)

    def _diffusion_potential(
        self, lattice: SparseLattice, goal: int, steps: int, use_scipy: bool
    ) -> torch.Tensor:
        if use_scipy:
            import scipy.sparse as sp

            A = lattice.to_scipy()
            in_deg = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
            Q = (A @ sp.diags(1.0 / np.maximum(in_deg, 1e-30))).tocsr()
            b = np.zeros(lattice.n, dtype=np.float32)
            b[int(goal)] = 1.0
            field = b.copy()
            for _ in range(int(steps)):
                b = Q @ b
                h = float(-(np.clip(b, 1e-12, None) * np.log2(np.clip(b, 1e-12, None))).sum(dtype=np.float32))
                gamma = min(self.gamma_max, self.gamma_backward + self.beta_adaptive * h)
                b = np.maximum(b, 0.0).astype(np.float32)
                b = np.power(b, gamma)
                b /= max(float(b.sum(dtype=np.float32)), 1e-30)
                field += b
                if float(b.sum(dtype=np.float32)) < 1e-12:
                    break
            field /= max(float(field[int(goal)]), 1e-30)
            return torch.from_numpy(field.astype(np.float32))

        b = torch.zeros(lattice.n)
        b[int(goal)] = 1.0
        field = b.clone()
        for _ in range(int(steps)):
            b = sparse_step(b, lattice.src, lattice.dst, lattice.w, lattice.inv_in, reverse=True)
            h = shannon_entropy(b)
            gamma = min(self.gamma_max, self.gamma_backward + self.beta_adaptive * h)
            b = soliton_sharpen(b, gamma)
            field = field + b
            if float(b.sum()) < 1e-12:
                break
        field = field / torch.clamp(field[int(goal)], min=1e-30)
        return field

    def _potential(self, lattice: SparseLattice, goal: int, max_steps: int, use_scipy: bool) -> torch.Tensor:
        if self.backward_mode == "none":
            return torch.ones(lattice.n)
        if self.backward_mode == "exact":
            return self._exact_potential(lattice, goal, max_steps)
        if self.backward_mode == "diffusion":
            return self._diffusion_potential(lattice, goal, max_steps, use_scipy)
        raise ValueError(f"unknown backward_mode: {self.backward_mode}")

    # -- forward relaxation -------------------------------------------------

    def _relax_scipy(
        self,
        lattice: SparseLattice,
        source: int,
        goal: int,
        b_np: np.ndarray,
        max_steps: int,
        return_trace: bool,
    ):
        import scipy.sparse as sp

        A = lattice.to_scipy()
        out_deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
        P = (sp.diags(1.0 / np.maximum(out_deg, 1e-30)) @ A).T.tocsr()

        def sharpen(v: np.ndarray, g: float) -> np.ndarray:
            v = np.maximum(v, 0.0).astype(np.float32)
            v = np.power(v, np.float32(g))
            return v / max(float(v.sum(dtype=np.float32)), 1e-30)

        def entropy(v: np.ndarray) -> float:
            p = np.clip(v, 1e-12, None)
            return float(-(p * np.log2(p)).sum(dtype=np.float32))

        z = np.zeros(lattice.n, dtype=np.float32)
        z[int(source)] = 1.0
        trace: List[torch.Tensor] = [torch.from_numpy(z.copy())] if return_trace else []
        tau = int(max_steps)
        for t in range(1, int(max_steps) + 1):
            zp = P @ z
            gamma = min(self.gamma_max, self.gamma_0 + self.beta_adaptive * entropy(zp))
            zp = sharpen(zp, gamma)
            if self.action_quantum > 0.0:
                cut = self.action_quantum * float(zp.max())
                zp = np.where(zp < cut, 0.0, zp).astype(np.float32)
                zp /= max(float(zp.sum(dtype=np.float32)), 1e-30)
            zn = zp * (b_np + np.float32(1e-30))
            s = float(zn.sum(dtype=np.float32))
            zn = zp if s < 1e-30 else (zn / s).astype(np.float32)
            z = zn
            if return_trace:
                trace.append(torch.from_numpy(z.copy()))
            if float(z[int(goal)]) >= self.eps_goal or (1.0 - float(z[int(goal)])) < self.eps_q:
                tau = t
                break
        return torch.from_numpy(z), tau, max(0.0, 1.0 - float(z[int(goal)])), trace

    def _relax_torch(
        self,
        lattice: SparseLattice,
        source: int,
        goal: int,
        b_field: torch.Tensor,
        max_steps: int,
        return_trace: bool,
    ):
        z = torch.zeros(lattice.n)
        z[int(source)] = 1.0
        trace: List[torch.Tensor] = [z.clone()] if return_trace else []
        tau = int(max_steps)
        for t in range(1, int(max_steps) + 1):
            zp = sparse_step(z, lattice.src, lattice.dst, lattice.w, lattice.inv_out, reverse=False)
            gamma = min(self.gamma_max, self.gamma_0 + self.beta_adaptive * shannon_entropy(zp))
            zp = soliton_sharpen(zp, gamma)
            if self.action_quantum > 0.0:
                cut = self.action_quantum * float(zp.max())
                zp = torch.where(zp < cut, torch.zeros_like(zp), zp)
                zp = zp / torch.clamp(zp.sum(), min=1e-30)
            zn = zp * (b_field + 1e-30)
            s = float(zn.sum())
            zn = zp if s < 1e-30 else zn / s
            z = zn
            if return_trace:
                trace.append(z.clone())
            if float(z[int(goal)]) >= self.eps_goal or (1.0 - float(z[int(goal)])) < self.eps_q:
                tau = t
                break
        return z, tau, max(0.0, 1.0 - float(z[int(goal)])), trace

    # -- public API ---------------------------------------------------------

    def forward(
        self,
        source_indices: TensorLike,
        lattice: SparseLattice,
        goal_indices: Optional[TensorLike] = None,
        fixed_hops: Optional[int] = None,
        return_trace: bool = False,
    ):
        sources = torch.as_tensor(source_indices, dtype=torch.long).view(-1)
        goals = (
            torch.as_tensor(goal_indices, dtype=torch.long).view(-1)
            if goal_indices is not None
            else sources.clone()
        )
        if sources.numel() != goals.numel():
            raise ValueError("source_indices and goal_indices must have equal length")
        if self.num_entities != lattice.n:
            self.num_entities = lattice.n

        max_steps = int(fixed_hops) if fixed_hops is not None else int(self.max_steps)
        use_scipy = (
            self.backend == "scipy"
            or (self.backend == "auto" and lattice.n > 4096)
        )

        outputs, hops, d_sems = [], [], []
        traces: Optional[List[torch.Tensor]] = [] if return_trace else None
        for b_idx in range(sources.numel()):
            s, g = int(sources[b_idx]), int(goals[b_idx])
            b_field = self._potential(lattice, g, max_steps, use_scipy)
            if use_scipy:
                z, tau, d_sem, tr = self._relax_scipy(
                    lattice, s, g, b_field.numpy(), max_steps, return_trace
                )
            else:
                z, tau, d_sem, tr = self._relax_torch(
                    lattice, s, g, b_field, max_steps, return_trace
                )
            outputs.append(z)
            hops.append(tau)
            d_sems.append(d_sem)
            if return_trace and traces is not None:
                traces.append(torch.stack(tr, dim=0))

        return (
            torch.stack(outputs, dim=0),
            torch.tensor(hops, dtype=torch.long),
            torch.tensor(d_sems, dtype=torch.float32),
            traces,
        )


# ---------------------------------------------------------------------------
# Companion shortcut: adjoint path + binary lifting
# ---------------------------------------------------------------------------

class CompanionJumpPlanner:
    """
    Exact shortest-path readout on the companion potential.

    The reverse BFS provides ``next_step[j]``, an out-neighbour one step closer
    to the goal.  Following that field from the source yields a geodesic without
    integrating the forward relaxation.  The path is then dyadically lifted:

        D hops  ->  popcount(D) jumps  <=  log2(D) + 1 jumps.

    The planner answers reachability and the path exactly.  It is not a
    replacement for the continuous relaxation when the full probability field
    is required.
    """

    def __init__(self, lattice: SparseLattice, base: float = 2.0) -> None:
        self.lattice = lattice
        self.base = base

    def plan(
        self,
        source: int,
        goal: int,
        return_path: bool = False,
    ) -> Dict[str, object]:
        t0 = time.time()
        dist, next_step = reverse_bfs(self.lattice, int(goal))
        source, goal = int(source), int(goal)

        if source == goal:
            return dict(reached=True, hops=0, jumps=0, jump_lengths=[],
                        path=[source] if return_path else None,
                        sec=time.time() - t0)

        if dist[source] < 0:
            return dict(reached=False, hops=-1, jumps=-1, jump_lengths=[],
                        path=None, sec=time.time() - t0)

        path: List[int] = [source]
        u = source
        while u != goal:
            u = int(next_step[u])
            if u < 0:
                return dict(reached=False, hops=-1, jumps=-1, jump_lengths=[],
                            path=None, sec=time.time() - t0)
            path.append(u)

        hops = len(path) - 1
        idx = 0
        jump_lengths: List[int] = []
        while idx < hops:
            remaining = hops - idx
            level = int(remaining).bit_length() - 1
            step = 1 << level
            idx += step
            jump_lengths.append(step)

        return dict(
            reached=bool(path[-1] == goal),
            hops=int(hops),
            jumps=int(len(jump_lengths)),
            jump_lengths=jump_lengths,
            path=path if return_path else None,
            sec=time.time() - t0,
        )


def binary_lift_path(path: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Compress a path into ``(jump_length, destination)`` pairs.

    At each index take the largest power of two that does not overshoot.  The
    number of pairs is ``popcount(len(path)-1) <= log2(D)+1``.
    """
    hops = len(path) - 1
    idx = 0
    jumps: List[Tuple[int, int]] = []
    while idx < hops:
        remaining = hops - idx
        step = 1 << (int(remaining).bit_length() - 1)
        idx += step
        jumps.append((step, int(path[idx])))
    return jumps
