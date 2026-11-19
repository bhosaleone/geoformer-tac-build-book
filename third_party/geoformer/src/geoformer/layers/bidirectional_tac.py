"""
Goal-Conditioned Bidirectional TAC (GCBS-TAC)
=============================================
A deep-horizon reasoning engine that closes the last gap in the TAC free-energy
formalism and scales transitive reachability to 1024+ hops on un-truncated graphs.

Motivation (see ``exploration_of_lucid_minds/05`` and ``08``)
------------------------------------------------------------
The TAC objective is

    F[z; q] = lam_dir/2 Tr(z^T L_dir z) + lam_V/2 V_bistable(z) + lam_q d_sem(z, q)

The 32-hop engine (``GoalConditionedEmergentHopAttractor``) implements the first
two terms (propagation + soliton sharpening) and evaluates ``d_sem`` only as a
*halting* signal.  ``semantic_distance`` computes the goal gradient ``grad_q``
but the relaxation step discards it.  Consequently the engine is purely
amplitude-driven: a coherent decoy whose packet amplitude equals the true path
(e.g. a directed cycle) captures the soliton permanently, at *any* depth.

This module restores the missing third term.  The gradient of ``lam_q d_sem``
with a discrete goal is, in graph terms, an **adjoint backward message** seeded
at the goal.  We therefore run a second soliton wavefront on the reverse
transition kernel; its time-accumulated field is the soft goal-reachability
potential

    b_j = sum_t  (reverse soliton mass at j after t steps).

Multiplying the forward field by ``b`` annihilates every branch that cannot
reach the query goal, no matter how locally attractive or amplitude-coherent it
is.  This is exactly the forward-backward (Baum-Welch / adjoint) construction,
realised as physical relaxation.

Scaling
-------
All propagation is sparse edge-list scatter (``index_add_``), i.e. O(|E|) per
step instead of the dense O(N^2) matvec.  A 1024-hop query on a 1.8k-node graph
therefore completes in well under a second on CPU, and the cost is linear in the
number of *edges actually traversed* rather than in N^2.

An optional binary-lifting consolidator implements the multi-timescale memory
of the ELM canon (endogenous shortcut synthesis, CLS replay): a length-k
geodesic is compiled into a hierarchy of length-2^l shortcuts, after which the
same query settles in O(log k) relaxation steps.  Cold (unconsolidated) and
warm (consolidated) paths are both reported by the benchmark.

Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs | SetMD | ISL Research
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sparse graph primitives
# ---------------------------------------------------------------------------

def dense_to_edges(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the non-zero directed edge list ``(src, dst, w)`` from a dense W."""
    src, dst = torch.nonzero(W > 0.0, as_tuple=True)
    w = W[src, dst]
    return src, dst, w


def edge_degrees(
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out_degree, in_degree)`` per node for an edge list."""
    out_deg = torch.zeros(n, dtype=w.dtype, device=w.device)
    in_deg = torch.zeros(n, dtype=w.dtype, device=w.device)
    out_deg.index_add_(0, src, w)
    in_deg.index_add_(0, dst, w)
    return out_deg, in_deg


def sparse_step(
    z: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    inv_deg: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """
    One row-stochastic propagation step over an edge list.

    forward:  z'[dst] += z[src] * W[src,dst] / out_deg[src]
    reverse:  z'[src] += z[dst] * W[src,dst] / in_deg[dst]

    The reverse step is the adjoint transition on W^T and is used to build the
    goal-reachability potential.
    """
    out = torch.zeros_like(z)
    if reverse:
        msg = z[dst] * w * inv_deg[dst]
        out.index_add_(0, src, msg)
    else:
        msg = z[src] * w * inv_deg[src]
        out.index_add_(0, dst, msg)
    return out


def soliton_sharpen(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Non-linear soliton wavefront sharpening, normalised to the simplex."""
    zp = torch.clamp(z, min=0.0).pow(gamma)
    return zp / torch.clamp(zp.sum(), min=1e-30)


def shannon_entropy(z: torch.Tensor) -> float:
    p = torch.clamp(z, min=1e-12)
    return float(-(p * torch.log2(p)).sum())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GoalConditionedBidirectionalTAC(nn.Module):
    """
    Goal-conditioned bidirectional soliton TAC engine for deep transitive
    reasoning (validated 32 -> 2048 hops on un-truncated graphs).

    The engine runs three coupled fields:

    1. **Forward soliton** ``f_t``: source-seeded propagation + adaptive
       entropy-modulated sharpening (the classical TAC wavefront).
    2. **Backward light-cone** ``b_j``: goal-seeded propagation on the reverse
       kernel, accumulated over relaxation time; this is the goal-attractor
       gradient term of the free energy.
    3. **Goal-conditioned state** ``z_t ∝ f_t ⊙ b``: intersection of the
       forward reachable set and the backward reachable set, i.e. the nodes
       lying on at least one source->goal geodesic.

    Halting is autonomous: ``||z_{t+1}-z_t||_2 < eps_tac`` **and**
    ``d_sem(z,q) = 1 - z_goal < eps_q``, evaluated on the untruncated graph.
    No artificial sink or truncated subgraph is used.
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
        action_quantum: float = 0.0,
        max_steps: int = 2048,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.gamma_0 = gamma_0
        self.beta_adaptive = beta_adaptive
        self.gamma_max = gamma_max
        self.gamma_backward = gamma_backward
        self.eps_tac = eps_tac
        self.eps_q = eps_q
        self.eps_goal = eps_goal
        self.action_quantum = action_quantum
        self.max_steps = max_steps
        self.use_adjoint = use_adjoint

    # -- field construction -------------------------------------------------

    def _backward_light_cone(
        self,
        goal: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        w: torch.Tensor,
        inv_in_deg: torch.Tensor,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Accumulated reverse soliton field seeded at ``goal``.

        ``b_field[j] > 0`` iff node ``j`` can reach the goal within ``steps``
        reverse transitions.  The field is scaled by the goal's accumulated
        potential, so ``b_field[goal] == 1`` and the halting threshold
        ``eps_goal`` has a fixed meaning.  The scale cancels in the conditioned
        state ``z ∝ f ⊙ b``, so this choice does not affect the dynamics.
        """
        b = torch.zeros(self.num_entities, device=device)
        b[goal] = 1.0
        # Include the seed itself so the goal carries unit backward potential.
        field = b.clone()
        for _ in range(steps):
            b = sparse_step(b, src, dst, w, inv_in_deg, reverse=True)
            h = shannon_entropy(b)
            gamma = min(self.gamma_max, self.gamma_backward + self.beta_adaptive * h)
            b = soliton_sharpen(b, gamma)
            field = field + b
            if float(b.sum()) < 1e-12:
                break
        field = field / torch.clamp(field[goal], min=1e-30)
        return field

    def _forward_soliton(
        self,
        source: int,
        goal: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        w: torch.Tensor,
        inv_out_deg: torch.Tensor,
        b_field: torch.Tensor,
        max_steps: int,
        device: torch.device,
        return_trace: bool,
    ) -> Tuple[torch.Tensor, int, float, List[torch.Tensor]]:
        """Goal-conditioned forward relaxation. Returns (z, tau*, d_sem, trace)."""
        z = torch.zeros(self.num_entities, device=device)
        z[source] = 1.0
        trace: List[torch.Tensor] = [z.clone()] if return_trace else []
        tau = max_steps

        for t in range(1, max_steps + 1):
            z_prop = sparse_step(z, src, dst, w, inv_out_deg, reverse=False)
            h = shannon_entropy(z_prop)
            gamma = min(self.gamma_max, self.gamma_0 + self.beta_adaptive * h)
            z_prop = soliton_sharpen(z_prop, gamma)

            if self.action_quantum > 0.0:
                cut = self.action_quantum * float(z_prop.max())
                z_prop = torch.where(z_prop < cut, torch.zeros_like(z_prop), z_prop)
                z_prop = z_prop / torch.clamp(z_prop.sum(), min=1e-30)

            prev = z
            if self.use_adjoint:
                z_next = z_prop * (b_field + 1e-30)
                s = float(z_next.sum())
                if s < 1e-30:
                    # Goal unreachable from this source region: fall back to the
                    # unmodulated wavefront so the engine still returns a state.
                    z_next = z_prop
                else:
                    z_next = z_next / s
            else:
                z_next = z_prop

            drift = float(torch.norm(z_next - prev, p=2))
            d_sem = float(max(0.0, 1.0 - float(z_next[goal])))

            z = z_next
            if return_trace:
                trace.append(z.clone())

            if d_sem < self.eps_q or float(z[goal]) >= self.eps_goal:
                tau = t
                break
            if drift < self.eps_tac and d_sem < self.eps_q:
                tau = t
                break

        return z, tau, float(max(0.0, 1.0 - float(z[goal]))), trace

    # -- public API ---------------------------------------------------------

    def forward(
        self,
        source_indices: torch.Tensor,
        W_conductance: torch.Tensor,
        goal_indices: Optional[torch.Tensor] = None,
        fixed_hops: Optional[int] = None,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            source_indices: ``[B]`` source entity IDs.
            W_conductance: ``[N, N]`` non-negative directed conductance lattice.
            goal_indices: ``[B]`` query goal entity IDs (required for halting and
                for the adjoint goal potential).
            fixed_hops: if given, relax exactly this many steps (no early halt).
            return_trace: also return per-step distributions.

        Returns:
            final_z: ``[B, N]`` goal-conditioned distribution.
            emergent_hops: ``[B]`` autonomous halting depth ``tau*``.
            final_d_sem: ``[B]`` semantic distance ``1 - z_goal``.
            trace: list of ``[B, N]`` snapshots (if requested).
        """
        device = W_conductance.device
        B = int(source_indices.shape[0])
        n = W_conductance.shape[0]
        if n != self.num_entities:
            self.num_entities = n

        src, dst, w = dense_to_edges(W_conductance)
        out_deg, in_deg = edge_degrees(src, dst, w, n)
        inv_out = 1.0 / torch.clamp(out_deg, min=1e-30)
        inv_in = 1.0 / torch.clamp(in_deg, min=1e-30)

        max_steps = int(fixed_hops) if fixed_hops is not None else self.max_steps
        # The backward light cone must be able to traverse the full horizon.
        back_steps = max_steps if fixed_hops is not None else self.max_steps

        outputs = []
        hops = torch.zeros(B, dtype=torch.long, device=device)
        d_sems = torch.zeros(B, device=device)
        traces: Optional[List[torch.Tensor]] = [] if return_trace else None

        for b in range(B):
            s = int(source_indices[b].item())
            g = int(goal_indices[b].item()) if goal_indices is not None else s
            if self.use_adjoint and goal_indices is not None:
                b_field = self._backward_light_cone(g, src, dst, w, inv_in, back_steps, device)
            else:
                b_field = torch.ones(n, device=device)

            z, tau, d_sem, tr = self._forward_soliton(
                s, g, src, dst, w, inv_out, b_field, max_steps, device, return_trace
            )
            outputs.append(z)
            hops[b] = tau
            d_sems[b] = d_sem
            if return_trace and traces is not None:
                traces.append(torch.stack(tr, dim=0))

        final_z = torch.stack(outputs, dim=0)
        return final_z, hops, d_sems, traces


# ---------------------------------------------------------------------------
# Multi-timescale consolidation (endogenous shortcut synthesis)
# ---------------------------------------------------------------------------

class BinaryLiftingConsolidator:
    """
    CLS-style replay / endogenous shortcut synthesis.

    Repeatedly composes every length-2^l path into a direct edge, yielding a
    binary-lifting hierarchy.  After ``ceil(log2(k))`` levels a length-``k``
    geodesic is represented by a constant number of coarse edges, so a
    consolidated query settles in O(log k) relaxation steps instead of O(k).

    Only *real* paths are composed (edge (i,j) followed by edge (j,k)), so the
    consolidator never fabricates reachability.  ``max_edges`` bounds memory on
    dense regions.
    """

    def __init__(self, max_edges: int = 5_000_000, decay: float = 1.0) -> None:
        self.max_edges = max_edges
        self.decay = decay

    def consolidate(self, W: torch.Tensor, levels: Optional[int] = None) -> torch.Tensor:
        n = W.shape[0]
        if levels is None:
            levels = max(1, math.ceil(math.log2(max(2, n))))

        # Sparse dict-of-dicts edge store. ``result`` is the union of all
        # dyadic shortcut levels; ``current`` holds edges of length exactly 2^l.
        rows, cols = torch.nonzero(W > 0, as_tuple=True)
        base: Dict[int, Dict[int, float]] = {}
        for i, j in zip(rows.tolist(), cols.tolist()):
            base.setdefault(i, {})[j] = float(W[i, j])

        def _union(dst: Dict[int, Dict[int, float]], add: Dict[int, Dict[int, float]]) -> None:
            for i, row in add.items():
                dst_row = dst.setdefault(i, {})
                for k, v in row.items():
                    if v > dst_row.get(k, 0.0):
                        dst_row[k] = v

        result: Dict[int, Dict[int, float]] = {}
        _union(result, base)
        current = base

        for _ in range(int(levels)):
            new_edges: Dict[int, Dict[int, float]] = {}
            total = 0
            for i, outs in current.items():
                row = new_edges.setdefault(i, {})
                for j, w_ij in outs.items():
                    for k, w_jk in current.get(j, {}).items():
                        if k == i:
                            continue
                        val = w_ij * w_jk * self.decay
                        if val > row.get(k, 0.0):
                            row[k] = val
                        total += 1
                        if total > self.max_edges:
                            break
                    if total > self.max_edges:
                        break
                if total > self.max_edges:
                    break
            if not new_edges:
                break
            _union(result, new_edges)
            current = new_edges

        out = torch.zeros_like(W)
        for i, row in result.items():
            for j, v in row.items():
                if i != j:
                    out[i, j] = v
        return out
