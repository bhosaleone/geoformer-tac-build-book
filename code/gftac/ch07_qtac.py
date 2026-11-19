# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 7 — Q-TAC Inside the Block: Continuous Relaxation Layers.

A transformer gets depth by stacking blocks. Q-TAC gets depth *without
stacking*: it takes the block's feed-forward output as a starting point and
lets it relax, under a gradient flow on a latent free energy, toward an
attractor. The number of relaxation steps is a depth the model can spend at
inference time, and the loop halts when the drift falls below a tolerance.

The latent free energy has three terms, each a pull on the state h:

    F(h) = 1/2 ||h - h_init||^2                 (restoration: stay near the
                                                 feed-forward target)
         + (lambda_dir/2) Tr(h^T L h)           (Dirichlet: smooth on the graph)
         + (lambda_s5/2) ||h - P_S5 h||^2       (S5: project onto the symmetry
                                                 invariant subspace)

and the relaxation is the Euler discretization of dh/dtau = -grad F:

    h_{tau+1} = h_tau - delta_tau * [ (h_tau - h_init)
                                    + lambda_dir * L h_tau
                                    + lambda_s5 * (h_tau - P_S5 h_tau) ].

This chapter proves the loop converges and bounds when it halts, builds the
loop and its multi-scale schedules, and pins the result against the
production SDK. (The S5 projector itself is built here because the loop needs
it; Chapter 8 develops its theory.)

Book theorems proved in the chapter and pinned by the tests:

  B.7.1  Lyapunov dissipation. F is a convex quadratic with gradient
         A h - h_init, A = (1+lambda_s5) I + lambda_dir L - lambda_s5 P_S5
         symmetric positive definite. Along the exact flow dF/dtau =
         -||grad F||^2 <= 0; the Euler step decreases F iff
         delta_tau < 2 ||grad F||^2 / (grad F^T A grad F), and in particular
         whenever delta_tau < 2 / lambda_max(A).
  B.7.2  Contraction and halting. For delta_tau <= 2/(lambda_min+lambda_max)
         the error obeys ||h_tau - h*|| <= (1 - delta_tau lambda_min)^tau
         ||h_0 - h*||, so the loop reaches tolerance eps within
         tau* = ceil( ln(||e_0||/eps) / -ln(1 - delta_tau lambda_min) ) steps.
  B.7.3  Schedule as a polynomial filter. The k-step linear propagator
         P_k = prod_i (I - delta_tau_i A) is a degree-k polynomial in A that
         depends only on the *multiset* {delta_tau_i}; hence the output is
         invariant under permuting the schedule, and the schedule acts as a
         spectral filter on [lambda_min, lambda_max] whose worst-case
         contraction is minimized by the Chebyshev step sizes.

Run the demonstration ledger::

    python3 code/gftac/ch07_qtac.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Allow standalone execution: python3 code/gftac/ch07_qtac.py
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

__all__ = [
    "BookS5Projector",
    "BookQuantizedTAC",
    "make_relaxation_schedule",
    "relax_variable",
    "chebyshev_schedule",
    "chain_laplacian",
    "demo_ledger",
]


# ----------------------------------------------------------------------
# The S5 Cayley projector (built here because Q-TAC needs it; Ch 8 develops
# its theory)
# ----------------------------------------------------------------------

class BookS5Projector(nn.Module):
    """Projection onto the S5-invariant subspace, with a description-length readout.

    Projects a hidden vector through a 26-dimensional space — the summed
    dimension of the seven irreducible representations of S5 — and back. The
    two linear maps are initialized orthogonally, so ``proj_out @ proj_in`` is
    an orthogonal projector. The forward pass also returns ``K_S``, a Shannon
    entropy over the seven irrep energy channels, rescaled by log2(120) —
    the description-complexity readout of Chapter 8.
    """

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.d_model = d_model
        self.irrep_dims = [1, 1, 4, 4, 5, 5, 6]
        self.total_irrep_dim = sum(self.irrep_dims)  # 26
        self.proj_in = nn.Linear(d_model, self.total_irrep_dim, bias=False)
        self.proj_out = nn.Linear(self.total_irrep_dim, d_model, bias=False)
        self._init_orthogonal_irreps()

    def _init_orthogonal_irreps(self) -> None:
        with torch.no_grad():
            if self.d_model >= self.total_irrep_dim:
                Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.total_irrep_dim))
                self.proj_in.weight.copy_(Q.t()[:self.total_irrep_dim, :])
                self.proj_out.weight.copy_(Q[:, :self.total_irrep_dim])
            else:
                Q, _ = torch.linalg.qr(torch.randn(self.total_irrep_dim, self.total_irrep_dim))
                self.proj_in.weight.copy_(Q[:self.total_irrep_dim, :self.d_model])
                self.proj_out.weight.copy_(Q.t()[:self.d_model, :self.total_irrep_dim])

    def projector_matrix(self) -> torch.Tensor:
        """The d_model x d_model matrix P = proj_out @ proj_in (orthogonal)."""
        return self.proj_out.weight @ self.proj_in.weight

    def compute_irrep_energies(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(h)
        energies = []
        start = 0
        for dim in self.irrep_dims:
            end = start + dim
            energies.append(torch.sum(z[..., start:end] ** 2, dim=-1).mean())
            start = end
        return torch.stack(energies)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_in(h)
        h_sym = self.proj_out(z)
        energies = self.compute_irrep_energies(h)
        total_e = torch.sum(energies) + 1e-8
        probs = energies / total_e
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
        K_S = 6.9069 * (entropy / math.log2(7.0))
        return h_sym, K_S


# ----------------------------------------------------------------------
# Q-TAC relaxation loop
# ----------------------------------------------------------------------

class BookQuantizedTAC(nn.Module):
    """Continuous relaxation until the Lyapunov drift falls below tolerance.

    Replicates the production SDK's ``QuantizedTAC``: a fixed Euler step
    ``delta_tau`` on the three-term gradient, a step budget taken from the
    schedule (its last element), and a drift-based early stop. The relaxation
    is the layer-free depth: more steps, more effective depth, with no extra
    parameters beyond the shared S5 projector.
    """

    def __init__(
        self,
        d_model: int = 32,
        tau_max: int = 4,
        delta_tau: float = 0.1,
        lambda_dir: float = 0.05,
        lambda_s5: float = 0.05,
        schedule: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_max = tau_max
        self.delta_tau = delta_tau
        self.lambda_dir = lambda_dir
        self.lambda_s5 = lambda_s5
        self.schedule = schedule
        self.s5_projector = BookS5Projector(d_model=d_model)

    def system_matrix(self, L_matrix: torch.Tensor) -> torch.Tensor:
        """The symmetric positive-definite operator A with grad F = A h - h_init.

        Acting on the flattened (N*D) state,

            A = (1 + lambda_s5) I  +  lambda_dir (L kron I_D)  -  lambda_s5 (I_N kron P).

        The Dirichlet term couples *nodes* (L acts on the N axis); the S5 term
        couples *features* (P acts on the D axis). The two therefore commute and
        A is symmetric positive definite, since L is PSD and I - P is PSD.
        """
        N = L_matrix.shape[0]
        D = self.d_model
        L = L_matrix[:N, :N]
        P = self.s5_projector.projector_matrix()
        dev, dt = L.device, L.dtype
        eye_N = torch.eye(N, device=dev, dtype=dt)
        eye_D = torch.eye(D, device=dev, dtype=dt)
        return ((1.0 + self.lambda_s5) * torch.eye(N * D, device=dev, dtype=dt)
                + self.lambda_dir * torch.kron(L, eye_D)
                - self.lambda_s5 * torch.kron(eye_N, P))

    def free_energy(self, h: torch.Tensor, h_init: torch.Tensor, L_matrix: torch.Tensor) -> float:
        """F(h) = 1/2||h-h_init||^2 + (l_dir/2)Tr(h^T L h) + (l_s5/2)||h-P h||^2."""
        L = L_matrix[:h.shape[1], :h.shape[1]]
        diff = h - h_init
        restoration = 0.5 * float((diff ** 2).sum(-1).mean().detach())
        h_sym, _ = self.s5_projector(h)
        s5 = 0.5 * float(((h - h_sym) ** 2).sum(-1).mean().detach())
        Lh = torch.matmul(L.unsqueeze(0), h)
        dirichlet = 0.5 * float((h * Lh).sum(-1).mean().detach())
        return restoration + self.lambda_dir * dirichlet + self.lambda_s5 * s5

    def forward(
        self,
        h_init: torch.Tensor,
        L_matrix: torch.Tensor,
        schedule: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        B, N, D = h_init.shape
        h = h_init.clone()

        active_schedule = schedule if schedule is not None else self.schedule
        if active_schedule is not None and len(active_schedule) > 0:
            max_steps = int(active_schedule[-1])
        else:
            max_steps = self.tau_max

        L = L_matrix[:N, :N]
        final_K_S = torch.tensor(0.0, device=h.device)
        actual_tau = 0

        for _ in range(max_steps):
            actual_tau += 1
            grad_dirichlet = torch.matmul(L.unsqueeze(0), h)
            h_sym, K_S = self.s5_projector(h)
            grad_s5 = h - h_sym
            final_K_S = K_S
            grad_restoration = h - h_init
            dh = -(grad_restoration + self.lambda_dir * grad_dirichlet
                   + self.lambda_s5 * grad_s5)
            h_next = h + self.delta_tau * dh
            diff = torch.norm(h_next - h, dim=-1).mean()
            h = h_next
            if diff.item() < 1e-4:
                break

        return h, actual_tau, final_K_S


# ----------------------------------------------------------------------
# Multi-scale schedules
# ----------------------------------------------------------------------

def chebyshev_schedule(k: int, lambda_min: float, lambda_max: float) -> List[float]:
    """The k Chebyshev step sizes that minimize the worst-case contraction.

    The optimal degree-k error polynomial on [lambda_min, lambda_max] with
    p(0) = 1 has roots at the shifted Chebyshev nodes; the step sizes are the
    reciprocals of those roots. Applied in any order (Theorem B.7.3), they
    contract the slowest mode by 1/T_k((lmax+lmin)/(lmax-lmin)).
    """
    c = 0.5 * (lambda_min + lambda_max)
    r = 0.5 * (lambda_max - lambda_min)
    return [1.0 / (c - r * math.cos((2 * i - 1) * math.pi / (2 * k)))
            for i in range(1, k + 1)]


def _fibonacci(length: int) -> List[int]:
    fib = [1, 2]
    while len(fib) < length:
        fib.append(fib[-1] + fib[-2])
    return fib[:length]


def _golden_convergents(length: int) -> List[float]:
    """Convergents p_k/q_k of the golden ratio [1;1,1,1,...] = Fibonacci ratios."""
    fib = [1, 1]
    while len(fib) < length + 2:
        fib.append(fib[-1] + fib[-2])
    return [fib[k + 1] / fib[k] for k in range(length)]


def make_relaxation_schedule(
    kind: str = "fixed",
    length: int = 8,
    dt: float = 0.1,
    lambda_min: float = 1.0,
    lambda_max: float = 2.0,
) -> List[float]:
    """Build a multi-scale step-size schedule.

    ``kind`` is one of ``"fixed"`` (constant steps), ``"fibonacci"`` (growing
    Fibonacci steps), ``"convergents"`` (golden-ratio continued-fraction
    convergents), or ``"chebyshev"`` (the minimax-optimal steps on
    ``[lambda_min, lambda_max]``).
    """
    if kind == "fixed":
        return [dt] * length
    if kind == "fibonacci":
        return [dt * f for f in _fibonacci(length)]
    if kind == "convergents":
        return [dt * c for c in _golden_convergents(length)]
    if kind == "chebyshev":
        return chebyshev_schedule(length, lambda_min, lambda_max)
    raise ValueError(f"unknown schedule kind: {kind!r}")


def relax_variable(
    h_init: torch.Tensor,
    L_matrix: torch.Tensor,
    step_sizes: Sequence[float],
    lambda_dir: float = 0.05,
    lambda_s5: float = 0.05,
    s5_projector: Optional[BookS5Projector] = None,
) -> torch.Tensor:
    """Exact variable-step relaxation: apply each step size in turn, no halting.

    This is the *linear* map of Theorem B.7.3 (for fixed projector and L), so
    the output depends only on the multiset of ``step_sizes``.
    """
    if s5_projector is None:
        raise ValueError("relax_variable requires an s5_projector")
    h = h_init.clone()
    N = h.shape[1]
    L = L_matrix[:N, :N]
    for dt in step_sizes:
        h_sym, _ = s5_projector(h)
        dh = -((h - h_init) + lambda_dir * torch.matmul(L.unsqueeze(0), h)
               + lambda_s5 * (h - h_sym))
        h = h + dt * dh
    return h


def chain_laplacian(n: int) -> torch.Tensor:
    """Combinatorial Laplacian of a path graph on n nodes."""
    L = torch.zeros(n, n)
    for i in range(n):
        L[i, i] = 2.0 if 0 < i < n - 1 else 1.0
        if i > 0:
            L[i, i - 1] = -1.0
        if i < n - 1:
            L[i, i + 1] = -1.0
    return L


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def demo_ledger() -> str:
    import copy

    lines: List[str] = []
    lines.append("Chapter 7 — Q-TAC Continuous Relaxation, measured")
    lines.append("=" * 68)

    torch.manual_seed(0)
    d_model, N = 32, 6
    L = chain_laplacian(N)
    h_init = torch.randn(1, N, d_model)

    tac = BookQuantizedTAC(d_model=d_model, tau_max=200, delta_tau=0.1)
    with torch.no_grad():
        A = tac.system_matrix(L)
        eig = torch.linalg.eigvalsh(A)
    lam_min, lam_max = float(eig.min()), float(eig.max())
    h_star = torch.linalg.solve(A, h_init.reshape(h_init.shape[0], -1).T).T.reshape(h_init.shape)

    # -- B.7.1: free energy decreases monotonically ---------------------
    lines.append("B.7.1 — the free energy descends monotonically")
    lines.append(f"  A eigenvalues: lambda_min = {lam_min:.6f}, lambda_max = {lam_max:.6f}"
                 f"  (kappa = {lam_max / lam_min:.4f})")
    lines.append(f"  stability bound 2/lambda_max = {2.0 / lam_max:.4f}  "
                 f"(delta_tau = {tac.delta_tau})")
    h = h_init.clone()
    prev_F, mono, energies = None, True, []
    for _ in range(40):
        F = tac.free_energy(h, h_init, L)
        energies.append(F)
        if prev_F is not None and F > prev_F + 1e-12:
            mono = False
        prev_F = F
        h_sym, _ = tac.s5_projector(h)
        dh = -((h - h_init) + tac.lambda_dir * torch.matmul(L.unsqueeze(0), h)
               + tac.lambda_s5 * (h - h_sym))
        h = (h + tac.delta_tau * dh).detach()
    lines.append(f"  F(h_0) = {energies[0]:.6f}   F(h_40) = {energies[-1]:.10f}   "
                 f"F(h*) = {tac.free_energy(h_star, h_init, L):.10f}")
    lines.append(f"  F non-increasing over 40 steps: {mono}")
    lines.append("")

    # -- B.7.2: geometric contraction and the halting bound -------------
    lines.append("B.7.2 — contraction ||e_tau|| vs the Grönwall bound")
    e0 = float((h_init - h_star).norm())
    lines.append(f"  ||e_0|| = {e0:.6f}")
    lines.append(f"  {'tau':>5}{'||e_tau|| measured':>22}{'bound (1-dt*lmin)^tau':>24}")
    h = h_init.clone()
    for step in range(1, 21):
        h_sym, _ = tac.s5_projector(h)
        dh = -((h - h_init) + tac.lambda_dir * torch.matmul(L.unsqueeze(0), h)
               + tac.lambda_s5 * (h - h_sym))
        h = (h + tac.delta_tau * dh).detach()
        if step in (1, 5, 10, 20):
            err = float((h - h_star).norm())
            bound = (1 - tac.delta_tau * lam_min) ** step * e0
            lines.append(f"  {step:>5}{err:>22.8f}{bound:>24.8f}")
    lines.append("")

    # -- B.7.2: the SDK loop and its halting ----------------------------
    with torch.no_grad():
        h_halt, tau_used, K_S = tac(h_init, L)
    lines.append("B.7.2 — the SDK-style loop with early halting")
    lines.append(f"  schedule budget (tau_max) = {tac.tau_max}")
    lines.append(f"  actual_tau (halting step) = {tau_used}")
    lines.append(f"  halting step <= schedule length: {tau_used <= tac.tau_max}")
    lines.append(f"  ||h_tau - h*|| at halt    = {float((h_halt - h_star).norm()):.3e}")
    lines.append(f"  final K_S (description readout) = {float(K_S):.6f}")
    lines.append("")

    # -- B.7.3: permutation invariance ----------------------------------
    lines.append("B.7.3 — schedule permutation invariance (same multiset)")
    base = [0.05, 0.1, 0.2, 0.4, 0.05, 0.1]
    perm = [0.4, 0.05, 0.1, 0.05, 0.2, 0.1]
    out_base = relax_variable(h_init, L, base, s5_projector=tac.s5_projector)
    out_perm = relax_variable(h_init, L, perm, s5_projector=tac.s5_projector)
    proj64 = copy.deepcopy(tac.s5_projector).double()
    out_b64 = relax_variable(h_init.double(), L.double(), base, s5_projector=proj64)
    out_p64 = relax_variable(h_init.double(), L.double(), perm, s5_projector=proj64)
    lines.append(f"  schedule          : {base}")
    lines.append(f"  permuted          : {perm}")
    lines.append(f"  max |out - out_perm|  float32 = {float((out_base - out_perm).abs().max().detach()):.3e}")
    lines.append(f"  max |out - out_perm|  float64 = {float((out_b64 - out_p64).abs().max().detach()):.3e}")
    lines.append("")

    # -- B.7.3: the schedule as a spectral filter -----------------------
    lines.append("B.7.3 — worst-case contraction per k steps on [lmin, lmax]")
    lines.append(f"  {'k':>4}{'best fixed':>16}{'chebyshev':>16}{'speed-up':>12}")
    c = 0.5 * (lam_min + lam_max)
    r = 0.5 * (lam_max - lam_min)
    for k in (1, 2, 3, 4, 6):
        fixed = ((lam_max - lam_min) / (lam_max + lam_min)) ** k
        cheb = 1.0 / math.cosh(k * math.acosh(max(1.0, c / r)))
        lines.append(f"  {k:>4}{fixed:>16.8f}{cheb:>16.8f}{fixed / cheb:>11.1f}x")
    lines.append("")

    # -- parity with the SDK --------------------------------------------
    from geoformer.layers.tac import QuantizedTAC as SDKTAC

    torch.manual_seed(1)
    sdk = SDKTAC(d_model=d_model, tau_max=8, delta_tau=0.1,
                 lambda_dir=0.05, lambda_s5=0.05)
    book = BookQuantizedTAC(d_model=d_model, tau_max=8, delta_tau=0.1,
                            lambda_dir=0.05, lambda_s5=0.05)
    book.load_state_dict(sdk.state_dict())
    hh = torch.randn(2, N, d_model)
    with torch.no_grad():
        h_sdk, tau_sdk, k_sdk = sdk(hh, L)
        h_bk, tau_bk, k_bk = book(hh, L)
    lines.append("Parity — production SDK QuantizedTAC")
    lines.append(f"  max |h_book - h_sdk| = {float((h_bk - h_sdk).abs().max()):.3e}")
    lines.append(f"  actual_tau: book {tau_bk}, sdk {tau_sdk}   (equal: {tau_bk == tau_sdk})")
    lines.append(f"  |K_S book - K_S sdk| = {float((k_bk - k_sdk).abs()):.3e}")
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
