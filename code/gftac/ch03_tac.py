# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 3 — TAC: Thermodynamic Attractor Convergence.

The second life of the lattice. Chapter 2 built a *store*; this chapter
gives it a *dynamics*. An activation state ``s`` in R^N relaxes downhill
on a free-energy landscape ``F[s]``, and the guarantees are thermodynamic:
energy never rises, convergence is exponential with a computable
timescale, and a double-well landscape stores memory in its basins.

Book theorems proved in the chapter and pinned by the tests:

  B.3.1  Lyapunov dissipation — under the gradient flow
         ds/dtau = -kappa grad F, we have
         dF/dtau = -kappa ||grad F||^2 <= 0.
         Energy is a Lyapunov function; the flow can only descend.
  B.3.2  Groenwall convergence timescale — for a quadratic potential with
         smallest curvature lambda, the flow obeys
         ||s(tau) - s*|| <= e^{-kappa lambda tau} ||s(0) - s*||,
         so the time to reach tolerance eps is
         T_eps <= (1/(kappa lambda)) ln(||s(0)-s*||/eps).
  B.3.3  Double-well bistability — V[s] = sum_i (1/4)(s_i^2 - 1)^2 has
         global minima at s_i = +/-1 (V = 0) and a saddle barrier at
         s_i = 0 (V = 1/4). Each coordinate stores one bit.

Run the demonstration ledger::

    python3 code/gftac/ch03_tac.py
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional

import torch

__all__ = [
    "DoubleWellPotential",
    "QuadraticPotential",
    "TacRelaxer",
    "RelaxResult",
    "predicted_steps",
    "demo_ledger",
]


# ----------------------------------------------------------------------
# The landscapes
# ----------------------------------------------------------------------

class DoubleWellPotential:
    """The bistable potential ``V[s] = sum_i (1/4)(s_i^2 - 1)^2``.

    Each coordinate lives on a double well with minima at ``+1`` and ``-1``
    (both at energy 0) and a local maximum at ``0`` (energy ``1/4``). This
    is the elementary memory cell of the substrate: a coordinate that has
    relaxed to ``+1`` or ``-1`` holds one bit, and the barrier at ``0`` is
    what protects the bit against small perturbations.

    Gradient: ``grad V = s * (s^2 - 1)``, elementwise. It vanishes at
    ``s_i in {-1, 0, +1}``; the first two per coordinate are stable minima,
    the third is the unstable separatrix.
    """

    def energy(self, s: torch.Tensor) -> torch.Tensor:
        return (0.25 * (s ** 2 - 1.0) ** 2).sum()

    def grad(self, s: torch.Tensor) -> torch.Tensor:
        return s * (s ** 2 - 1.0)

    @staticmethod
    def minima() -> torch.Tensor:
        """The two global minima of a single coordinate."""
        return torch.tensor([-1.0, 1.0])

    def barrier_height(self) -> float:
        """Energy gap between the barrier at 0 and a minimum: ``1/4``."""
        return 0.25


class QuadraticPotential:
    """The convex potential ``F[s] = (1/2) s^T Q s`` with ``Q`` symmetric PSD.

    Gradient: ``grad F = Q s``. The smallest eigenvalue of ``Q`` — the
    *curvature* — sets the worst-case convergence rate of Theorem B.3.2.
    Two instances matter to this book:

    * ``QuadraticPotential.isotropic(n, lam)`` — ``Q = lam I``: every
      direction decays at the same rate, which is the clean setting for
      the dimension-independence experiment of §3.6.
    * ``QuadraticPotential.from_laplacian(L)`` — ``Q = L``: the Dirichlet
      energy of Chapter 2. Its smallest eigenvalue is 0 (the constant
      mode), so the operative rate is the *Fiedler value* ``lambda_2`` —
      the precise place where Chapter 2's algebraic connectivity becomes
      Chapter 3's convergence speed.
    """

    def __init__(self, Q: torch.Tensor):
        if Q.dim() != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix")
        self.Q = Q

    @classmethod
    def isotropic(cls, n: int, curvature: float = 1.0) -> "QuadraticPotential":
        return cls(curvature * torch.eye(n))

    @classmethod
    def from_laplacian(cls, L: torch.Tensor) -> "QuadraticPotential":
        return cls(L)

    def energy(self, s: torch.Tensor) -> torch.Tensor:
        return 0.5 * (s @ self.Q @ s)

    def grad(self, s: torch.Tensor) -> torch.Tensor:
        return self.Q @ s

    def min_curvature(self) -> float:
        """Smallest eigenvalue of ``Q`` (the worst-case rate's curvature)."""
        return float(torch.linalg.eigvalsh(self.Q)[0].item())

    def spectral_gap(self) -> float:
        """Second-smallest eigenvalue of ``Q``.

        For a Laplacian this is the Fiedler value ``lambda_2`` and is the
        rate that governs relaxation *on the mean-zero subspace* — the
        constant mode is a null direction of the Dirichlet energy.
        """
        evals = torch.linalg.eigvalsh(self.Q)
        if evals.numel() < 2:
            return 0.0
        return float(evals[1].item())


# ----------------------------------------------------------------------
# The relaxer
# ----------------------------------------------------------------------

class RelaxResult(NamedTuple):
    """Outcome of a relaxation run (tuple-unpackable, four fields)."""

    final_state: torch.Tensor
    steps: int
    energy_history: List[float]
    converged: bool


class TacRelaxer:
    """Euler–Maruyama relaxation of a potential under the TAC flow.

    The continuous dynamics is the overdamped Langevin equation

        ds/dtau = -kappa grad F(s) + sqrt(2 kappa T) xi(tau),

    with ``kappa`` the mobility (relaxation rate) and ``T`` the temperature.
    At ``T = 0`` this is the pure gradient flow ``ds/dtau = -kappa grad F``
    of Theorem B.3.1; at ``T > 0`` it is the stochastic version that lets
    the state escape shallow basins — the Kramers mechanism §3.7 previews.

    A single step of size ``h`` is

        s <- s - h kappa grad F(s) + sqrt(2 h kappa T) * N(0, I).

    Halting is by *drift*: the run stops when ``||kappa grad F(s)|| <= tol``,
    i.e. when the deterministic velocity has become negligible. This is the
    natural stopping rule for the flow and is what makes the step counts of
    §3.6 comparable across dimensions.
    """

    def __init__(
        self,
        potential,
        kappa: float = 1.0,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.potential = potential
        self.kappa = float(kappa)
        self.temperature = float(temperature)
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    def drift(self, s: torch.Tensor) -> torch.Tensor:
        """The deterministic velocity ``kappa grad F(s)``."""
        return self.kappa * self.potential.grad(s)

    def step(self, s: torch.Tensor, h: float) -> torch.Tensor:
        """One Euler–Maruyama step of size ``h``."""
        s_next = s - h * self.drift(s)
        if self.temperature > 0.0:
            std = math.sqrt(2.0 * h * self.kappa * self.temperature)
            noise = torch.randn(s.shape, generator=self.generator) * std
            s_next = s_next + noise
        return s_next

    def relax(
        self,
        s0: torch.Tensor,
        h: float = 0.05,
        max_steps: int = 5000,
        tol: float = 1e-6,
    ) -> RelaxResult:
        """Relax from ``s0`` until the drift norm falls to ``tol``.

        Returns a :class:`RelaxResult` carrying the final state, the number
        of steps taken, the energy at every recorded step (including the
        initial one), and whether the drift tolerance was met.
        """
        s = s0.clone()
        history: List[float] = [float(self.potential.energy(s))]
        converged = False
        steps = 0
        for _ in range(max_steps):
            if float(torch.linalg.vector_norm(self.drift(s))) <= tol:
                converged = True
                break
            s = self.step(s, h)
            steps += 1
            history.append(float(self.potential.energy(s)))
        return RelaxResult(s, steps, history, converged)


# ----------------------------------------------------------------------
# The Groenwall step prediction (Theorem B.3.2, discrete form)
# ----------------------------------------------------------------------

def predicted_steps(
    kappa: float,
    curvature: float,
    h: float,
    s0_norm: float,
    tol: float,
) -> int:
    """Steps for the isotropic quadratic to reach the drift tolerance.

    For ``F[s] = (1/2) lam ||s||^2`` the explicit Euler iterate is
    ``s_k = (1 - h kappa lam)^k s_0``, so the drift norm
    ``||kappa lam s_k|| = kappa lam (1 - h kappa lam)^k ||s_0||``. Setting
    it equal to ``tol`` and solving for ``k`` gives the prediction

        k = ln(tol / (kappa lam ||s_0||)) / ln(1 - h kappa lam).

    This is the discrete shadow of Theorem B.3.2's continuum timescale.
    Returns ``max_steps``-style infinity sentinel ``-1`` if the step is not
    in the stable range ``0 < 1 - h kappa lam < 1``.
    """
    r = 1.0 - h * kappa * curvature
    if not (0.0 < r < 1.0):
        return -1
    arg = tol / (kappa * curvature * s0_norm)
    if arg <= 0:
        return 0
    return int(math.ceil(math.log(arg) / math.log(r)))


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def demo_ledger() -> str:
    """The Chapter 3 demonstration: dissipation, timescale, basins, T."""
    lines: List[str] = []
    lines.append("Chapter 3 — TAC: relaxation, measured")
    lines.append("=" * 64)

    # -- B.3.1: Lyapunov dissipation on the double well ------------------
    n = 8
    s0 = 0.6 * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    relaxer = TacRelaxer(DoubleWellPotential(), kappa=1.0, temperature=0.0)
    res = relaxer.relax(s0, h=0.05, max_steps=2000, tol=1e-5)
    lines.append(
        f"B.3.1 — Lyapunov dissipation (double well, N={n}, "
        f"h=0.05, kappa=1.0, init s0=0.6*[+,-,+,-,...])"
    )
    lines.append(f"{'step':>6}{'energy F':>14}{'||grad F||':>14}{'-kappa||grad F||^2':>20}")
    checkpoints = [0, 1, 2, 5, 10, 20, res.steps]
    for k in checkpoints:
        # reconstruct the state at step k by replaying (cheap, N=8)
        s = s0.clone()
        for _ in range(k):
            s = relaxer.step(s, 0.05)
        g = relaxer.potential.grad(s)
        dFdt = -relaxer.kappa * float(g @ g)
        lines.append(
            f"{k:>6}{float(relaxer.potential.energy(s)):>14.6f}"
            f"{float(torch.linalg.vector_norm(g)):>14.6f}{dFdt:>20.6f}"
        )
    pattern = res.final_state[:4].tolist()
    lines.append(
        f"  final state[:4] = {pattern} -> two stable basins per coordinate; "
        f"energy strictly decreasing above"
    )
    lines.append("")

    # -- B.3.2: Groenwall timescale on the isotropic quadratic -----------
    lam = 1.0
    q = QuadraticPotential.isotropic(16, curvature=lam)
    r2 = TacRelaxer(q, kappa=1.0, temperature=0.0)
    s0q = torch.ones(16) / math.sqrt(16.0)  # unit norm
    lines.append(
        f"B.3.2 — Groenwall timescale (quadratic, N=16, lam={lam}, "
        f"kappa=1.0, h=0.05, ||s0||=1)"
    )
    lines.append(f"{'tol':>10}{'predicted':>12}{'actual':>10}")
    for tol in (1e-3, 1e-5, 1e-6):
        pred = predicted_steps(1.0, lam, 0.05, float(torch.linalg.vector_norm(s0q)), tol)
        actual = r2.relax(s0q, h=0.05, max_steps=5000, tol=tol).steps
        lines.append(f"{tol:>10.0e}{pred:>12}{actual:>10}")
    lines.append("")

    # -- B.3.3: double-well basins --------------------------------------
    lines.append("B.3.3 — double-well bistability (N=4, h=0.05, kappa=1.0)")
    for init in (0.01, -0.01, 0.0):
        s0b = init * torch.ones(4)
        rb = TacRelaxer(DoubleWellPotential(), kappa=1.0).relax(
            s0b, h=0.05, max_steps=2000, tol=1e-4
        )
        lines.append(
            f"  init s = {init:+.2f}*1  ->  final s = "
            f"{rb.final_state[0].item():+.6f}*1   ({rb.steps} steps)"
        )
    lines.append("")

    # -- §3.6: dimension independence -----------------------------------
    lines.append("§3.6 — dimension independence (isotropic quadratic, unit-norm init)")
    for nd in (10, 100, 1000):
        qd = QuadraticPotential.isotropic(nd, curvature=1.0)
        rd = TacRelaxer(qd, kappa=1.0).relax(
            torch.ones(nd) / math.sqrt(nd), h=0.05, max_steps=5000, tol=1e-6
        )
        lines.append(f"  N = {nd:>5}: {rd.steps} steps to the drift tolerance")
    lines.append("")

    # -- §3.7: temperature (Langevin / Kramers preview) ------------------
    lines.append("§3.7 — temperature: fluctuation and escape (Langevin preview)")
    lines.append(f"{'T':>8}{'std(s) measured':>18}{'sqrt(T/lam)':>14}")
    for T in (0.02, 0.08):
        qh = QuadraticPotential.isotropic(1, curvature=1.0)
        rh = TacRelaxer(qh, kappa=1.0, temperature=T, seed=11)
        finals = []
        for _ in range(300):
            s = torch.zeros(1)
            for _ in range(250):
                s = rh.step(s, 0.02)
            finals.append(float(s[0]))
        std = float(torch.tensor(finals).std())
        lines.append(f"{T:>8.2f}{std:>18.4f}{math.sqrt(T / 1.0):>14.4f}")
    lines.append("")
    lines.append(f"{'T':>8}{'escape fraction':>18}")
    for T in (0.04, 0.08, 0.12):
        esc = 0
        trials = 150
        for trial in range(trials):
            rw = TacRelaxer(DoubleWellPotential(), kappa=1.0, temperature=T, seed=1000 + trial)
            s = -1.0 * torch.ones(1)
            for _ in range(300):
                s = rw.step(s, 0.02)
            if float(s[0]) > 0.0:
                esc += 1
        lines.append(f"{T:>8.2f}{esc / trials:>18.3f}")
    lines.append("  (barrier dV = 0.25; escape rises with T — the Kramers preview)")
    lines.append("=" * 64)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
