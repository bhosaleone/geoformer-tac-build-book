"""
Exact Cover (X3C) Continuous Thermodynamic Relaxation
=====================================================
Maps Karp's NP-complete Exact Cover problem onto a continuous energy manifold
with soft-spin bifurcation potentials.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
from ..core.manifold import ContinuousManifold
from ..core.coprocessor import TACCoprocessor


class ExactCoverManifold(ContinuousManifold):
    """
    Continuous manifold for Exact Cover:
    Incidence matrix A in {0, 1}^(num_sets x num_elements).
    Coverage vector: c(s) = s^T A. Target coverage is 1 for every element.
    Penalty: F[s] = 1/M * ||s^T A - 1||^2 + lambda * sum_i s_i^2 * (1 - s_i)^2.
    """
    def __init__(self, A: np.ndarray, initial_state: np.ndarray, lambda_spin: float = 0.1):
        self.A = np.asarray(A, dtype=np.float64)
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.lambda_spin = float(lambda_spin)
        self.num_sets, self.num_elements = self.A.shape

    def get_energy(self, s: Optional[np.ndarray] = None) -> float:
        state = self.state if s is None else s
        coverage = np.dot(state, self.A)
        cov_res = float(np.sum((coverage - 1.0)**2)) / float(self.num_elements)
        spin_pen = self.lambda_spin * float(np.sum((state * (1.0 - state))**2))
        return cov_res + spin_pen

    def get_gradient(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        state = self.state if s is None else s
        coverage = np.dot(state, self.A)
        diff = coverage - 1.0

        # Gradient of coverage error
        grad_cov = 2.0 * np.dot(self.A, diff) / float(self.num_elements)

        # Gradient of spin penalty s^2 (1 - s)^2: 2 * s * (1 - s) * (1 - 2s) = 2 * s * (s - 1) * (2s - 1)
        grad_spin = 2.0 * state * (state - 1.0) * (2.0 * state - 1.0)

        return grad_cov + self.lambda_spin * grad_spin


class ExactCoverSolver:
    """
    Solves Exact Cover instances via continuous Langevin relaxation.
    """
    def __init__(self, kappa: float = 0.05, temperature: float = 0.005, max_steps: int = 1000):
        self.coprocessor = TACCoprocessor(
            kappa=kappa,
            temperature=temperature,
            equilibrium_threshold=0.05,
            max_steps=max_steps
        )

    def solve(self, A: np.ndarray) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        num_sets, num_elements = A.shape
        s0 = np.random.uniform(0.1, 0.9, num_sets)
        manifold = ExactCoverManifold(A, s0)

        start_time = time.perf_counter()
        s_relaxed, steps, diag = self.coprocessor.tac_offload(manifold, return_diagnostics=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Round continuous state to binary decisions
        binary_s = (s_relaxed > 0.5).astype(int)
        final_coverage = np.dot(binary_s, A)
        uncovered = int(np.sum(final_coverage == 0))
        overcovered = int(np.sum(final_coverage > 1))
        perfectly_covered = int(np.sum(final_coverage == 1))

        diag.update({
            "elapsed_ms": elapsed_ms,
            "uncovered": uncovered,
            "overcovered": overcovered,
            "perfectly_covered": perfectly_covered,
            "num_sets": num_sets,
            "num_elements": num_elements,
            "is_exact_cover": (uncovered == 0 and overcovered == 0)
        })
        return binary_s, steps, diag
