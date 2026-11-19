"""
Continuous Spin-Glass and Hilbert Space Mapper
==============================================
Maps discrete spin-glass Hamiltonians and quadratic optimization problems
into continuous thermodynamic cost functionals with soft-spin bifurcation potentials.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
from typing import Dict, Any, Tuple
from .manifold import ContinuousManifold


class ContinuousSpinManifold(ContinuousManifold):
    """
    Continuous energy manifold representing a coupled spin network:
        F[s] = - 1/2 * s^T J s - h^T s + lambda_bifurcation * sum_i (s_i^2 - 1)^2

    The double-well potential (s_i^2 - 1)^2 acts as a continuous bifurcation mechanism,
    driving real coordinates s_i in [-1, +1] toward discrete bipolar eigenstates {-1, +1}.
    """
    def __init__(
        self,
        J: np.ndarray,
        h: np.ndarray,
        initial_state: np.ndarray,
        lambda_bifurcation: float = 1.0
    ):
        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.lambda_bifurcation = float(lambda_bifurcation)
        self.N = len(self.state)

    def get_energy(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float64)
        ising_e = -0.5 * float(np.dot(s, np.dot(self.J, s))) - float(np.dot(self.h, s))
        well_e = self.lambda_bifurcation * float(np.sum((s**2 - 1.0)**2))
        return ising_e + well_e

    def get_gradient(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        # Gradient of -1/2 s^T J s is -J s (for symmetric J)
        # Gradient of -h^T s is -h
        # Gradient of lambda * (s^2 - 1)^2 is 4 * lambda * s * (s^2 - 1)
        ising_grad = -np.dot(self.J, s) - self.h
        well_grad = 4.0 * self.lambda_bifurcation * s * (s**2 - 1.0)
        return ising_grad + well_grad


class HilbertSpaceMapper:
    """
    Constructs continuous spin manifolds over topological substrates.
    """
    def __init__(self, topology: Any):
        self.topology = topology
        self.N = topology.num_nodes
        self._edge_list = self._precompute_edges()

    def _precompute_edges(self) -> np.ndarray:
        edges = []
        for i in range(self.N):
            for j in self.topology.adjacency_list.get(i, []):
                if i < j:
                    edges.append((i, j))
        if edges:
            return np.array(edges, dtype=np.int32)
        return np.empty((0, 2), dtype=np.int32)

    def construct_cost_functional(
        self,
        quantum_coupling_matrix: np.ndarray,
        lambda_bifurcation: float = 1.0
    ) -> ContinuousSpinManifold:
        """
        Creates a ContinuousSpinManifold from the given coupling matrix J.
        """
        J = np.asarray(quantum_coupling_matrix, dtype=np.float64)
        h = np.zeros(self.N, dtype=np.float64)
        s0 = np.random.uniform(-0.1, 0.1, self.N)
        return ContinuousSpinManifold(J, h, s0, lambda_bifurcation=lambda_bifurcation)
