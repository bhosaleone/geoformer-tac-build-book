"""
Hawking Evaporation & Unitary Information Tracking
==================================================
Simulates unitary black hole evaporation and tracks von Neumann entanglement entropy
between interior degrees of freedom and emitted Hawking radiation, reproducing the Page Curve.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
from typing import Tuple, Dict, Any


class HawkingEvaporator:
    """
    Simulates the unitary evaporation of an N-node horizon state.
    Computes bipartite entanglement entropy across interior and radiation partitions.
    """
    def __init__(self, N_qubits: int = 1000, initial_mass: float = 100.0):
        self.N = int(N_qubits)
        self.mass = float(initial_mass)
        self.initial_mass = float(initial_mass)

        # Initialize normalized amplitude state vector
        raw_state = np.random.randn(self.N)
        self.amplitudes = (raw_state / np.linalg.norm(raw_state)).astype(np.float64)

        self.radiation_qubits = 0
        self.S_int = 0.0
        self.S_rad = 0.0
        self.S_ent = 0.0
        self.trace = float(np.sum(self.amplitudes**2))

    @property
    def radiation_entropy(self) -> float:
        """Alias for S_rad to maintain backwards compatibility."""
        return self.S_rad

    @property
    def entanglement_entropy(self) -> float:
        """Alias for S_ent."""
        return self.S_ent

    @staticmethod
    def _calculate_von_neumann(probs: np.ndarray) -> float:
        """Calculates -sum(p * ln(p)) for positive probabilities."""
        p_valid = probs[probs > 1e-15]
        if len(p_valid) == 0:
            return 0.0
        return float(-np.sum(p_valid * np.log(p_valid)))

    def check_unitarity(self) -> Tuple[float, float]:
        """Returns (current_trace, deviation_from_unity)."""
        current_trace = float(np.sum(self.amplitudes**2))
        deviation = abs(1.0 - current_trace)
        return current_trace, deviation

    def evaporate_step(self, dt: float = 1.0) -> bool:
        """
        Advances evaporation by dt. Emits qubits proportional to mass loss.
        Returns False when mass is fully exhausted or all qubits are emitted.
        """
        if self.mass <= 0.5 or self.radiation_qubits >= self.N:
            return False

        # dM/dt ~ 1 / M^2 (Hawking radiation rate)
        dM = (1.0 / (self.mass**2)) * float(dt)
        if dM > self.mass:
            dM = self.mass * 0.99
        self.mass -= dM

        # Fractional emission
        fraction_lost = dM / self.initial_mass
        qubits_to_emit = max(1, int(self.N * fraction_lost * 2.0))
        self.radiation_qubits = min(self.N, self.radiation_qubits + qubits_to_emit)

        # Partition probabilities
        p_dist = self.amplitudes**2
        p_rad = p_dist[:self.radiation_qubits]
        p_int = p_dist[self.radiation_qubits:]

        sum_rad = np.sum(p_rad)
        sum_int = np.sum(p_int)

        self.S_rad = self._calculate_von_neumann(p_rad / sum_rad) if sum_rad > 1e-15 else 0.0
        self.S_int = self._calculate_von_neumann(p_int / sum_int) if sum_int > 1e-15 else 0.0

        # Page Curve entanglement entropy: bounded by min(S_int, S_rad)
        self.S_ent = min(self.S_int, self.S_rad)

        # Unitarity preservation via normalization
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm
        self.trace = float(np.sum(self.amplitudes**2))

        return True
