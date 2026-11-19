"""
Event Horizon Ringdown Continuous Manifold
=========================================
Models the thermodynamic relaxation of perturbed black hole horizons during ringdown.
Demonstrates dimension-independent O(1) continuous relaxation on harmonic spacetime metrics.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
from typing import Optional
from ..core.manifold import ContinuousManifold


class BlackHoleManifold(ContinuousManifold):
    """
    Models the continuous deviation of an event horizon from stationary Kerr equilibrium.
    Equilibrium corresponds to the origin s = 0.
    Cost functional: F[s] = 1/2 * k * sum(s_i^2)  (harmonic quasi-normal modes).
    """
    def __init__(self, N_dimensions: int = 100, initial_distortion: float = 1.0, stiffness: float = 1.0):
        self.N = int(N_dimensions)
        self.stiffness = float(stiffness)
        self.state = np.random.randn(self.N).astype(np.float64) * float(initial_distortion)
        self.mass = 1.0
        self.spin = 0.0

    def get_energy(self, current_state: Optional[np.ndarray] = None) -> float:
        s = self.state if current_state is None else current_state
        return 0.5 * self.stiffness * float(np.sum(s ** 2))

    def get_gradient(self, current_state: Optional[np.ndarray] = None) -> np.ndarray:
        s = self.state if current_state is None else current_state
        return self.stiffness * s


class RuggedManifold(ContinuousManifold):
    """
    Non-convex, multimodal energy landscape combining harmonic restoring force with sinusoidal ripples:
        F[s] = sum(s_i^2) + sum(sin(omega * s_i))
        grad F[s] = 2 * s_i + omega * cos(omega * s_i)
    """
    def __init__(self, initial_state: np.ndarray, omega: float = 10.0):
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.N = len(self.state)
        self.omega = float(omega)

    def get_energy(self, state: Optional[np.ndarray] = None) -> float:
        s = self.state if state is None else state
        return float(np.sum(s**2) + np.sum(np.sin(self.omega * s)))

    def get_gradient(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        s = self.state if state is None else state
        return 2.0 * s + self.omega * np.cos(self.omega * s)
