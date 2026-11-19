"""
Continuous Manifold Abstract Base Class for TAC
================================================
Defines the mathematical contract for continuous thermodynamic energy landscapes.
"""

from abc import ABC, abstractmethod
import numpy as np


class ContinuousManifold(ABC):
    """
    Abstract base class for high-dimensional continuous energy manifolds.
    Any physical system or combinatorial problem offloaded to the TAC coprocessor
    must implement get_energy(s) and get_gradient(s).
    """

    @abstractmethod
    def get_gradient(self, state: np.ndarray) -> np.ndarray:
        r"""
        Computes the gradient vector \nabla_{\mathcal{S}} \mathcal{F}[s].
        Acts as the thermodynamic driving force in the Langevin equation:
        \dot{s} = -\kappa \nabla_{\mathcal{S}} \mathcal{F}[s] + \sqrt{2 \kappa T} \eta(t)
        """
        pass

    def get_energy(self, state: np.ndarray) -> float:
        r"""
        Evaluates the scalar cost functional \mathcal{F}[s].
        Optional for pure gradient flow, but required for Lyapunov monitoring and energy auditing.
        """
        return 0.5 * float(np.sum(state ** 2))
