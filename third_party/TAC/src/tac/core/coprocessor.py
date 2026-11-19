r"""
Virtual Thermodynamic Automaton Computer (VTAC) Coprocessor
===========================================================
Simulates the physical thermodynamic relaxation of continuous manifolds
governed by Langevin stochastic differential equations:

    ds/dt = -kappa * \nabla_{\mathcal{S}} \mathcal{F}[s] + \sqrt{2 * kappa * T} * \eta(t)

Authors: Shrikant Bhosale (Atmabhan Pandit)
Affiliation: MBBS Student, GMC Sambhajinagar, Maharashtra, India
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .manifold import ContinuousManifold


class TACCoprocessor:
    """
    Thermodynamic Coprocessor Engine.
    Offloads classical continuous optimization problems to analog-style Langevin relaxation.
    """
    def __init__(
        self,
        kappa: float = 0.01,
        temperature: float = 0.0,
        dt: float = 1.0,
        equilibrium_threshold: float = 1e-4,
        max_steps: int = 2000,
        annealing_decay: float = 1.0
    ):
        """
        Args:
            kappa (float): Kinetic relaxation rate of the substrate.
            temperature (float): Ambient thermal noise (T).
            dt (float): Integration time step.
            equilibrium_threshold (float): Normalized gradient norm threshold for macroscopic convergence.
            max_steps (int): Upper bound on relaxation steps.
            annealing_decay (float): Multiplicative decay factor for temperature per step.
        """
        self.kappa = float(kappa)
        self.temperature = float(temperature)
        self.dt = float(dt)
        self.equilibrium_threshold = float(equilibrium_threshold)
        self.max_steps = int(max_steps)
        self.annealing_decay = float(annealing_decay)

    def tac_offload(
        self,
        manifold: Any,
        initial_state: Optional[np.ndarray] = None,
        return_diagnostics: bool = False
    ) -> Tuple[np.ndarray, int]:
        r"""
        Executes thermodynamic relaxation of the given continuous manifold.

        Args:
            manifold: Object implementing `get_gradient(s)` and optional `get_energy(s)`.
                      Can also have a `.state` attribute.
            initial_state: Optional initial state vector. If None, extracts `manifold.state`.
            return_diagnostics: If True, returns (state, steps, diagnostics_dict).

        Returns:
            Tuple of (relaxed_state, steps) for backwards compatibility.
        """
        if initial_state is not None:
            state = initial_state.copy().astype(np.float64)
        elif hasattr(manifold, "state"):
            state = manifold.state.copy().astype(np.float64)
        else:
            raise ValueError("Manifold must provide a `.state` attribute or explicit initial_state.")

        N = len(state)
        curr_temp = self.temperature
        steps = 0
        energy_history = []
        grad_norms = []

        while steps < self.max_steps:
            # 1. Evaluate driving force
            gradient = manifold.get_gradient(state)
            norm_grad = float(np.linalg.norm(gradient) / np.sqrt(N))
            grad_norms.append(norm_grad)

            if hasattr(manifold, "get_energy"):
                energy_history.append(float(manifold.get_energy(state)))

            # 2. Check macroscopic equilibrium
            if norm_grad < self.equilibrium_threshold and steps > 0:
                break

            # 3. Stochastic thermal Langevin perturbation
            if curr_temp > 1e-12:
                noise = np.random.randn(N) * np.sqrt(2.0 * self.kappa * curr_temp * self.dt)
            else:
                noise = 0.0

            # 4. Physical Langevin state update (Euler-Maruyama step)
            state -= (self.kappa * gradient * self.dt) + noise
            steps += 1

            # 5. Annealing decay
            curr_temp *= self.annealing_decay

        diagnostics = {
            "steps": steps,
            "final_grad_norm": grad_norms[-1] if grad_norms else 0.0,
            "energy_history": energy_history,
            "converged": (grad_norms[-1] < self.equilibrium_threshold) if grad_norms else True
        }

        if return_diagnostics:
            return state, steps, diagnostics
        return state, steps
