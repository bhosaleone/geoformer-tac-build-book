"""
Continuous 3-SAT Energy Formulation
===================================
Maps propositional 3-SAT instances onto a continuous energy functional:
    F[s] = 1/M * sum_c (prod_{l in c} (1 - val(l)))^2 + lambda * sum_i s_i^2 * (1 - s_i)^2

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from ..core.manifold import ContinuousManifold
from ..core.coprocessor import TACCoprocessor


class SATManifold(ContinuousManifold):
    """
    Continuous manifold for 3-SAT.
    clauses: List of tuples [(var_idx, sign), ...] with sign in {+1, -1}.
    """
    def __init__(self, clauses: List[List[Tuple[int, int]]], num_vars: int, initial_state: np.ndarray, lambda_spin: float = 0.1):
        self.clauses = clauses
        self.num_vars = int(num_vars)
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.lambda_spin = float(lambda_spin)

    def get_energy(self, s: Optional[np.ndarray] = None) -> float:
        state = self.state if s is None else s
        clause_cost = 0.0
        for clause in self.clauses:
            prod = 1.0
            for var_idx, sign in clause:
                val = state[var_idx] if sign == 1 else (1.0 - state[var_idx])
                prod *= (1.0 - val)
            clause_cost += prod ** 2

        spin_cost = self.lambda_spin * float(np.sum((state * (1.0 - state))**2))
        return (clause_cost / len(self.clauses)) + spin_cost

    def get_gradient(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        state = self.state if s is None else s
        grad = np.zeros(self.num_vars, dtype=np.float64)

        for clause in self.clauses:
            l_terms = []
            for var_idx, sign in clause:
                val = state[var_idx] if sign == 1 else (1.0 - state[var_idx])
                l_terms.append(1.0 - val)

            clause_prod = float(np.prod(l_terms))
            clause_cost_grad = 2.0 * clause_prod

            for i, (var_idx, sign) in enumerate(clause):
                other_terms_prod = float(np.prod(l_terms[:i] + l_terms[i+1:]))
                d_l_ds = -1.0 if sign == 1 else 1.0
                grad[var_idx] += clause_cost_grad * other_terms_prod * d_l_ds

        grad /= len(self.clauses)
        # Soft-spin penalty derivative
        spin_grad = 2.0 * state * (state - 1.0) * (2.0 * state - 1.0)
        return grad + self.lambda_spin * spin_grad


class SATSolver:
    """
    Continuous relaxation solver for 3-SAT.
    """
    def __init__(self, kappa: float = 0.02, temperature: float = 0.005, max_steps: int = 1500):
        self.coprocessor = TACCoprocessor(
            kappa=kappa,
            temperature=temperature,
            equilibrium_threshold=0.02,
            max_steps=max_steps
        )

    def solve(self, clauses: List[List[Tuple[int, int]]], num_vars: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        s0 = np.random.uniform(0.2, 0.8, num_vars)
        manifold = SATManifold(clauses, num_vars, s0)

        start_time = time.perf_counter()
        s_relaxed, steps, diag = self.coprocessor.tac_offload(manifold, return_diagnostics=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        final_s = (s_relaxed > 0.5).astype(int)
        satisfied = 0
        for clause in clauses:
            is_sat = False
            for var_idx, sign in clause:
                val = final_s[var_idx]
                if (sign == 1 and val == 1) or (sign == -1 and val == 0):
                    is_sat = True
                    break
            if is_sat:
                satisfied += 1

        diag.update({
            "elapsed_ms": elapsed_ms,
            "satisfied_clauses": satisfied,
            "total_clauses": len(clauses),
            "ratio": satisfied / len(clauses) if clauses else 1.0,
            "all_satisfied": (satisfied == len(clauses))
        })
        return final_s, steps, diag
