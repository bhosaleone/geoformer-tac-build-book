"""
Turing-TAC Coupler: Continuous Thermodynamic Architecture
=========================================================
A continuous-variable dynamical systems and non-equilibrium thermodynamic relaxation framework
for combinatorial optimization, manifold ringdown physics, and group-theoretic projections.

Mathematical invariants formally verified via Microsoft Z3 SMT Theorem Prover.

Authors: Shrikant Bhosale (Atmabhan Pandit)
Affiliation: MBBS Student, GMC Sambhajinagar, Maharashtra, India
"""

__version__ = "1.0.0"
__author__ = "Shrikant Bhosale (Atmabhan Pandit)"
__license__ = "MIT (Software) / CC BY 4.0 (Documentation)"

from .core.coprocessor import TACCoprocessor
from .core.manifold import ContinuousManifold
from .core.topology import CrystalTopology5D, FlexibleCrystal5D
from .core.hilbert_mapper import HilbertSpaceMapper, ContinuousSpinManifold
from .physics.ringdown import BlackHoleManifold, RuggedManifold
from .physics.hawking import HawkingEvaporator
from .physics.s5 import S5ProjectionFramework
from .combinatorial.exact_cover import ExactCoverManifold, ExactCoverSolver
from .combinatorial.sat import SATManifold, SATSolver
from .proofs.z3_verifier import TACZ3Verifier

__all__ = [
    "__version__",
    "__author__",
    "TACCoprocessor",
    "ContinuousManifold",
    "CrystalTopology5D",
    "FlexibleCrystal5D",
    "HilbertSpaceMapper",
    "ContinuousSpinManifold",
    "BlackHoleManifold",
    "RuggedManifold",
    "HawkingEvaporator",
    "S5ProjectionFramework",
    "ExactCoverManifold",
    "ExactCoverSolver",
    "SATManifold",
    "SATSolver",
    "TACZ3Verifier",
]
