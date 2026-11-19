"""
Blind Query-Conditioned Attractor Module (Level 33).
Aliases and integrates BlindPG v0.2.1 for seamless pipeline compatibility.
"""

from .blind_pg import BlindPG

# Alias for backward compatibility
BlindQueryConditionedAttractor = BlindPG

__all__ = ["BlindQueryConditionedAttractor", "BlindPG"]
