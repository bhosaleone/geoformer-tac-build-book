# SPDX-License-Identifier: AGPL-3.0-only
"""
gftac — the GeoFormer & TAC build-along library.

Every module in this package is constructed chapter-by-chapter in the book
``GeoFormer & TAC — The Build Book``. Modules are named by the chapter in which
they are forged (ch01_..., ch02_..., ...). Where a module mirrors a component
of the production SDK (geoformer/src/geoformer), the chapter's test suite
contains a parity test asserting both implementations agree.

Book Theorems are numbered B.<chapter>.<n> to stay distinct from the research
program's Q-n theorem series.
"""

__version__ = "0.1.0"

# Canonical Model: Eklavya (एकलव्य)
from gftac.ch11_integrated_model import (
    EklavyaConfig,
    EklavyaModel,
    EklavyaLoss,
)

__all__ = [
    "__version__",
    "EklavyaConfig",
    "EklavyaModel",
    "EklavyaLoss",
]
