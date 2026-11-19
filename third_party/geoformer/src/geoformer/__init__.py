"""
geoformer-sdk: Quantized Propagation Geometry & Neuromorphic GeoFormer Framework.
Author: Atmabhan Pandit (Shrikant Bhosale)
TWIST POOL Labs │ ISL Research │ NanoCERN Unit, Sambhaji Nagar / Pune
"""

__version__ = "0.1.0"

# 1. Quantization Sub-system
from .quant.ste import RoundSTE, SignSTE, AbsMaxQuantizer
from .quant.init import variance_preserving_init_
from .quant.ternary_linear import QuantizedLinear

# 2. Metric Geometry & Graph Dynamics
from .core.laplacian import GraphLaplacian
from .core.metric_warping import DiscreteMetricWarpingEngine

# 3. Geometric Attention & Continuous Relaxation Layers
from .layers.ffn import TernarySwiGLU
from .layers.cayley import S5CayleyProjector
from .layers.attention import QuantizedGHA
from .layers.tac import QuantizedTAC
from .layers.emergent_multihop import (
    EmergentHopAttractor,
    GoalConditionedEmergentHopAttractor,
    DeepHopTransitiveRouter,
)

# 4. Neural Network Models
from .models.geoformer import GeoFormer, GeoFormerBlock
from .models.block import QGeoFormerBlock, RMSNorm
from .models.relational import RelationalQGeoFormer
from .models.transitive import DeepTransitiveModel

# 5. Neuromorphic Memristor Hardware & Energy
from .hardware.crossbar import PhysicalMemristorCrossbar
from .hardware.emulator import NeuromorphicQGeoFormer
from .hardware.energy import NanojouleEnergyAuditor

# 6. Optimization
from .optim.kramers import DualPathKramersOptimizer

# 7. Training Pipeline & Model Setup
from .training.config import TrainingConfig
from .training.losses import GeoFormerTACLoss
from .training.model import GeoFormerTACModel
from .training.trainer import GeoFormerTACTrainer
from .training.datasets import (
    TransitiveKnowledgeGraphDataset,
    GraphNodeClassificationDataset,
    create_transitive_dataloaders,
)

# 8. Autonomous Endogenous Dynamics & Schedulers
from .autonomous.schedules import (
    continued_fraction_convergents,
    fibonacci_schedule,
    make_relaxation_schedule,
)
from .autonomous.encodings import (
    quasi_crystalline_encoding,
    encoding_collision_rate,
)
from .autonomous.endogenous_engine import (
    ConsolidationConfig,
    ConsolidationReport,
    EndogenousConsolidator,
)

__all__ = [
    "__version__",
    "GeoFormer",
    "GeoFormerBlock",
    "RoundSTE",
    "SignSTE",
    "AbsMaxQuantizer",
    "variance_preserving_init_",
    "QuantizedLinear",
    "GraphLaplacian",
    "DiscreteMetricWarpingEngine",
    "TernarySwiGLU",
    "S5CayleyProjector",
    "QuantizedGHA",
    "QuantizedTAC",
    "EmergentHopAttractor",
    "GoalConditionedEmergentHopAttractor",
    "DeepHopTransitiveRouter",
    "QGeoFormerBlock",
    "RMSNorm",
    "RelationalQGeoFormer",
    "DeepTransitiveModel",
    "PhysicalMemristorCrossbar",
    "NeuromorphicQGeoFormer",
    "NanojouleEnergyAuditor",
    "DualPathKramersOptimizer",
    "TrainingConfig",
    "GeoFormerTACLoss",
    "GeoFormerTACModel",
    "GeoFormerTACTrainer",
    "TransitiveKnowledgeGraphDataset",
    "GraphNodeClassificationDataset",
    "create_transitive_dataloaders",
    "continued_fraction_convergents",
    "fibonacci_schedule",
    "make_relaxation_schedule",
    "quasi_crystalline_encoding",
    "encoding_collision_rate",
    "ConsolidationConfig",
    "ConsolidationReport",
    "EndogenousConsolidator",
]

# Vendored parity subset: unrelated security package omitted (see THIRD_PARTY_NOTICES.md).

