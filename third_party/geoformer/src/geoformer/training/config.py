"""
Training Configuration for GeoFormer and TAC Architecture Models.
Defines all architectural, thermodynamic, optimization, and training hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
import os


@dataclass
class TrainingConfig:
    """
    Comprehensive Hyperparameter Configuration for Training GeoFormer + TAC.
    """
    # 1. Experiment & Task Metadata
    experiment_name: str = "geoformer_tac_default"
    task_type: str = "transitive_multihop"  # "transitive_multihop" | "node_classification" | "sequence_reasoning"
    seed: int = 42
    device: str = "cpu"  # "cpu" | "cuda"

    # 2. Model Architecture Hyperparameters
    num_entities: int = 256
    num_relations: int = 8
    num_classes: int = 256
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    mode: str = "ternary"  # "ternary" (BitNet b1.58 STE) | "int8" | "fp32"
    dropout: float = 0.1
    node_encoding: str = "learned"  # "learned" | "quasi"

    # 3. Continuous TAC Relaxation Parameters (Layer Level)
    tau_max: int = 3
    lambda_geo: float = 0.10
    lambda_dir_layer: float = 0.05
    schedule: Optional[List[int]] = None

    # 4. Goal-Conditioned Multi-Hop Attractor (Attractor Level)
    max_reasoning_hops: int = 32
    gamma_0: float = 2.5
    beta_adaptive: float = 0.25
    gamma_max: float = 4.5
    kappa_attractor: float = 0.5
    eps_tac: float = 1e-4
    eps_q: float = 0.15
    lambda_dir_attractor: float = 1.0
    lambda_v_attractor: float = 0.2
    lambda_q_attractor: float = 1.0
    eta_residual: float = 0.05
    confidence_thresh: float = 0.65
    metric_boost_scale: float = 20.0

    # 5. Fast Plastic Conductance Lattice
    delta_w: float = 1.0 / 15.0  # 4-bit lattice step
    plastic_rate: float = 1.0     # Hebbian scaling per fact

    # 6. Optimization Hyperparameters
    optimizer_type: str = "kramers"  # "kramers" (Dual Path Kramers) | "adamw" | "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    kramers_beta1: float = 0.90
    kramers_alpha_tau: float = 0.20
    kramers_d0: float = 1e-4
    kramers_gamma_r: float = 0.50
    lr_warmup_epochs: int = 2
    lr_decay_type: str = "cosine"  # "cosine" | "linear" | "none"

    # 7. Loss Weights
    lambda_task: float = 1.0
    lambda_dir: float = 0.05       # Dirichlet energy penalty
    lambda_lyap: float = 0.01      # Lyapunov equilibrium drift penalty
    lambda_ent: float = 0.01       # Soliton Shannon entropy penalty
    lambda_quant: float = 0.001    # Ternary weight regularization penalty

    # 8. Training Loop Execution
    batch_size: int = 32
    epochs: int = 20
    eval_interval: int = 1
    eval_depths: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 24, 32])
    checkpoint_dir: str = "checkpoints/geoformer_tac"
    save_best_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
