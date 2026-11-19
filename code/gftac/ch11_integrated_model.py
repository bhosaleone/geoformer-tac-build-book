# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 11 — The Integrated Model & Loss Suite.

Assembles the full, unified foundation model: BookGeoFormerTACModel, combining:
  1. Token & Relation Embeddings + Golden-Ratio Quasi-Crystalline Positional Encodings
  2. Stacked BookGeoFormerBlocks (Pre-RMSNorm Q-GHA -> Ternary SwiGLU -> Continuous Q-TAC)
  3. Fast-Timescale Plastic Conductance Lattice W >= 0
  4. Goal-Conditioned Thermodynamic Attractor Convergence (Q-TAC Attractor)
  5. The Complete Thermodynamic Loss Suite:
       L_total = L_CE + lambda_dir * E_Dir + lambda_entropy * H(z) + lambda_lyap * L_lyap + lambda_quant * R(theta)

Theorems proved in Chapter 11:
  B.11.1  Thermodynamic Loss Convexity & Energy Lower Bound:
          The composite loss functional L_total is bounded below by zero and strictly
          penalizes non-smooth graph diffusion (E_Dir >= 0) and high-entropy modal
          dispersion (H(z) >= 0).
  B.11.2  Two-Timescale Gradient Decoupling:
          Gradients flowing backward from L_CE and L_total update continuous synaptic
          latents theta via STE without modifying the discrete conductance lattice W,
          which updates exclusively via local Hebbian increments.
  B.11.3  Pre-RMSNorm Residual Gradient Highways:
          In the deep L-block stack, the backward Jacobian ||d h_0 / d h_L|| remains
          strictly non-vanishing, bounded as ||I + sum J_l|| without exponential decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import torch
import torch.nn as nn
import torch.nn.functional as F

from gftac.ch04_ternary import TernaryLinear
from gftac.ch06_swiglu import BookRMSNorm
from gftac.ch09_model import (
    BookGeoFormerBlock,
    quasi_crystalline_encoding,
)
from gftac.ch10_plasticity import PlasticConductanceEngine, PlasticityConfig

__all__ = [
    "BookModelConfig",
    "BookAttractor",
    "BookGeoFormerTACModel",
    "BookGeoFormerTACLoss",
    "EklavyaConfig",
    "EklavyaModel",
    "EklavyaLoss",
    "demo_ledger",
]


@dataclass
class BookModelConfig:
    """Hyperparameter configuration for the integrated GeoFormer + TAC architecture."""
    num_entities: int = 16
    num_relations: int = 8
    num_classes: Optional[int] = None
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.0
    bits_w: int = 4
    w_max: float = 1.0
    delta_w: float = 1.0 / 15.0
    plastic_rate: float = 1.0
    max_reasoning_hops: int = 8
    metric_boost_scale: float = 5.0
    gamma_sharpen: float = 2.0
    eps_tac: float = 1e-4
    lambda_dir: float = 0.05
    lambda_entropy: float = 0.01
    lambda_lyap: float = 0.01
    lambda_quant: float = 1e-4
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.num_classes is None:
            self.num_classes = self.num_entities


class BookAttractor(nn.Module):
    """Goal-Conditioned Thermodynamic Attractor Engine.

    Executes continuous relaxation over the plastic metric lattice W,
    relaxing probability mass z downhill along the graph transition kernel.
    """

    def __init__(
        self,
        num_entities: int = 16,
        max_hops: int = 8,
        gamma_sharpen: float = 2.0,
        eps_tac: float = 1e-4,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.max_hops = max_hops
        self.gamma_sharpen = gamma_sharpen
        self.eps_tac = eps_tac

    def normalize_transition_kernel(self, W: torch.Tensor) -> torch.Tensor:
        """Constructs row-stochastic transition kernel P = D^{-1} W with honest dead ends."""
        row_sums = W.sum(dim=-1, keepdim=True)
        # Avoid division by zero: dead-end rows remain zero
        mask = row_sums > 1e-8
        P = torch.zeros_like(W)
        P = torch.where(mask, W / torch.clamp(row_sums, min=1e-8), P)
        return P

    def forward(
        self,
        source_indices: torch.Tensor,
        W: torch.Tensor,
        fixed_hops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relaxes initial one-hot mass at source along the metric lattice.

        Returns:
            z: Final settled probability mass distribution [B, N]
            emergent_hops: Number of steps taken until equilibrium [B]
            lyapunov_drift: Final drift ||z_{t+1} - z_t|| [B]
        """
        B = source_indices.shape[0]
        N = self.num_entities
        device = source_indices.device

        # 1. Initialize distribution z_0 as one-hot mass at source entity
        z = F.one_hot(source_indices, num_classes=N).float().to(device)

        # 2. Row-stochastic kernel P
        P = self.normalize_transition_kernel(W)

        steps_budget = fixed_hops if fixed_hops is not None else self.max_hops
        emergent_hops = torch.zeros(B, dtype=torch.long, device=device)
        is_converged = torch.zeros(B, dtype=torch.bool, device=device)
        last_drift = torch.zeros(B, device=device)

        for step in range(1, steps_budget + 1):
            # Graph propagation step: z_next = z @ P
            z_prop = torch.matmul(z, P)
            mass = z_prop.sum(dim=-1, keepdim=True)
            has_mass = mass > 1e-6

            # Soliton sharpening: non-linear contrastive exponent to prevent path dilution
            z_pow = torch.pow(torch.clamp(z_prop, min=0.0), self.gamma_sharpen)
            denom = torch.clamp(z_pow.sum(dim=-1, keepdim=True), min=1e-9)
            z_sharpened = z_pow / denom

            # If node has no outgoing edges (honest dead end), preserve current state z
            z_norm = torch.where(has_mass, z_sharpened, z)

            # Drift norm
            drift = torch.norm(z_norm - z, p=2, dim=-1)
            last_drift = drift

            if fixed_hops is None:
                newly_converged = (~is_converged) & (drift < self.eps_tac)
                emergent_hops = torch.where(newly_converged, torch.tensor(step, device=device), emergent_hops)
                is_converged = is_converged | newly_converged
                z = torch.where(is_converged.unsqueeze(-1), z, z_norm)
                if is_converged.all():
                    break
            else:
                z = z_norm

        if fixed_hops is not None:
            emergent_hops.fill_(fixed_hops)
        else:
            emergent_hops = torch.where(emergent_hops == 0, torch.tensor(steps_budget, device=device), emergent_hops)

        return z, emergent_hops, last_drift


class BookGeoFormerTACModel(nn.Module):
    """The unified GeoFormer + TAC Trainable Foundation Model.

    Combines:
      - Entity and relation embeddings
      - Quasi-crystalline positional encodings
      - Stack of BookGeoFormerBlocks
      - Final RMSNorm + Classifier Linear head
      - Fast-timescale plastic lattice W_conductance
      - Goal-conditioned attractor relaxation
    """

    def __init__(self, config: Optional[BookModelConfig] = None):
        super().__init__()
        self.config = config or BookModelConfig()

        # 1. Embeddings
        self.entity_emb = nn.Embedding(self.config.num_entities, self.config.d_model)
        self.relation_emb = nn.Embedding(self.config.num_relations, self.config.d_model)

        # 2. Stacked GeoFormer Blocks
        self.blocks = nn.ModuleList([
            BookGeoFormerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])

        # 3. Output normalization and classifier
        self.final_norm = BookRMSNorm(self.config.d_model)
        self.classifier = TernaryLinear(self.config.d_model, self.config.num_classes)

        # 4. Fast-timescale plastic conductance lattice W [N, N]
        self.register_buffer(
            "W_conductance",
            torch.zeros(self.config.num_entities, self.config.num_entities, dtype=torch.float32)
        )

        # 5. Thermodynamic Attractor
        self.attractor = BookAttractor(
            num_entities=self.config.num_entities,
            max_hops=self.config.max_reasoning_hops,
            gamma_sharpen=self.config.gamma_sharpen,
            eps_tac=self.config.eps_tac,
        )

    def update_conductance(
        self,
        src_indices: torch.Tensor,
        tgt_indices: torch.Tensor,
        weight_inc: float = 1.0,
    ) -> None:
        """Fast-timescale Hebbian update along observed edges."""
        with torch.no_grad():
            delta = self.config.delta_w * self.config.plastic_rate * float(weight_inc)
            s_list = src_indices.detach().cpu().view(-1).tolist()
            t_list = tgt_indices.detach().cpu().view(-1).tolist()
            for s, t in zip(s_list, t_list):
                s_idx, t_idx = int(s), int(t)
                if s_idx == t_idx:
                    continue
                cur = float(self.W_conductance[s_idx, t_idx].item())
                self.W_conductance[s_idx, t_idx] = min(self.config.w_max, cur + delta)
            self.W_conductance.fill_diagonal_(0.0)

    def forward(
        self,
        src_indices: torch.Tensor,
        rel_indices: Optional[torch.Tensor] = None,
        goal_indices: Optional[torch.Tensor] = None,
        fixed_hops: Optional[int] = None,
        d_matrix: Optional[torch.Tensor] = None,
        L_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through neural representations + attractor dynamics."""
        B = src_indices.shape[0]

        # 1. Embeddings
        e_emb = self.entity_emb(src_indices)  # [B, D]
        if rel_indices is not None:
            r_emb = self.relation_emb(rel_indices)  # [B, D]
            h = (e_emb + r_emb).unsqueeze(1)        # [B, 1, D]
        else:
            h = e_emb.unsqueeze(1)                  # [B, 1, D]

        # 2. Pass through GeoFormer Blocks
        for block in self.blocks:
            h = block(h, d_matrix=d_matrix, L_matrix=L_matrix)

        h_feat = self.final_norm(h).squeeze(1)  # [B, D]
        neural_logits = self.classifier(h_feat) # [B, num_classes]

        # 3. Attractor Relaxation on plastic conductance lattice
        attractor_z, emergent_hops, lyap_drift = self.attractor(
            source_indices=src_indices,
            W=self.W_conductance,
            fixed_hops=fixed_hops,
        )

        # 4. Row-stochastic transition kernel P_dir
        P_dir = self.attractor.normalize_transition_kernel(self.W_conductance)

        # 5. Geodesic Metric Highway Boost
        metric_boost = attractor_z * self.config.metric_boost_scale

        # 6. Fused logits
        if rel_indices is not None:
            # 1-hop facts (rel == 0) use direct neural logits; multi-hop uses metric boost
            is_direct = (rel_indices == 0).unsqueeze(-1)
            final_logits = torch.where(is_direct, neural_logits, neural_logits + metric_boost)
        else:
            final_logits = neural_logits + metric_boost

        return {
            "logits": final_logits,
            "neural_logits": neural_logits,
            "attractor_z": attractor_z,
            "emergent_hops": emergent_hops,
            "lyap_drift": lyap_drift,
            "P_dir": P_dir,
            "h_feat": h_feat,
        }

    def count_parameters(self) -> Tuple[int, float]:
        """Returns total parameter count and BitNet b1.58 packed memory footprint (KB)."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        packed_kb = (total_params * 0.20) / 1024.0
        return total_params, packed_kb


class BookGeoFormerTACLoss(nn.Module):
    """The unified Thermodynamic Loss Suite for GeoFormer + TAC.

    L_total = L_CE + lambda_dir * E_Dir + lambda_entropy * H(z) + lambda_lyap * L_lyap + lambda_quant * R(theta)
    """

    def __init__(self, config: Optional[BookModelConfig] = None):
        super().__init__()
        self.config = config or BookModelConfig()
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_dirichlet_energy(self, z: torch.Tensor, P_dir: torch.Tensor) -> torch.Tensor:
        """Computes the directed Dirichlet energy of the mass flow z:

            E_Dir(z) = 1/2 sum_b z_b (I - P_dir) z_b^T
        """
        z_diff = z - torch.matmul(z, P_dir)
        dirichlet_energy = 0.5 * torch.sum(z * z_diff, dim=-1).mean()
        return torch.clamp(dirichlet_energy, min=0.0)

    def compute_shannon_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Computes Shannon entropy H(z) = -sum_j z_j log2(z_j + eps)."""
        probs = torch.clamp(z, min=1e-12)
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1).mean()
        return entropy

    def compute_ternary_regularization(self, model: nn.Module) -> torch.Tensor:
        """Computes soft ternary penalty driving weights towards {-1, 0, +1}:

            R(w) = mean((|w| - 1)^2 * |w|)
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0
        for name, p in model.named_parameters():
            if "weight" in name and p.dim() >= 2:
                w_abs = torch.abs(p)
                penalty = ((w_abs - 1.0) ** 2) * w_abs
                reg_loss = reg_loss + penalty.mean()
                count += 1
        return reg_loss / max(1, count)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        attractor_z: Optional[torch.Tensor] = None,
        P_dir: Optional[torch.Tensor] = None,
        lyap_drift: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculates total loss and individual diagnostic terms."""
        # 1. Primary Task Cross-Entropy
        task_loss = self.ce_loss(logits, targets)

        # 2. Directed Dirichlet Energy
        if attractor_z is not None and P_dir is not None:
            dir_loss = self.compute_dirichlet_energy(attractor_z, P_dir)
        else:
            dir_loss = torch.tensor(0.0, device=logits.device)

        # 3. Lyapunov Drift Stationarity Penalty
        if lyap_drift is not None:
            lyap_loss = torch.mean(lyap_drift ** 2)
        else:
            lyap_loss = torch.tensor(0.0, device=logits.device)

        # 4. Attractor Shannon Entropy
        if attractor_z is not None:
            ent_loss = self.compute_shannon_entropy(attractor_z)
        else:
            ent_loss = torch.tensor(0.0, device=logits.device)

        # 5. Ternary Weight Regularization
        quant_loss = self.compute_ternary_regularization(model)

        total_loss = (
            task_loss
            + self.config.lambda_dir * dir_loss
            + self.config.lambda_entropy * ent_loss
            + self.config.lambda_lyap * lyap_loss
            + self.config.lambda_quant * quant_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "dirichlet_loss": float(dir_loss.item()),
            "entropy_loss": float(ent_loss.item()),
            "lyap_loss": float(lyap_loss.item()),
            "quant_loss": float(quant_loss.item()),
        }

        return total_loss, loss_dict


# Canonical Eklavya Architecture Bindings
EklavyaConfig = BookModelConfig
EklavyaModel = BookGeoFormerTACModel
EklavyaLoss = BookGeoFormerTACLoss


def demo_ledger() -> Dict[str, Any]:
    """Demonstrates forward pass, backward pass, and loss decomposition."""
    torch.manual_seed(42)
    cfg = BookModelConfig(num_entities=8, num_relations=2, num_classes=8, d_model=32, n_layers=2)
    model = BookGeoFormerTACModel(cfg)
    loss_fn = BookGeoFormerTACLoss(cfg)

    # Synthetic batch of 4 queries: src -> tgt
    src = torch.tensor([0, 1, 2, 3])
    rel = torch.tensor([1, 1, 1, 1])
    tgt = torch.tensor([2, 3, 4, 5])

    # Fast plastic update
    model.update_conductance(src, tgt, weight_inc=2.0)

    # Forward pass
    out = model(src_indices=src, rel_indices=rel)
    total_loss, loss_dict = loss_fn(
        logits=out["logits"],
        targets=tgt,
        model=model,
        attractor_z=out["attractor_z"],
        P_dir=out["P_dir"],
        lyap_drift=out["lyap_drift"],
    )

    # Backward pass
    total_loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_params, packed_kb = model.count_parameters()

    return {
        "loss_dict": loss_dict,
        "grad_norm": float(grad_norm.item()),
        "total_params": total_params,
        "packed_kb": packed_kb,
        "emergent_hops": out["emergent_hops"].tolist(),
    }


if __name__ == "__main__":
    print("=" * 68)
    print("Chapter 11 — The Integrated Model & Loss Suite Ledger")
    print("=" * 68)
    res = demo_ledger()
    print("Loss Components:")
    for k, v in res["loss_dict"].items():
        print(f"  {k:<18}: {v:.6f}")
    print(f"\nGradient Norm (clipped): {res['grad_norm']:.4f}")
    print(f"Parameters: {res['total_params']} ({res['packed_kb']:.2f} KB packed at 1.58 bits/param)")
    print(f"Emergent reasoning hops: {res['emergent_hops']}")
    print("=" * 68)
