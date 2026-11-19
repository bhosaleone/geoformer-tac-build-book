"""
GeoFormer + TAC Unified Trainable Model Architecture.
Integrates BitNet b1.58 Ternary Weights, Multi-Head Geodesic Highway Attention (Q-GHA),
Continuous Q-TAC Latent Relaxation, Fast Hebbian Metric Plasticity, and
Goal-Conditioned Emergent Multi-Hop Attractor Dynamics.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant.ternary_linear import QuantizedLinear
from ..models.block import RMSNorm
from ..models.geoformer import GeoFormerBlock
from ..layers.emergent_multihop import GoalConditionedEmergentHopAttractor
from .config import TrainingConfig


class GeoFormerTACModel(nn.Module):
    """
    Unified GeoFormer + TAC Trainable Foundation Architecture.
    """
    def __init__(self, config: Optional[TrainingConfig] = None):
        super().__init__()
        self.config = config or TrainingConfig()

        # 1. Embeddings & Input Projections
        self.entity_emb = nn.Embedding(self.config.num_entities, self.config.d_model)
        self.relation_emb = nn.Embedding(self.config.num_relations, self.config.d_model)

        # Feature projection for vector inputs (node classification / continuous features)
        self.in_proj = QuantizedLinear(
            self.config.d_model, self.config.d_model, mode=self.config.mode
        )

        # 2. Stacked GeoFormer Blocks with Continuous Q-TAC Relaxation
        self.blocks = nn.ModuleList([
            GeoFormerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                mode=self.config.mode,
                tau_max=self.config.tau_max,
                lambda_geo=self.config.lambda_geo,
                lambda_dir=self.config.lambda_dir_layer,
                dropout=self.config.dropout,
                schedule=self.config.schedule,
            )
            for _ in range(self.config.n_layers)
        ])

        # 3. Normalization & Output Classifier
        self.final_norm = RMSNorm(self.config.d_model)
        self.classifier = QuantizedLinear(
            self.config.d_model, self.config.num_classes, mode=self.config.mode
        )

        # 4. Fast-Timescale Plastic Metric Conductance Lattice W [N, N]
        self.register_buffer(
            "W_conductance",
            torch.zeros(self.config.num_entities, self.config.num_entities)
        )

        # 5. Goal-Conditioned Emergent Multi-Hop Attractor (TAC Layer)
        self.attractor = GoalConditionedEmergentHopAttractor(
            num_entities=self.config.num_entities,
            gamma_0=self.config.gamma_0,
            beta_adaptive=self.config.beta_adaptive,
            gamma_max=self.config.gamma_max,
            kappa=self.config.kappa_attractor,
            eps_tac=self.config.eps_tac,
            eps_q=self.config.eps_q,
            lambda_dir=self.config.lambda_dir_attractor,
            lambda_v=self.config.lambda_v_attractor,
            lambda_q=self.config.lambda_q_attractor,
            eta_residual=self.config.eta_residual,
            max_steps=self.config.max_reasoning_hops,
            confidence_thresh=self.config.confidence_thresh,
        )

    def update_conductance(
        self,
        src_indices: torch.Tensor,
        tgt_indices: torch.Tensor,
        weight_inc: float = 1.0
    ):
        """
        Advances fast-timescale plastic conductance lattice along observed edges.
        Sculpts Geodesic Highways: d_ij = 1 / (W_ij + eps).
        """
        with torch.no_grad():
            delta = self.config.delta_w * self.config.plastic_rate * weight_inc
            src_cpu = src_indices.detach().cpu()
            tgt_cpu = tgt_indices.detach().cpu()
            for s, t in zip(src_cpu, tgt_cpu):
                s_idx, t_idx = s.item(), t.item()
                cur = self.W_conductance[s_idx, t_idx].item()
                self.W_conductance[s_idx, t_idx] = min(1.0, cur + delta)

    def forward(
        self,
        src_indices: torch.Tensor,
        rel_indices: Optional[torch.Tensor] = None,
        goal_indices: Optional[torch.Tensor] = None,
        fixed_hops: Optional[int] = None,
        d_matrix: Optional[torch.Tensor] = None,
        L_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass executing neural representation + emergent Q-TAC relaxation.
        """
        device = src_indices.device
        B = src_indices.shape[0]

        # 1. Embeddings: Entity + Relation
        e_emb = self.entity_emb(src_indices)  # [B, D]
        if rel_indices is not None:
            r_emb = self.relation_emb(rel_indices)  # [B, D]
            h = (e_emb + r_emb).unsqueeze(1)       # [B, 1, D]
        else:
            h = e_emb.unsqueeze(1)                 # [B, 1, D]

        # 2. Pass through stacked GeoFormer Blocks
        for block in self.blocks:
            h = block(h, d_matrix=d_matrix, L_matrix=L_matrix)

        h_feat = self.final_norm(h).squeeze(1)  # [B, D]
        neural_logits = self.classifier(h_feat) # [B, num_classes]

        # 3. Emergent Q-TAC Attractor Relaxation along plastic metric highways
        query_emb = h_feat  # [B, D]
        node_embs = self.entity_emb.weight  # [N, D]

        attractor_z, emergent_hops, final_d_sem, _ = self.attractor(
            source_indices=src_indices,
            W_conductance=self.W_conductance,
            query_emb=query_emb,
            node_embs=node_embs,
            goal_indices=goal_indices,
            fixed_hops=fixed_hops,
        )

        # 4. Row-stochastic transition kernel P_dir for Dirichlet regularizer
        P_dir = self.attractor.normalize_transition_kernel(self.W_conductance)

        # 5. Geodesic Metric Highway Boost
        metric_boost = attractor_z * self.config.metric_boost_scale

        # 6. Fused decision
        # Direct 1-hop facts (rel == 0) use neural logits; multi-hop deduction uses metric boost
        if rel_indices is not None:
            is_direct = (rel_indices == 0).unsqueeze(-1)
            final_logits = torch.where(is_direct, neural_logits, neural_logits + metric_boost)
        else:
            final_logits = neural_logits + metric_boost

        return {
            "logits": final_logits,
            "neural_logits": neural_logits,
            "attractor_z": attractor_z,
            "emergent_hops": emergent_hops,
            "final_d_sem": final_d_sem,
            "P_dir": P_dir,
            "h_feat": h_feat,
        }

    def count_parameters(self) -> Tuple[int, float]:
        """
        Returns total parameters and BitNet b1.58 packed memory footprint (KB).
        Packed at 1.58 bits/param = 0.20 bytes/param.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        packed_kb = (total_params * 0.20) / 1024.0
        return total_params, packed_kb
