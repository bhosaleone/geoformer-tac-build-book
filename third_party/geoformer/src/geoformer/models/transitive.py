"""
Deep Transitive Reasoning Model for Higher-Order Path Scaling.
Supports 2-hop, 3-hop, and 4-hop zero-shot path discovery via matrix power routing
and optional Hippocampal Relational Working Memory (RML).
"""

import torch
import torch.nn as nn
from ..quant.ternary_linear import QuantizedLinear
from ..layers.memory import HippocampalRelationalMemory


class DeepTransitiveModel(nn.Module):
    """
    Tier 5/6 Architecture configured for deep composition over 128+ entities.
    Employs arbitrary-order conductance powers W^k (up to 16+ hops),
    emergent Q-TAC attractor relaxation, and Hippocampal Relational Working Memory.
    """
    def __init__(
        self,
        num_entities=128,
        num_relations=4,
        d_model=512,
        n_neurons=256,
        mode="ternary",
        adaptive=True,
        use_memory=False,
        max_hops=16,
        emergent=False,
        gamma_sharpen=2.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        self.adaptive = adaptive
        self.use_memory = use_memory
        self.max_hops = max_hops
        self.emergent = emergent
        self.gamma_sharpen = gamma_sharpen

        self.entity_emb = nn.Embedding(num_entities, d_model)
        self.relation_emb = nn.Embedding(num_relations, d_model)

        self.fc1 = QuantizedLinear(d_model * 2, d_model, mode=mode)
        self.ln1 = nn.LayerNorm(d_model)
        self.fc2 = QuantizedLinear(d_model, d_model, mode=mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.classifier = QuantizedLinear(d_model, num_entities, mode=mode)

        if adaptive:
            self.W = nn.Parameter(torch.full((num_entities, num_entities), 0.02))
        else:
            self.register_buffer("W", torch.full((num_entities, num_entities), 0.02))

        if use_memory:
            self.memory = HippocampalRelationalMemory(
                d_model=d_model,
                num_entities=num_entities,
                num_slots=max(16, max_hops + 1),
                mode=mode
            )

    def forward(self, entity_idx, relation_idx, hops=None, emergent=None):
        e_emb = self.entity_emb(entity_idx)
        r_emb = self.relation_emb(relation_idx)
        x_in = torch.cat([e_emb, r_emb], dim=-1)

        h1 = torch.relu(self.ln1(self.fc1(x_in)))
        h2 = torch.relu(self.ln2(self.fc2(h1)))

        base_logits = self.classifier(h2)

        if not self.adaptive:
            return base_logits

        is_emergent = self.emergent if emergent is None else emergent
        W_pos = torch.relu(self.W)
        row_sums = torch.clamp(W_pos.sum(dim=-1, keepdim=True), min=1e-6)
        W_norm = W_pos / row_sums

        if is_emergent:
            # Autonomous Q-TAC Attractor Relaxation without hardcoded hop counts
            B = entity_idx.shape[0]
            z = F.one_hot(entity_idx, self.num_entities).float()
            is_converged = torch.zeros(B, dtype=torch.bool, device=entity_idx.device)

            for step in range(self.max_hops):
                z_prev = z.clone()
                z_next = torch.matmul(z_prev, W_norm)
                # Attractor sharpening to prevent deep-hop attenuation
                if self.gamma_sharpen > 1.0:
                    z_pow = torch.pow(torch.clamp(z_next, min=0.0), self.gamma_sharpen)
                    z_next = z_pow / torch.clamp(z_pow.sum(dim=-1, keepdim=True), min=1e-9)

                drift = torch.norm(z_next - z_prev, dim=-1)
                peak_conf, _ = torch.max(z_next, dim=-1)
                captured = (peak_conf >= 0.70) | (drift < 1e-4)
                is_converged = is_converged | captured
                z = torch.where(is_converged.unsqueeze(-1), z_prev, z_next)
                if is_converged.all():
                    break

            metric_boost = z
        else:
            # Scalable Matrix Powers supporting arbitrary hop depths (up to max_hops)
            target_depth = self.max_hops
            if hops is not None:
                target_depth = max(target_depth, int(hops) if not isinstance(hops, torch.Tensor) else int(hops.max().item()) + 1)
            elif relation_idx is not None:
                target_depth = max(target_depth, int(relation_idx.max().item()) + 1)

            powers = [W_pos]
            cur_pow = W_norm
            powers.append(cur_pow)
            for _ in range(2, target_depth + 1):
                cur_pow = torch.matmul(cur_pow, W_norm)
                powers.append(cur_pow)

            boost_list = []
            for i in range(len(entity_idx)):
                if hops is not None:
                    h_val = hops[i].item() if isinstance(hops, torch.Tensor) else hops
                    rel_idx = min(h_val, len(powers) - 1)
                else:
                    rel = relation_idx[i].item()
                    rel_idx = min(rel, len(powers) - 1)
                boost_list.append(powers[rel_idx][entity_idx[i]])

            metric_boost = torch.stack(boost_list, dim=0)

        if self.use_memory:
            effective_hops = (relation_idx + 1) if hops is None else hops
            mem_logits, _ = self.memory(
                h_query=h2,
                entity_embs=self.entity_emb.weight,
                W_conductance=W_pos,
                hops=effective_hops,
                emergent=is_emergent
            )
            logits = base_logits + 2.0 * metric_boost + 3.0 * mem_logits
        else:
            logits = base_logits + 3.0 * metric_boost

        return logits
