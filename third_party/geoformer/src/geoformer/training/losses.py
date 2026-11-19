"""
Loss Functions for GeoFormer and TAC Architecture Models.
Combines Task Cross-Entropy, Metric Dirichlet Energy, Lyapunov Drift Stationarity,
Soliton Shannon Entropy, and BitNet b1.58 Ternary Quantization Regularization.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig


class GeoFormerTACLoss(nn.Module):
    """
    Unified Thermodynamic Loss Formulation for GeoFormer + TAC.
    """
    def __init__(self, config: Optional[TrainingConfig] = None):
        super().__init__()
        self.config = config or TrainingConfig()
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_dirichlet_energy(self, z: torch.Tensor, P_dir: torch.Tensor) -> torch.Tensor:
        """
        Computes the directed Dirichlet energy of the probability mass z:
            E_Dir(z) = 1/2 * sum_b z_b (I - P_dir) z_b^T
        Measures smoothness of mass flow across the directed metric lattice.
        """
        # z: [B, N], P_dir: [N, N]
        z_diff = z - torch.matmul(z, P_dir)  # [B, N] = z * L_dir
        dirichlet_energy = 0.5 * torch.sum(z * z_diff, dim=-1).mean()
        return torch.clamp(dirichlet_energy, min=0.0)

    def compute_shannon_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Shannon entropy H(z) in bits:
            H(z) = - sum_j z_j * log2(z_j + eps)
        """
        probs = torch.clamp(z, min=1e-12)
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1).mean()
        return entropy

    def compute_ternary_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        Computes soft ternary penalty driving continuous latent weights towards {-1, 0, +1}:
            R(w) = mean( (abs(w) - 1.0)^2 * abs(w) )
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0
        for name, p in model.named_parameters():
            if "weight" in name and p.dim() >= 2:
                w_abs = torch.abs(p)
                # Zero at w=0 and w=+/-1: (w^2 - 1)^2 * w^2 or (|w| - 1)^2 * |w|
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
        lyapunov_drift: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates total loss and diagnostic dictionary.
        """
        # 1. Primary Task Cross-Entropy Loss
        task_loss = self.ce_loss(logits, targets)

        # 2. Metric Dirichlet Energy Regularization
        if attractor_z is not None and P_dir is not None:
            dir_loss = self.compute_dirichlet_energy(attractor_z, P_dir)
        else:
            dir_loss = torch.tensor(0.0, device=logits.device)

        # 3. Lyapunov Stationarity Penalty
        if lyapunov_drift is not None:
            lyap_loss = torch.mean(lyapunov_drift ** 2)
        else:
            lyap_loss = torch.tensor(0.0, device=logits.device)

        # 4. Soliton Shannon Entropy Penalty (contrastive peak preservation)
        if attractor_z is not None:
            ent_loss = self.compute_shannon_entropy(attractor_z)
        else:
            ent_loss = torch.tensor(0.0, device=logits.device)

        # 5. Ternary Synaptic Weight Regularization (BitNet b1.58 STE stability)
        if self.config.mode == "ternary":
            quant_loss = self.compute_ternary_regularization(model)
        else:
            quant_loss = torch.tensor(0.0, device=logits.device)

        # Total Loss formulation
        total_loss = (
            self.config.lambda_task * task_loss
            + self.config.lambda_dir * dir_loss
            + self.config.lambda_lyap * lyap_loss
            + self.config.lambda_ent * ent_loss
            + self.config.lambda_quant * quant_loss
        )

        metrics = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "dirichlet_loss": dir_loss.item(),
            "lyapunov_loss": lyap_loss.item(),
            "entropy_loss": ent_loss.item(),
            "quant_loss": quant_loss.item(),
        }

        return total_loss, metrics
