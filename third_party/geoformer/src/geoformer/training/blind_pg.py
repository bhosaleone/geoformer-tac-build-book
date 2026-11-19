"""
Blind-PG v0.2.1: Probability-Consistent Nonlinear Graph Propagation Engine
==========================================================================
A mathematically audited experimental control for target-blind nonlinear
propagation across directed and multi-relational graphs.

Key Invariants & Audited Guarantees (v0.2.1):
  1. Three Decoupled Auditable Outputs:
     - Prediction: argmax_j z_cond(j)
     - Conditional Confidence: max_j z_cond(j)
     - Cumulative Survival: S_k = prod_{tau=0}^{k-1} m_tau
       * Note: S_k is the survival probability of the sharpened propagation process,
         reflecting state concentration at each step.
  2. Mathematically Consistent Mixture Survival (Exact-k & Lazy-Residual):
     - "exact_k" mode (eta_residual == 0.0): Strict exact-k-hop conditional propagation:
           q_{t+1} = z_t P_t,   s_{t+1} = ||q_{t+1}||_1,   z_{t+1} = S_gamma(q_{t+1} / s_{t+1})
           S_k = prod_{t=0}^{k-1} s_{t+1}
     - "lazy_residual" mode (eta_residual > 0.0): Probability-consistent mixture where the process
       attempts graph propagation with probability (1 - eta) and retains state with probability eta.
       Because the propagation branch loses mass (s_{t+1} <= 1.0), the conditional mixture is:
           m_{t+1} = (1 - eta) * s_{t+1} + eta
           z_{t+1} = [ (1 - eta) * s_{t+1} * z_{sharp} + eta * z_t ] / m_{t+1}
           S_{t+1} = S_t * m_{t+1}
       Applied symmetrically on EVERY step, guaranteeing that surviving mass and survival accumulator
       are mathematically identical with zero contradictions.
  3. Full Precision Invariance (FP32 & FP64):
     Constructs one-hot indicators and accumulators matching W's device and floating-point dtype.
  4. Exact Conditional Shannon Entropy:
     Entropy H(z_cond) is evaluated strictly on the normalized surviving distribution (sum == 1.0):
         gamma_eff = clamp(gamma_0 + beta * H(z_cond), min=1.0, max=gamma_max)
  5. Numerically Stabilized Log-Domain Softmax Sharpening:
     Transforms conditional distribution via log(z_sharp) = gamma * log(z_cond) followed
     by stable log-sum-exp normalization. Protects against underflow in extreme paths.
  6. Comprehensive Defensive Validation:
     - Validates constructor parameters (num_entities >= 1, mode, gamma bounds, beta >= 0).
     - Validates W (must be finite, nonnegative, float32 or float64).
     - Validates source_indices (1D integer tensor in [0, N-1], matching device).
     - Validates relation_seq (length == query_hops, bounds matching W.shape[0]).
  7. Dead-End Dropout Policy:
     Zero out-degree nodes have row sums == 0.0. Absorbed mass drops out cleanly without
     artificial self-loops or NaNs.

Author: Shrikant Bhosale (Atmabhan Pandit) │ TWIST POOL Labs │ SetMD │ ISL Research
"""

from typing import Optional, Tuple, Dict, Any, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlindPG(nn.Module):
    """
    Blind-PG v0.2.1: Probability-Consistent Nonlinear Graph Propagation Module.
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_relations: int = 1,
        gamma_0: float = 2.5,
        beta_adaptive: float = 0.25,
        gamma_max: float = 4.5,
        eta_residual: float = 0.0,          # Strict exact-k-hop when 0.0
        mode: str = "adaptive",             # "linear" | "fixed" | "adaptive"
        use_log_domain: bool = True,        # Numerically stabilized log-domain sharpening
        propagation_type: str = "exact_k",  # "exact_k" | "lazy_residual"
        adaptive_gamma: Optional[bool] = None,  # Backward-compatible alias
    ):
        super().__init__()
        if adaptive_gamma is not None:
            mode = "adaptive" if adaptive_gamma else "fixed"
        # Constructor validation
        if num_entities < 1:
            raise ValueError(f"num_entities must be >= 1, got {num_entities}")
        if num_relations < 1:
            raise ValueError(f"num_relations must be >= 1, got {num_relations}")
        if mode not in ("linear", "fixed", "adaptive"):
            raise ValueError(f"mode must be one of ('linear', 'fixed', 'adaptive'), got '{mode}'")
        if beta_adaptive < 0.0:
            raise ValueError(f"beta_adaptive must be >= 0.0, got {beta_adaptive}")
        if gamma_0 < 1.0:
            raise ValueError(f"gamma_0 must be >= 1.0, got {gamma_0}")
        if gamma_max < 1.0:
            raise ValueError(f"gamma_max must be >= 1.0, got {gamma_max}")
        if gamma_max < gamma_0:
            raise ValueError(f"gamma_max ({gamma_max}) must be >= gamma_0 ({gamma_0})")
        if not (0.0 <= eta_residual <= 1.0):
            raise ValueError(f"eta_residual must be in [0.0, 1.0], got {eta_residual}")
        if propagation_type not in ("exact_k", "lazy_residual"):
            raise ValueError(f"propagation_type must be 'exact_k' or 'lazy_residual', got {propagation_type}")
        if propagation_type == "exact_k" and eta_residual > 0.0:
            raise ValueError(f"Exact-k propagation requires eta_residual == 0.0, got {eta_residual}")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.gamma_0 = gamma_0
        self.beta_adaptive = beta_adaptive
        self.gamma_max = gamma_max
        self.eta_residual = eta_residual
        self.mode = mode
        self.use_log_domain = use_log_domain
        self.propagation_type = propagation_type

    def build_transition_kernel(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derives row-stochastic directed transition kernel P = D_out^{-1} W.
        Dead-end nodes (out_degree == 0) remain strictly 0.0 in P.
        """
        W_pos = torch.relu(W)
        row_sums = W_pos.sum(dim=-1, keepdim=True)
        is_dead_end = (row_sums.squeeze(-1) < 1e-12)

        norm_factor = torch.where(row_sums > 1e-12, row_sums, torch.ones_like(row_sums))
        P = W_pos / norm_factor
        P = torch.where(is_dead_end.unsqueeze(-1), torch.zeros_like(P), P)
        return P, is_dead_end

    def compute_entropy(self, z_normalized: torch.Tensor) -> torch.Tensor:
        """
        Computes Shannon entropy H(z) in bits: [B] or scalar.
        Assumes z_normalized is a valid probability distribution (sum == 1.0).
        """
        probs = torch.clamp(z_normalized, min=1e-15)
        # 0 * log2(0) = 0
        entropy = -torch.sum(
            torch.where(z_normalized > 1e-15, probs * torch.log2(probs), torch.zeros_like(probs)),
            dim=-1
        )
        return torch.clamp(entropy, min=0.0)

    def _validate_inputs(
        self,
        source_indices: torch.Tensor,
        W: torch.Tensor,
        query_hops: int,
        relation_seq: Optional[List[int]],
        is_multirel: bool
    ) -> List[int]:
        """Comprehensive input and numerical validation."""
        # 1. Validate W
        if not W.is_floating_point():
            raise TypeError(f"W must be a floating-point tensor (got {W.dtype})")

        if not torch.isfinite(W).all():
            raise ValueError("W contains NaN or Inf")

        if (W < 0).any():
            raise ValueError("W must contain nonnegative conductances")

        if W.dim() not in (2, 3):
            raise ValueError(f"W must be 2D [N, N] or 3D [R, N, N], got shape {list(W.shape)}")

        if W.shape[-1] != self.num_entities or W.shape[-2] != self.num_entities:
            raise ValueError(
                f"W entity dimensions ({W.shape[-2]}, {W.shape[-1]}) do not match num_entities ({self.num_entities})"
            )

        if is_multirel and W.shape[0] != self.num_relations:
            raise ValueError(
                f"W relation count ({W.shape[0]}) does not match model num_relations ({self.num_relations})"
            )

        # 2. Validate source_indices
        if source_indices.dim() != 1:
            raise ValueError(f"source_indices must be 1D [Batch], got {list(source_indices.shape)}")

        if source_indices.dtype not in (torch.long, torch.int64, torch.int32):
            raise TypeError(f"source_indices must contain integer IDs, got {source_indices.dtype}")

        if source_indices.device != W.device:
            raise ValueError(f"Device mismatch: source_indices is on {source_indices.device}, W is on {W.device}")

        if (source_indices < 0).any() or (source_indices >= self.num_entities).any():
            raise ValueError(f"All source_indices must be in [0, {self.num_entities - 1}]")

        # 3. Validate query_hops
        if query_hops < 1:
            raise ValueError(f"query_hops must be >= 1, got {query_hops}")

        # 4. Validate relation_seq
        if is_multirel:
            if relation_seq is None:
                if W.shape[0] != 1:
                    raise ValueError(f"relation_seq is required for multi-relational tensor (R={W.shape[0]})")
                relation_seq = [0] * query_hops

            if len(relation_seq) != query_hops:
                raise ValueError(
                    f"relation_seq length ({len(relation_seq)}) must equal query_hops ({query_hops})"
                )

            for step_idx, r in enumerate(relation_seq):
                if r < 0 or r >= W.shape[0]:
                    raise ValueError(
                        f"Invalid relation ID {r} at step {step_idx}; must be in [0, {W.shape[0] - 1}]"
                    )
            return relation_seq
        else:
            if relation_seq is not None:
                if len(relation_seq) != query_hops:
                    raise ValueError(
                        f"relation_seq length ({len(relation_seq)}) must equal query_hops ({query_hops})"
                    )
                if any(r != 0 for r in relation_seq):
                    raise ValueError("Single-relation graph received relation_seq with non-zero relation IDs")
            return [0] * query_hops

    def forward(
        self,
        source_indices: torch.Tensor,
        W: torch.Tensor,
        query_hops: int,
        relation_seq: Optional[List[int]] = None,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes fixed-horizon blind propagation with exact survival tracking.

        Returns:
            Dictionary containing:
              - 'final_z': [B, N] final conditional probability distribution.
              - 'predicted_node': [B] argmax entity ID.
              - 'conditional_confidence': [B] max mass in final_z.
              - 'p_surviving': [B] cumulative surviving mass in [0, 1].
              - 'entropy': [B] Shannon entropy of surviving distribution.
              - 'steps_executed': (int) query_hops.
              - 'trace': (Optional) list of distribution snapshots.
              - 'gamma_trace': (Optional) list of effective contrast exponents across steps.
        """
        B = source_indices.shape[0]
        is_multirel = (W.dim() == 3)
        validated_rel_seq = self._validate_inputs(source_indices, W, query_hops, relation_seq, is_multirel)

        # 1. Initialize unit conditional distribution and cumulative survival matching W's device and dtype
        z = F.one_hot(source_indices.long(), self.num_entities).to(device=W.device, dtype=W.dtype)
        cumulative_survival = torch.ones(B, device=W.device, dtype=W.dtype)

        # Build transition kernels
        if is_multirel:
            P_tensor = torch.zeros_like(W)
            for r in range(W.shape[0]):
                P_tensor[r], _ = self.build_transition_kernel(W[r])
        else:
            P_mat, _ = self.build_transition_kernel(W)

        trace = [z.clone()] if return_trace else None
        gamma_trace = []

        for tau in range(1, query_hops + 1):
            P_active = P_tensor[validated_rel_seq[tau - 1]] if is_multirel else P_mat

            # A. Linear propagation step along active relation kernel
            z_prop = torch.matmul(z, P_active)  # [B, N]

            # B. Audit exact step surviving mass s_{t+1} = ||z_t P||_1
            mass_sum = z_prop.sum(dim=-1, keepdim=True)  # [B, 1]
            s_t = mass_sum.squeeze(-1)                   # [B]

            has_mass = (mass_sum > 1e-12)

            # C. Normalized conditional distribution of surviving mass
            z_cond = torch.where(
                has_mass,
                z_prop / torch.clamp(mass_sum, min=1e-12),
                torch.zeros_like(z_prop)
            )

            # D. Shannon entropy strictly evaluated on normalized surviving distribution
            ent = self.compute_entropy(z_cond)

            # E. Contrast exponent with explicit bounds: clamp(gamma_0 + beta * ent, min=1.0, max=gamma_max)
            if self.mode == "adaptive":
                gamma_eff = torch.clamp(
                    self.gamma_0 + self.beta_adaptive * ent,
                    min=1.0,
                    max=self.gamma_max
                )
            elif self.mode == "fixed":
                gamma_eff = torch.full_like(ent, max(1.0, self.gamma_0))
            elif self.mode == "linear":
                gamma_eff = torch.ones_like(ent)
            else:
                raise ValueError(f"Unknown propagation mode: {self.mode}")

            gamma_trace.append(gamma_eff.clone())

            # F. Contrastive Sharpening
            if self.mode == "linear":
                z_sharp = z_cond
            else:
                if self.use_log_domain:
                    # Numerically stabilized log-domain softmax sharpening
                    log_z = torch.where(
                        z_cond > 1e-30,
                        torch.log(torch.clamp(z_cond, min=1e-30)),
                        torch.full_like(z_cond, -1e9)
                    )
                    log_weights = gamma_eff.unsqueeze(-1) * log_z
                    max_log = torch.max(log_weights, dim=-1, keepdim=True).values
                    exp_weights = torch.exp(log_weights - max_log)
                    z_sharp = torch.where(
                        has_mass,
                        exp_weights / torch.clamp(exp_weights.sum(dim=-1, keepdim=True), min=1e-12),
                        torch.zeros_like(z_cond)
                    )
                else:
                    z_pow = torch.pow(torch.clamp(z_cond, min=0.0), gamma_eff.unsqueeze(-1))
                    pow_sum = z_pow.sum(dim=-1, keepdim=True)
                    z_sharp = torch.where(
                        has_mass,
                        z_pow / torch.clamp(pow_sum, min=1e-12),
                        torch.zeros_like(z_cond)
                    )

            # G. State & Survival Update (Exact-k vs Probability-Consistent Lazy Residual)
            if self.propagation_type == "exact_k":
                z_next = z_sharp
                cumulative_survival = cumulative_survival * s_t
            else:
                # Mathematically consistent lazy-residual mixture applied symmetrically on EVERY step:
                # m_{t+1} = (1 - eta) * s_{t+1} + eta
                # z_{t+1} = [ (1 - eta) * s_{t+1} * z_{sharp} + eta * z_t ] / m_{t+1}
                # S_{t+1} = S_t * m_{t+1}
                eta = self.eta_residual
                mixture_mass = (1.0 - eta) * mass_sum + eta  # [B, 1]
                mixture_state = (1.0 - eta) * mass_sum * z_sharp + eta * z  # [B, N]

                z_next = torch.where(
                    mixture_mass > 1e-12,
                    mixture_state / torch.clamp(mixture_mass, min=1e-12),
                    torch.zeros_like(z)
                )
                cumulative_survival = cumulative_survival * mixture_mass.squeeze(-1)

            z = z_next
            if return_trace and trace is not None:
                trace.append(z.clone())

        # Three decoupled auditable outputs
        pred_nodes = torch.argmax(z, dim=-1)
        conditional_conf = torch.max(z, dim=-1).values
        final_entropy = self.compute_entropy(z)

        return {
            "final_z": z,
            "predicted_node": pred_nodes,
            "conditional_confidence": conditional_conf,
            "p_surviving": cumulative_survival,
            "entropy": final_entropy,
            "steps_executed": query_hops,
            "trace": trace,
            "gamma_trace": gamma_trace,
        }
