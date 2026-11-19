"""
Trainer Engine for GeoFormer and TAC Architecture Models.
Implements Two-Timescale Learning:
  - Fast-timescale: Hebbian plastic conductance updates on metric lattice W
  - Slow-timescale: Synaptic gradient backpropagation via Straight-Through Estimation
Supports Dual Path Kramers Thermodynamic Optimizer & Comprehensive Zero-Shot Multi-Hop Evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model import GeoFormerTACModel
from .losses import GeoFormerTACLoss
from ..optim.kramers import DualPathKramersOptimizer


class GeoFormerTACTrainer:
    """
    End-to-End Trainer for GeoFormer and TAC Neural Network Architectures.
    """
    def __init__(
        self,
        model: GeoFormerTACModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loaders: Optional[Dict[int, DataLoader]] = None,
        ground_truth_W: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loaders = val_loaders or {}
        self.ground_truth_W = ground_truth_W

        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.model.to(self.device)

        self.criterion = GeoFormerTACLoss(config=config)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "task_loss": [],
            "dirichlet_loss": [],
            "entropy_loss": [],
            "quant_loss": [],
            "multihop_eval": [],
        }
        self.best_metric = -float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer_type == "kramers":
            return DualPathKramersOptimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                beta1=self.config.kramers_beta1,
                alpha_tau=self.config.kramers_alpha_tau,
                d_0=self.config.kramers_d0,
                gamma_r=self.config.kramers_gamma_r,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )

    def _build_scheduler(self):
        total_steps = self.config.epochs * len(self.train_loader)
        warmup_steps = self.config.lr_warmup_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            if self.config.lr_decay_type == "cosine":
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {
            "total_loss": 0.0,
            "task_loss": 0.0,
            "dirichlet_loss": 0.0,
            "entropy_loss": 0.0,
            "quant_loss": 0.0,
        }
        num_batches = len(self.train_loader)

        # Modulated thermal calendar signal for Kramers optimizer
        calendar_signal = math.cos(2.0 * math.pi * epoch / max(1, self.config.epochs))

        for batch_idx, batch in enumerate(self.train_loader):
            src = batch["src"].to(self.device)
            rel = batch["rel"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through GeoFormer + TAC architecture
            outputs = self.model(
                src_indices=src,
                rel_indices=rel,
                goal_indices=tgt,
            )

            # Compute thermodynamic loss
            loss, loss_dict = self.criterion(
                logits=outputs["logits"],
                targets=tgt,
                model=self.model,
                attractor_z=outputs["attractor_z"],
                P_dir=outputs["P_dir"],
            )

            # Backward pass via BitNet b1.58 STE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            if isinstance(self.optimizer, DualPathKramersOptimizer):
                self.optimizer.step(calendar_signal=calendar_signal)
            else:
                self.optimizer.step()

            self.scheduler.step()

            # Fast-Timescale Hebbian plastic update on conductance lattice W
            self.model.update_conductance(src, tgt, weight_inc=self.config.plastic_rate)

            # Accumulate metrics
            for k in epoch_metrics:
                epoch_metrics[k] += loss_dict.get(k, 0.0)

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, num_batches)

        return epoch_metrics

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader: DataLoader, depth: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluates model zero-shot on unseen queries at specified depth.
        """
        self.model.eval()
        total_samples = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top10 = 0
        reciprocal_ranks = []
        target_masses = []
        entropies = []
        emergent_steps = []

        for batch in dataloader:
            src = batch["src"].to(self.device)
            rel = batch["rel"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            B = src.shape[0]

            outputs = self.model(
                src_indices=src,
                rel_indices=rel,
                goal_indices=tgt,
                fixed_hops=None,  # Autonomous emergent reasoning (no hop hints)
            )

            logits = outputs["logits"]
            attractor_z = outputs["attractor_z"]
            inferred_hops = outputs["emergent_hops"]

            probs = F.softmax(logits, dim=-1)

            # Top-k accuracy
            _, pred_top10 = torch.topk(logits, k=min(10, self.config.num_classes), dim=-1)
            pred_top1 = pred_top10[:, 0]

            correct_top1 += (pred_top1 == tgt).sum().item()
            for i in range(B):
                target_i = tgt[i].item()
                top10_i = pred_top10[i].tolist()
                if target_i in top10_i[:3]:
                    correct_top3 += 1
                if target_i in top10_i:
                    correct_top10 += 1
                    rank = top10_i.index(target_i) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            # Soliton target probability mass
            t_mass = torch.gather(attractor_z, 1, tgt.unsqueeze(-1)).squeeze(-1)
            target_masses.extend(t_mass.cpu().tolist())

            # Attractor Shannon entropy
            z_prob = torch.clamp(attractor_z, min=1e-12)
            ent = -torch.sum(z_prob * torch.log2(z_prob), dim=-1)
            entropies.extend(ent.cpu().tolist())

            # Inferred emergent hops
            emergent_steps.extend(inferred_hops.cpu().tolist())
            total_samples += B

        n = max(1, total_samples)
        return {
            "depth": depth if depth is not None else 1,
            "samples": total_samples,
            "hits@1": (correct_top1 / n) * 100.0,
            "hits@3": (correct_top3 / n) * 100.0,
            "hits@10": (correct_top10 / n) * 100.0,
            "mrr": float(sum(reciprocal_ranks) / n),
            "p_target_mass": float(sum(target_masses) / n),
            "entropy": float(sum(entropies) / n),
            "emergent_hops": float(sum(emergent_steps) / n),
        }

    @torch.no_grad()
    def evaluate_multihop_depths(self) -> Dict[int, Dict[str, float]]:
        """
        Runs autonomous zero-shot evaluation across all configured reasoning depths.
        """
        results = {}
        for depth, loader in sorted(self.val_loaders.items()):
            res = self.evaluate_dataloader(loader, depth=depth)
            results[depth] = res
        return results

    def save_checkpoint(self, path: str, is_best: bool = False, extra_metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "W_conductance": self.model.W_conductance.cpu(),
            "config": self.config.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_metric": self.best_metric,
            "timestamp": time.time(),
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(os.path.abspath(path)), "best_model.pt")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.W_conductance.copy_(checkpoint["W_conductance"].to(self.device))
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)

    def train(self) -> Dict[str, Any]:
        """
        Executes full training protocol across all epochs.
        """
        params, packed_kb = self.model.count_parameters()
        print("=" * 88)
        print(f"🚀 INITIALIZING TRAINING PIPELINE: {self.config.experiment_name}")
        print(f"   Architecture: GeoFormer + TAC (BitNet b1.58 STE | Mode: {self.config.mode.upper()})")
        print(f"   Trainable Parameters: {params:,} | Packed Footprint: {packed_kb:.2f} KB")
        print(f"   Optimizer: {self.config.optimizer_type.upper()} | LR: {self.config.learning_rate}")
        print(f"   Entities: {self.config.num_entities} | Layers: {self.config.n_layers} | d_model: {self.config.d_model}")
        print(f"   Max Reasoning Horizon: {self.config.max_reasoning_hops} Hops | Epochs: {self.config.epochs}")
        print("=" * 88)

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["total_loss"])
            self.history["task_loss"].append(train_metrics["task_loss"])
            self.history["dirichlet_loss"].append(train_metrics["dirichlet_loss"])
            self.history["entropy_loss"].append(train_metrics["entropy_loss"])
            self.history["quant_loss"].append(train_metrics["quant_loss"])

            # Periodic multi-hop evaluation
            multihop_report = {}
            if epoch % self.config.eval_interval == 0 and len(self.val_loaders) > 0:
                multihop_report = self.evaluate_multihop_depths()
                self.history["multihop_eval"].append({
                    "epoch": epoch,
                    "depths": multihop_report,
                })

                # Score based on deep reasoning performance (average across depths)
                score = sum(m["hits@1"] for m in multihop_report.values()) / max(1, len(multihop_report))
                is_best = score > self.best_metric
                if is_best:
                    self.best_metric = score

                # Log epoch summary
                print(f"[Epoch {epoch:02d}/{self.config.epochs:02d}] "
                      f"Loss: {train_metrics['total_loss']:.4f} (Task: {train_metrics['task_loss']:.4f}, "
                      f"Dir: {train_metrics['dirichlet_loss']:.4f}) | "
                      f"Val Avg Hits@1: {score:.1f}% | Time: {epoch_time:.2f}s"
                      f"{' ★ BEST' if is_best else ''}")

                # Save checkpoint
                ckpt_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
                self.save_checkpoint(ckpt_path, is_best=is_best)
            else:
                print(f"[Epoch {epoch:02d}/{self.config.epochs:02d}] "
                      f"Loss: {train_metrics['total_loss']:.4f} | Time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print("=" * 88)
        print(f"✨ TRAINING COMPLETE! Elapsed Time: {total_time:.2f}s | Best Avg Score: {self.best_metric:.2f}%")
        print("=" * 88)

        # Save final checkpoint and history
        final_ckpt_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_ckpt_path, is_best=False)

        history_path = os.path.join(self.config.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
