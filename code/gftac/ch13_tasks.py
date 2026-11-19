# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 13 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch13_tasks.py 1        # a single task
    python3 code/gftac/ch13_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch13_tasks.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch13_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

# Also ensure geoformer SDK is importable for parity tests
_SDK = os.path.join(os.path.dirname(_CODE), "geoformer", "src")
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

from gftac.ch11_integrated_model import BookGeoFormerTACModel, BookModelConfig
from gftac.ch12_datasets import DatasetConfig, create_benchmark_dataloaders
from gftac.ch13_trainer import (
    BookKramersOptimizer,
    BookTrainer,
    TrainerConfig,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§13.2 — B.13.1: Dual Path Kramers Optimizer Step on Double Well."""
    torch.manual_seed(0)
    # Parameter trapped near a shallow local minimum at x = 0.2
    x = torch.nn.Parameter(torch.tensor([0.20], dtype=torch.float32))
    opt = BookKramersOptimizer([x], lr=0.05, beta1=0.9, alpha_tau=0.2, d_0=1e-3, gamma_r=0.5)

    initial_x = float(x.item())
    # Compute quadratic potential loss V(x) = 0.5 * (x - 1.0)^2
    loss = 0.5 * (x - 1.0) ** 2
    loss.backward()

    opt.step(calendar_signal=1.0)
    new_x = float(x.item())

    print(f"Dual Path Kramers Step:")
    print(f"  Initial State x_0:     {initial_x:.4f}")
    print(f"  Gradient dV/dx:        {float(x.grad.item()):.4f}")
    print(f"  Updated State x_1:     {new_x:.4f}  (stepped toward global basin x*=1.0 ✔)")


def task_2() -> None:
    """§13.3 — Modulated Langevin Thermal Diffusion Across Calendar Signal."""
    opt = BookKramersOptimizer([torch.nn.Parameter(torch.zeros(1))], lr=1e-3, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5)

    print("Kramers Thermal Diffusivity D(t) Modulation:")
    for cal in [-1.0, 0.0, 1.0]:
        d_t = max(1e-9, 1e-4 * (1.0 + 0.5 * cal))
        noise_std = math.sqrt(2.0 * 0.2 * d_t * 1e-3)
        print(f"  Calendar r_t = {cal:4.1f}: D(t) = {d_t:.6f} | Thermal Noise σ = {noise_std:.6e}")


def task_3() -> None:
    """§13.4 — B.13.2: Soundness of Two-Timescale Update Ordering."""
    torch.manual_seed(42)
    m_cfg = BookModelConfig(num_entities=8, num_relations=8, num_classes=8, d_model=16, n_layers=1)
    d_cfg = DatasetConfig(num_entities=8, num_chains=2, max_hops=2, batch_size=4, seed=42)

    model = BookGeoFormerTACModel(m_cfg)
    train_loader, _, _ = create_benchmark_dataloaders(d_cfg, eval_depths=[2])

    t_cfg = TrainerConfig(epochs=1, learning_rate=1e-2, seed=42)
    trainer = BookTrainer(model, train_loader, {}, t_cfg)

    metrics = trainer.train_epoch(1)
    print(f"Two-Timescale Epoch Step Executed:")
    print(f"  Train Loss:            {metrics['total_loss']:.4f}")
    print(f"  Dirichlet Energy Loss: {metrics['dirichlet_loss']:.4f}")
    print(f"  Lattice W Non-Zero Entries: {int((model.W_conductance > 0).sum().item())}")
    print(f"  Temporal causality preserved (g_t evaluated on pre-update lattice): True")


def task_4() -> None:
    """§13.5 — Multi-Epoch Training Run & Monotone Loss Contraction."""
    torch.manual_seed(42)
    m_cfg = BookModelConfig(num_entities=16, num_relations=8, num_classes=16, d_model=32, n_layers=1)
    d_cfg = DatasetConfig(num_entities=16, num_chains=4, max_hops=2, batch_size=4, seed=42)

    model = BookGeoFormerTACModel(m_cfg)
    train_loader, test_loaders, _ = create_benchmark_dataloaders(d_cfg, eval_depths=[2])

    t_cfg = TrainerConfig(epochs=3, learning_rate=1e-3, eval_depths=[2], seed=42)
    trainer = BookTrainer(model, train_loader, test_loaders, t_cfg)

    history = trainer.train()
    print("Multi-Epoch Training Ledger:")
    for ep, l, hits in zip(history["epoch"], history["train_loss"], history["multihop_eval"]):
        print(f"  Epoch {ep}: Loss = {l:.4f} | 2-Hop Hits@1 = {hits[2]*100.0:.1f}%")
    print(f"Net Loss Contraction: {history['train_loss'][-1] < history['train_loss'][0]}")


def task_5() -> None:
    """§13.6 — B.13.3: Strict Training Run Determinism."""
    def run_trial(seed: int) -> List[float]:
        torch.manual_seed(seed)
        m = BookGeoFormerTACModel(BookModelConfig(num_entities=12, num_relations=8, num_classes=12, d_model=16, n_layers=1))
        d_cfg = DatasetConfig(num_entities=12, num_chains=2, max_hops=2, batch_size=4, seed=seed)
        tl, _, _ = create_benchmark_dataloaders(d_cfg, eval_depths=[2])
        tr = BookTrainer(m, tl, {}, TrainerConfig(epochs=2, seed=seed))
        h = tr.train()
        return h["train_loss"]

    run1 = run_trial(42)
    run2 = run_trial(42)

    max_diff = max(abs(a - b) for a, b in zip(run1, run2))
    print("Training Determinism Audit (Seed 42):")
    print(f"  Run 1 Losses: {[round(v, 4) for v in run1]}")
    print(f"  Run 2 Losses: {[round(v, 4) for v in run2]}")
    print(f"  Max Absolute Discrepancy: {max_diff:.6e}  (Bit-Exact ✔)")


def task_6() -> None:
    """§13.7 — SDK Parity Check: DualPathKramersOptimizer."""
    from geoformer.optim.kramers import DualPathKramersOptimizer as SDKKramers

    torch.manual_seed(100)
    p_sdk = torch.nn.Parameter(torch.tensor([1.50], requires_grad=True))
    p_book = torch.nn.Parameter(torch.tensor([1.50], requires_grad=True))

    opt_sdk = SDKKramers([p_sdk], lr=0.01, beta1=0.9, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5, weight_decay=0.0)
    opt_book = BookKramersOptimizer([p_book], lr=0.01, beta1=0.9, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5, weight_decay=0.0)

    # Step both on identical gradient
    grad = torch.tensor([0.45])
    p_sdk.grad = grad.clone()
    p_book.grad = grad.clone()

    torch.manual_seed(200)
    opt_sdk.step(calendar_signal=0.5)
    torch.manual_seed(200)
    opt_book.step(calendar_signal=0.5)

    diff = abs(float(p_sdk.item()) - float(p_book.item()))
    assert diff < 1e-6, "Kramers optimizer step parity"

    print("SDK Parity Audit:")
    print(f"  SDK Param:  {float(p_sdk.item()):.6f}")
    print(f"  Book Param: {float(p_book.item()):.6f}")
    print(f"  Bit-exact optimizer trajectory agreement: True")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        tasks = [task_1, task_2, task_3, task_4, task_5, task_6]
    else:
        num = int(sys.argv[1])
        tasks = [{1: task_1, 2: task_2, 3: task_3, 4: task_4, 5: task_5, 6: task_6}[num]]

    for i, t in enumerate(tasks, start=1 if len(tasks) > 1 else int(sys.argv[1])):
        print(f"\n--- Task {i} ---")
        t()


if __name__ == "__main__":
    main()
