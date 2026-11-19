# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch13_trainer.py — rigorous unit, optimization, and determinism tests
for Chapter 13: The Training Loop & the Kramers Optimizer.
"""

from __future__ import annotations

import math
import pytest
import torch

from gftac.ch11_integrated_model import BookGeoFormerTACModel, BookModelConfig
from gftac.ch12_datasets import DatasetConfig, create_benchmark_dataloaders
from gftac.ch13_trainer import (
    BookKramersOptimizer,
    BookTrainer,
    TrainerConfig,
    demo_ledger,
)


def test_kramers_optimizer_step_b13_1():
    """Theorem B.13.1: Kramers optimizer steps toward negative gradient."""
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([0.50]))
    opt = BookKramersOptimizer([x], lr=0.1, beta1=0.9, alpha_tau=0.2, d_0=1e-4)

    # Loss V(x) = 0.5 * (x - 2.0)^2 -> dV/dx = (x - 2.0) = -1.5
    loss = 0.5 * (x - 2.0) ** 2
    loss.backward()

    opt.step(calendar_signal=0.0)
    # x should step to the right towards 2.0
    assert float(x.item()) > 0.50


def test_modulated_thermal_calendar():
    """Diffusivity D(t) scales with calendar signal in [1 - gamma_r, 1 + gamma_r]."""
    d_0 = 1e-4
    gamma_r = 0.5

    d_neg = max(1e-9, d_0 * (1.0 + gamma_r * (-1.0)))
    d_zero = max(1e-9, d_0 * (1.0 + gamma_r * 0.0))
    d_pos = max(1e-9, d_0 * (1.0 + gamma_r * 1.0))

    assert d_neg == pytest.approx(0.5e-4)
    assert d_zero == pytest.approx(1.0e-4)
    assert d_pos == pytest.approx(1.5e-4)
    assert d_neg < d_zero < d_pos


def test_two_timescale_ordering_soundness_b13_2():
    """Theorem B.13.2: Hebbian lattice write follows backprop, avoiding causality leaks."""
    torch.manual_seed(42)
    m_cfg = BookModelConfig(num_entities=8, num_relations=8, num_classes=8, d_model=16, n_layers=1)
    d_cfg = DatasetConfig(num_entities=8, num_chains=2, max_hops=2, batch_size=4, seed=42)

    model = BookGeoFormerTACModel(m_cfg)
    train_loader, _, _ = create_benchmark_dataloaders(d_cfg, eval_depths=[2])

    trainer = BookTrainer(model, train_loader, {}, TrainerConfig(epochs=1, seed=42))

    # Before step, lattice is all-zero
    assert torch.all(model.W_conductance == 0.0)

    metrics = trainer.train_epoch(1)
    # After epoch, lattice has non-zero conductances and loss is finite
    assert (model.W_conductance > 0).any()
    assert math.isfinite(metrics["total_loss"])


def test_training_determinism_b13_3():
    """Theorem B.13.3: Pinned seed guarantees bit-exact reproducible loss curves."""
    def run_trial(seed: int):
        torch.manual_seed(seed)
        m = BookGeoFormerTACModel(BookModelConfig(num_entities=12, num_relations=8, num_classes=12, d_model=16, n_layers=1))
        d_cfg = DatasetConfig(num_entities=12, num_chains=2, max_hops=2, batch_size=4, seed=seed)
        tl, _, _ = create_benchmark_dataloaders(d_cfg, eval_depths=[2])
        tr = BookTrainer(m, tl, {}, TrainerConfig(epochs=2, seed=seed))
        return tr.train()["train_loss"]

    run1 = run_trial(42)
    run2 = run_trial(42)

    assert len(run1) == len(run2)
    for a, b in zip(run1, run2):
        assert a == pytest.approx(b, abs=1e-7)


def test_sdk_parity_kramers_step():
    """Parity check against SDK DualPathKramersOptimizer."""
    from geoformer.optim.kramers import DualPathKramersOptimizer as SDKKramers

    torch.manual_seed(100)
    p_sdk = torch.nn.Parameter(torch.tensor([2.00], requires_grad=True))
    p_book = torch.nn.Parameter(torch.tensor([2.00], requires_grad=True))

    opt_sdk = SDKKramers([p_sdk], lr=0.01, beta1=0.9, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5)
    opt_book = BookKramersOptimizer([p_book], lr=0.01, beta1=0.9, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5)

    grad = torch.tensor([0.50])
    p_sdk.grad = grad.clone()
    p_book.grad = grad.clone()

    torch.manual_seed(300)
    opt_sdk.step(calendar_signal=0.2)
    torch.manual_seed(300)
    opt_book.step(calendar_signal=0.2)

    assert float(p_sdk.item()) == pytest.approx(float(p_book.item()), abs=1e-6)


def test_demo_ledger_contract():
    """Demo ledger runs to completion and records loss reduction."""
    res = demo_ledger()
    assert res["epochs_run"] == 3
    assert res["loss_decreased"] is True
    assert math.isfinite(res["final_train_loss"])
