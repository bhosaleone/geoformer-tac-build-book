# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch11_integrated_model.py — rigorous unit, gradient, and parity tests
for Chapter 11: The Integrated Model & Loss Suite.
"""

from __future__ import annotations

import io
import math
import pytest
import torch

from gftac.ch11_integrated_model import (
    BookAttractor,
    BookGeoFormerTACLoss,
    BookGeoFormerTACModel,
    BookModelConfig,
    demo_ledger,
)


def test_forward_pass_shapes():
    """Validates tensor shapes across all heads of BookGeoFormerTACModel."""
    B, N, D = 4, 16, 32
    cfg = BookModelConfig(num_entities=N, num_relations=4, num_classes=N, d_model=D, n_layers=2)
    model = BookGeoFormerTACModel(cfg)

    src = torch.tensor([0, 1, 2, 3])
    rel = torch.tensor([0, 1, 2, 3])
    out = model(src_indices=src, rel_indices=rel)

    assert out["logits"].shape == (B, N)
    assert out["neural_logits"].shape == (B, N)
    assert out["attractor_z"].shape == (B, N)
    assert out["emergent_hops"].shape == (B,)
    assert out["P_dir"].shape == (N, N)
    assert out["h_feat"].shape == (B, D)

    # Attractor distribution must be a valid probability distribution (row sum = 1.0)
    z_sums = out["attractor_z"].sum(dim=-1)
    assert torch.allclose(z_sums, torch.ones(B), atol=1e-4)


def test_thermodynamic_loss_components_finite():
    """Theorem B.11.1: Loss components are non-negative, finite, and well-behaved."""
    cfg = BookModelConfig(num_entities=8, num_relations=2, num_classes=8, d_model=32, n_layers=1)
    model = BookGeoFormerTACModel(cfg)
    loss_fn = BookGeoFormerTACLoss(cfg)

    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([1, 2, 3])
    model.update_conductance(src, tgt, weight_inc=1.5)

    out = model(src_indices=src)
    total_loss, loss_dict = loss_fn(
        logits=out["logits"],
        targets=tgt,
        model=model,
        attractor_z=out["attractor_z"],
        P_dir=out["P_dir"],
        lyap_drift=out["lyap_drift"],
    )

    assert math.isfinite(loss_dict["total_loss"])
    assert loss_dict["task_loss"] >= 0.0
    assert loss_dict["dirichlet_loss"] >= 0.0
    assert loss_dict["entropy_loss"] >= 0.0
    assert loss_dict["lyap_loss"] >= 0.0
    assert loss_dict["quant_loss"] >= 0.0


def test_backward_gradient_highway_b11_3():
    """Theorem B.11.3: Unbroken backward gradient highways through Pre-RMSNorm backbone."""
    torch.manual_seed(42)
    cfg = BookModelConfig(num_entities=8, num_relations=2, num_classes=8, d_model=32, n_layers=2)
    model = BookGeoFormerTACModel(cfg)
    loss_fn = BookGeoFormerTACLoss(cfg)

    src = torch.tensor([0, 1])
    tgt = torch.tensor([2, 3])

    out = model(src_indices=src)
    loss, _ = loss_fn(logits=out["logits"], targets=tgt, model=model)
    loss.backward()

    # Assert non-zero, non-NaN gradients across entire depth
    assert model.entity_emb.weight.grad is not None
    assert float(model.entity_emb.weight.grad.norm()) > 1e-4
    assert model.blocks[0].attn.v_proj.weight.grad is not None
    assert float(model.blocks[0].attn.v_proj.weight.grad.norm()) > 1e-4
    assert model.blocks[0].ffn.w_gate.weight.grad is not None
    assert float(model.blocks[0].ffn.w_gate.weight.grad.norm()) > 1e-4
    assert model.classifier.weight.grad is not None
    assert float(model.classifier.weight.grad.norm()) > 1e-4


def test_two_timescale_gradient_decoupling_b11_2():
    """Theorem B.11.2: Backprop touches latents; lattice W updates only via Hebbian hook."""
    cfg = BookModelConfig(num_entities=8, d_model=32, n_layers=1)
    model = BookGeoFormerTACModel(cfg)
    loss_fn = BookGeoFormerTACLoss(cfg)

    src = torch.tensor([0, 1])
    tgt = torch.tensor([2, 3])

    # Lattice W initially zero
    assert torch.all(model.W_conductance == 0.0)

    out = model(src_indices=src)
    loss, _ = loss_fn(logits=out["logits"], targets=tgt, model=model)
    loss.backward()

    # Lattice buffer has no gradients and is untouched by backprop
    assert model.W_conductance.grad is None
    assert torch.all(model.W_conductance == 0.0)

    # Hebbian update modifies W exclusively
    model.update_conductance(src, tgt, weight_inc=1.0)
    assert model.W_conductance[0, 2] > 0.0
    assert model.W_conductance[1, 3] > 0.0


def test_state_dict_roundtrip():
    """Model state_dict serialization preserves all weights and lattice buffers."""
    cfg = BookModelConfig(num_entities=8, d_model=32, n_layers=1)
    model1 = BookGeoFormerTACModel(cfg)
    model1.update_conductance(torch.tensor([0]), torch.tensor([1]), weight_inc=2.0)

    buf = io.BytesIO()
    torch.save(model1.state_dict(), buf)
    buf.seek(0)

    model2 = BookGeoFormerTACModel(cfg)
    model2.load_state_dict(torch.load(buf, weights_only=True))

    assert torch.equal(model1.W_conductance, model2.W_conductance)
    assert torch.equal(model1.entity_emb.weight, model2.entity_emb.weight)


def test_sdk_parity_loss_components():
    """Bit-exact loss component parity with SDK GeoFormerTACLoss."""
    from geoformer.training.config import TrainingConfig as SDKConfig
    from geoformer.training.losses import GeoFormerTACLoss as SDKLoss

    sdk_cfg = SDKConfig(num_entities=8, num_relations=2, num_classes=8, d_model=32)
    sdk_loss = SDKLoss(sdk_cfg)
    book_cfg = BookModelConfig(num_entities=8, num_relations=2, num_classes=8, d_model=32)
    book_loss = BookGeoFormerTACLoss(book_cfg)

    torch.manual_seed(99)
    z = torch.softmax(torch.randn(3, 8), dim=-1)
    P = torch.softmax(torch.randn(8, 8), dim=-1)

    dir_sdk = sdk_loss.compute_dirichlet_energy(z, P).item()
    dir_book = book_loss.compute_dirichlet_energy(z, P).item()
    ent_sdk = sdk_loss.compute_shannon_entropy(z).item()
    ent_book = book_loss.compute_shannon_entropy(z).item()

    assert dir_sdk == pytest.approx(dir_book, abs=1e-6)
    assert ent_sdk == pytest.approx(ent_book, abs=1e-6)


def test_demo_ledger_contract():
    """Demo ledger runs completely and yields valid parameter accounting."""
    res = demo_ledger()
    assert res["total_params"] > 0
    assert res["packed_kb"] > 0.0
    assert res["grad_norm"] > 0.0
    assert len(res["emergent_hops"]) == 4
