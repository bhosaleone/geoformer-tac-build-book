# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 9 tests — Assembling the Block & the Full Model.

Guards the theorems and implementation of Chapter 9:
  B.9.1  Quasi-crystalline non-collision & determinism: zero collisions on N <= 2048.
  B.9.2  Parameter and footprint accounting formulas: exact match with module parameters.
  B.9.3  Pre-RMSNorm residual gradient flow, forward finiteness, and state_dict roundtrip.
  Parity: Bit-exact agreement with geoformer.models.geoformer.GeoFormerBlock.
"""

import io
import math
import pytest
import torch

from gftac.ch09_model import (
    BookGeoFormer,
    BookGeoFormerBlock,
    block_packed_bytes,
    block_parameter_count,
    chain_laplacian,
    encoding_collision_rate,
    quasi_crystalline_encoding,
)


# ----------------------------------------------------------------------
# B.9.1 — Quasi-crystalline positional encodings
# ----------------------------------------------------------------------

def test_b91_unit_sphere_normalization():
    for N in (64, 256, 1024):
        enc = quasi_crystalline_encoding(N, dim=32, basis="golden")
        norms = torch.norm(enc, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_b91_zero_collision_rate_on_large_sequences():
    for N in (256, 512, 1024, 2048):
        enc = quasi_crystalline_encoding(N, dim=64, basis="golden")
        c_rate = encoding_collision_rate(enc, rtol=1e-5)
        assert c_rate == 0.0


def test_b91_sqrt2_basis_zero_collisions():
    enc = quasi_crystalline_encoding(512, dim=32, basis="sqrt2")
    c_rate = encoding_collision_rate(enc, rtol=1e-5)
    assert c_rate == 0.0


def test_b91_determinism():
    enc1 = quasi_crystalline_encoding(128, dim=32, basis="golden")
    enc2 = quasi_crystalline_encoding(128, dim=32, basis="golden")
    assert torch.equal(enc1, enc2)


# ----------------------------------------------------------------------
# B.9.2 — Parameter and footprint accounting
# ----------------------------------------------------------------------

def test_b92_block_parameter_count_matches_module():
    for d_model in (16, 32, 64):
        block = BookGeoFormerBlock(d_model=d_model, n_heads=4)
        measured = sum(p.numel() for p in block.parameters())
        predicted = block_parameter_count(d_model)
        assert measured == predicted


def test_b92_packed_byte_footprint():
    d_model = 32
    d_ff = math.floor(8.0 / 3.0 * d_model)
    ternary_weights = 4 * d_model * d_model + 3 * d_model * d_ff + 52 * d_model
    expected_packed = math.ceil(ternary_weights / 5.0) + (2 * d_model * 4)
    assert block_packed_bytes(d_model) == expected_packed


def test_b92_full_model_parameter_count():
    model = BookGeoFormer(
        vocab_size=100,
        num_classes=5,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=128,
    )
    total_p, packed_kb = model.count_parameters()
    assert total_p == sum(p.numel() for p in model.parameters())
    assert packed_kb == (total_p * 0.20) / 1024.0


# ----------------------------------------------------------------------
# B.9.3 — Forward, backward, and checkpoint roundtrip
# ----------------------------------------------------------------------

def test_b93_forward_finiteness_and_shape():
    model = BookGeoFormer(
        vocab_size=200,
        num_classes=10,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        node_encoding="quasi",
    )
    x = torch.randint(0, 200, (3, 12))
    out = model(x)
    assert out.shape == (3, 12, 10)
    assert torch.isfinite(out).all()


def test_b93_continuous_feature_projection():
    model = BookGeoFormer(
        in_features=16,
        num_classes=4,
        d_model=32,
        n_layers=1,
        n_heads=4,
    )
    # Batch of continuous vectors [B, in_features]
    x = torch.randn(5, 16)
    out = model(x)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()


def test_b93_backward_gradient_propagation():
    torch.manual_seed(42)
    model = BookGeoFormer(
        vocab_size=100,
        num_classes=5,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
    )
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradients flow to head, blocks, and embeddings
    assert model.head.weight.grad is not None
    assert float(model.head.weight.grad.norm()) > 0.0
    for block in model.blocks:
        assert block.attn.q_proj.weight.grad is not None
        assert float(block.attn.q_proj.weight.grad.norm()) > 0.0
        assert block.ffn.w_gate.weight.grad is not None
        assert float(block.ffn.w_gate.weight.grad.norm()) > 0.0
    assert model.tok_emb.weight.grad is not None
    assert float(model.tok_emb.weight.grad.norm()) > 0.0


def test_b93_state_dict_roundtrip():
    torch.manual_seed(99)
    model1 = BookGeoFormer(
        vocab_size=150,
        num_classes=8,
        d_model=32,
        n_layers=2,
        n_heads=4,
    )
    x = torch.randint(0, 150, (2, 10))
    with torch.no_grad():
        out1 = model1(x)

    buf = io.BytesIO()
    torch.save(model1.state_dict(), buf)
    buf.seek(0)

    model2 = BookGeoFormer(
        vocab_size=150,
        num_classes=8,
        d_model=32,
        n_layers=2,
        n_heads=4,
    )
    model2.load_state_dict(torch.load(buf, weights_only=True))
    with torch.no_grad():
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)


# ----------------------------------------------------------------------
# SDK Parity
# ----------------------------------------------------------------------

def test_parity_geoformer_block_bit_exact():
    from geoformer.models.geoformer import GeoFormerBlock as SDKBlock

    for d_model in (16, 32):
        torch.manual_seed(d_model)
        sdk_block = SDKBlock(d_model=d_model, n_heads=4, tau_max=2, lambda_geo=0.1, lambda_dir=0.05, dropout=0.0)
        for p in (sdk_block.attn.q_proj, sdk_block.attn.k_proj, sdk_block.attn.v_proj, sdk_block.attn.out_proj,
                  sdk_block.ffn.w_gate, sdk_block.ffn.w_up, sdk_block.ffn.w_down):
            p.act_bits = None

        book_block = BookGeoFormerBlock(d_model=d_model, n_heads=4, tau_max=2, lambda_geo=0.1, lambda_dir=0.05, dropout=0.0)
        book_block.load_state_dict(sdk_block.state_dict())

        test_h = torch.randn(2, 6, d_model)
        L_test = chain_laplacian(6)
        idx = torch.arange(6, dtype=torch.float32)
        d_test = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

        with torch.no_grad():
            out_sdk = sdk_block(test_h, d_matrix=d_test, L_matrix=L_test)
            out_bk = book_block(test_h, d_matrix=d_test, L_matrix=L_test)

        assert torch.allclose(out_bk, out_sdk, atol=0.0)
