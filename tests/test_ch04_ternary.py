# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 4 tests — every theorem of the ternary-synapse chapter, pinned.

Claims guarded here, by name:

  B.4.1  Scale invariance and variance preservation of ternary quantization.
  B.4.2  Straight-through estimation has the identity gradient.
  B.4.3  Five trits pack losslessly into one byte.
  §4.6   The 249,665-parameter foundation model packs to 49,933 B.
  §4.7   Parity with the production SDK (geoformer.quant.ternary_linear).
"""

import math

import pytest
import torch

from gftac.ch04_ternary import (
    FOUNDATION_PARAMS,
    TernaryLinear,
    pack_trits,
    packed_bits_per_parameter,
    packed_bytes,
    ternary_quantize,
    unpack_trits,
    variance_preserving_init_,
)


# ----------------------------------------------------------------------
# B.4.1 — the trit and its scale
# ----------------------------------------------------------------------

def test_ternary_quantize_trit_domain_and_scale():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    w_q, gamma = ternary_quantize(w)
    # every output is a trit
    assert bool(((w_q == -1.0) | (w_q == 0.0) | (w_q == 1.0)).all())
    # gamma is the mean absolute weight (the +eps guard vanishes in float32)
    assert abs(float(gamma) - float(w.abs().mean())) < 1e-6
    # effective weights are integer multiples of gamma
    w_eff = w_q * gamma
    assert torch.allclose(w_eff, torch.round(w_eff / gamma) * gamma, atol=1e-9)


def test_b41_scale_invariance():
    torch.manual_seed(1)
    w = torch.randn(16, 8)
    q1, _ = ternary_quantize(w)
    for c in (0.001, 0.1, 10.0, 1000.0):
        q2, _ = ternary_quantize(c * w)
        assert torch.equal(q1, q2)  # gamma is homogeneous of degree one


def test_b41_variance_preservation_ratio():
    torch.manual_seed(2)
    w = torch.empty(256, 512)
    variance_preserving_init_(w, mode="ternary")
    w_q, gamma = ternary_quantize(w)
    e_q2 = float((w_q ** 2).mean())
    # the shape-set second moment of the normalized trits (2/3 is the
    # idealized target; a Gaussian draw lands near 0.69)
    assert abs(e_q2 - 2.0 / 3.0) < 0.05
    var_eff = float((w_q * gamma).var())
    target = (2.0 / 3.0) * float(gamma ** 2)
    assert abs(var_eff / target - 1.0) < 0.10


def test_variance_preserving_init_scale():
    torch.manual_seed(3)
    w = torch.empty(8, 400)
    variance_preserving_init_(w, mode="ternary")
    assert abs(float(w.std()) - math.sqrt(1.5 / 400)) < 0.01
    wi = torch.empty(8, 400)
    variance_preserving_init_(wi, mode="int8")
    assert abs(float(wi.std()) - math.sqrt(2.0 / 400)) < 0.01


# ----------------------------------------------------------------------
# B.4.2 — straight-through estimation
# ----------------------------------------------------------------------

def test_b42_ste_gradient_identity():
    torch.manual_seed(4)
    layer = TernaryLinear(4, 3, bias=False)
    x = torch.randn(2, 4)
    out = layer(x)
    out.sum().backward()
    # dL/dw must equal the gradient of the de-quantized layer, L = sum(out)
    expected = torch.ones_like(out).T @ x
    assert torch.allclose(layer.weight.grad, expected, atol=1e-6)


def test_b42_true_map_is_locally_constant():
    # A tiny nudge does not change the trit pattern, so the true derivative
    # is zero almost everywhere; STE supplies the missing gradient.
    torch.manual_seed(5)
    w = torch.randn(32, 16) * 0.3
    q1, _ = ternary_quantize(w)
    q2, _ = ternary_quantize(w + 1e-4)
    assert int((q1 != q2).sum()) == 0


def test_b42_ste_gradient_is_nonzero_and_flows():
    torch.manual_seed(6)
    layer = TernaryLinear(5, 5, bias=False)
    layer(torch.randn(3, 5)).sum().backward()
    assert float(layer.weight.grad.norm()) > 0.0
    assert layer.weight.grad.shape == layer.weight.shape


# ----------------------------------------------------------------------
# B.4.3 — packing five trits per byte
# ----------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 6, 7, 10, 13, 100, 243, 1000])
def test_b43_pack_unpack_roundtrip(n):
    torch.manual_seed(n)
    trits = torch.randint(-1, 2, (n,))
    packed = pack_trits(trits)
    assert packed.numel() == packed_bytes(n) == (n + 4) // 5
    assert bool((packed <= 255).all())
    restored = unpack_trits(packed, n)
    assert torch.equal(restored, trits)


def test_b43_map_is_injective_over_all_243_patterns():
    # enumerate every 5-trit group and check the packed bytes are distinct
    seen = set()
    for code in range(3 ** 5):
        digits = []
        c = code
        for _ in range(5):
            digits.append((c % 3) - 1)
            c //= 3
        b = int(pack_trits(torch.tensor(digits))[0])
        assert 0 <= b <= 255
        seen.add(b)
    assert len(seen) == 243  # injective, no collisions


def test_b43_pack_rejects_out_of_domain():
    with pytest.raises(ValueError):
        pack_trits([0, 1, 2])
    with pytest.raises(ValueError):
        pack_trits([-2, 0, 1])


def test_b43_known_byte():
    # [+1, 0, -1, +1, 0] -> digits [2,1,0,2,1] -> 2 + 3 + 0 + 54 + 81 = 140
    assert int(pack_trits([1, 0, -1, 1, 0])[0]) == 140


# ----------------------------------------------------------------------
# §4.6 — the foundation footprint
# ----------------------------------------------------------------------

def test_footprint_of_foundation_model():
    n = FOUNDATION_PARAMS
    assert n == 249_665
    assert packed_bytes(n) == 49_933
    assert abs(packed_bytes(n) / 1024.0 - 48.76) < 0.01
    assert abs(packed_bits_per_parameter() - 1.6) < 1e-12
    # a trit costs log2(3) = 1.585 bits of information; packing five per
    # byte spends 1.6, so the scheme is log2(3)/1.6 = 99.06% efficient
    efficiency = math.log2(3) / packed_bits_per_parameter()
    assert abs(efficiency - 0.9906) < 0.001


def test_footprint_beats_fp32_by_twenty():
    n = FOUNDATION_PARAMS
    fp32_bytes = n * 4
    ternary_bytes = packed_bytes(n)
    assert fp32_bytes / ternary_bytes > 19.9


# ----------------------------------------------------------------------
# §4.7 — parity with the production SDK
# ----------------------------------------------------------------------

def test_parity_ternary_quantize_with_sdk():
    from geoformer.quant.ste import AbsMaxQuantizer

    torch.manual_seed(7)
    w = torch.randn(32, 16)
    q_book, g_book = ternary_quantize(w)
    q_sdk, g_sdk = AbsMaxQuantizer.quantize_weight_ternary(w)
    assert torch.equal(q_book, q_sdk)
    assert abs(float(g_book) - float(g_sdk)) < 1e-12


def test_parity_forward_with_sdk():
    from geoformer.quant.ternary_linear import QuantizedLinear

    torch.manual_seed(8)
    sdk = QuantizedLinear(6, 4, bias=False, mode="ternary", act_bits=None)
    book = TernaryLinear(6, 4, bias=False)
    book.load_state_dict(sdk.state_dict())
    x = torch.randn(3, 6)
    assert torch.allclose(book(x), sdk(x), atol=1e-6)


def test_parity_foundation_param_count_from_sdk():
    from geoformer.models.geoformer import GeoFormer

    model = GeoFormer(
        in_features=1433, num_classes=7, d_model=64, n_layers=3, n_heads=4,
        mode="ternary",
    )
    n = sum(p.numel() for p in model.parameters())
    assert n == FOUNDATION_PARAMS
    # the SDK's own 0.20-bytes-per-param approximation matches the exact
    # ceil(n/5) packing to better than a byte
    assert abs(n * 0.20 - packed_bytes(n)) < 1.0
