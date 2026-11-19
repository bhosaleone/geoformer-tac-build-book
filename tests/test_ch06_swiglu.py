# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 6 tests — the RMSNorm scale law, the SiLU gate, parameter
accounting, and parity with the production SDK.

Claims guarded here, by name:

  B.6.1  RMSNorm scale law: n(cx) = rho(c) n(x), direction invariant.
  B.6.2  SiLU gate suppression: bounds, minimum at z* = -1.2784645.
  B.6.3  Parameter accounting: 3 d_model d_ff = 8 d_model^2 (unfloored).
  §6.5   Parity vs geoformer.layers.ffn.TernarySwiGLU (weight-only AND the
         SDK's default 8-bit activations) and geoformer.models.block.RMSNorm.
"""

import math

import torch
import torch.nn.functional as F

from gftac.ch06_swiglu import (
    BookRMSNorm,
    BookSwiGLU,
    int_activation_quantize,
    silu_minimum,
    swiglu_packed_bytes,
    swiglu_param_count,
)


# ----------------------------------------------------------------------
# B.6.1 — RMSNorm scale law
# ----------------------------------------------------------------------

def _n(x, eps):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def test_b61_direction_invariant():
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    eps = 1e-6
    nx = _n(x, eps)
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        ncx = _n(c * x, eps)
        cos = F.cosine_similarity(ncx, nx, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)


def test_b61_rho_formula():
    torch.manual_seed(1)
    x = torch.randn(2, 12)
    eps = 1e-6
    m = float(x.pow(2).mean())
    nx = _n(x, eps)
    for c in (0.05, 0.5, 1.0, 2.0, 50.0):
        ncx = _n(c * x, eps)
        rho = float(ncx.norm() / nx.norm())
        rho_pred = math.sqrt((m + eps) / (m + eps / c ** 2))
        assert abs(rho - rho_pred) < 1e-5


def test_b61_unit_rms():
    torch.manual_seed(2)
    x = torch.randn(4, 16)
    eps = 1e-6
    m = float(x.pow(2).mean())
    nx = _n(x, eps)
    assert abs(float(nx.pow(2).mean().sqrt()) - math.sqrt(m / (m + eps))) < 1e-6
    # with eps = 0 the output has exactly unit RMS
    n0 = _n(x, 0.0)
    assert abs(float(n0.pow(2).mean().sqrt()) - 1.0) < 1e-6


def test_b61_eps_zero_is_exact_scale_invariance():
    torch.manual_seed(3)
    x = torch.randn(2, 10)
    for c in (0.001, 0.5, 3.0, 1000.0):
        assert torch.allclose(_n(c * x, 0.0), _n(x, 0.0), atol=1e-6)


def test_b61_scale_gain_law():
    # RMSNorm(cx) = rho(c) RMSNorm(x), gain included.
    torch.manual_seed(4)
    norm = BookRMSNorm(8)
    with torch.no_grad():
        norm.scale.uniform_(0.5, 1.5)
    x = torch.randn(2, 8)
    eps = norm.eps
    m = float(x.pow(2).mean())
    for c in (0.5, 2.0, 20.0):
        rho = math.sqrt((m + eps) / (m + eps / c ** 2))
        assert torch.allclose(norm(c * x), rho * norm(x), atol=1e-5)


# ----------------------------------------------------------------------
# B.6.2 — the SiLU gate
# ----------------------------------------------------------------------

def test_b62_sign_bounds():
    z = torch.linspace(-6, 6, 2001)
    g = F.silu(z)
    pos = z > 0
    neg = z < 0
    assert bool((g[pos] > 0).all()) and bool((g[pos] < z[pos]).all())
    assert bool((g[neg] < 0).all()) and bool((g[neg] > z[neg]).all())
    assert float(F.silu(torch.tensor(0.0))) == 0.0


def test_b62_minimum_location_and_value():
    z_star, g_min = silu_minimum()
    assert abs(z_star - (-1.2784645)) < 1e-4
    assert abs(g_min - (-0.2784645)) < 1e-5
    # nothing dips below the minimum
    z = torch.linspace(-8, 8, 100001, dtype=torch.float64)
    assert float(F.silu(z).min()) >= g_min - 1e-9


def test_b62_limits():
    # g(z) -> 0 as z -> -infinity
    assert abs(float(F.silu(torch.tensor(-50.0)))) < 1e-18
    # g(z)/z -> 1 as z -> +infinity
    for z in (20.0, 40.0):
        ratio = float(F.silu(torch.tensor(z))) / z
        assert abs(ratio - 1.0) < 1e-6


def test_b62_gate_statistics():
    torch.manual_seed(1)
    z = torch.randn(1_000_000)
    g = F.silu(z)
    assert abs(float((g < 0).float().mean()) - 0.5) < 0.005
    assert abs(float(g.abs().mean()) - 0.3990) < 0.005
    assert abs(float((g.abs() < 0.1).float().mean()) - 0.1618) < 0.005


# ----------------------------------------------------------------------
# B.6.3 — parameter accounting
# ----------------------------------------------------------------------

def test_b63_default_hidden_width():
    assert BookSwiGLU(64).d_ff == int(8 / 3 * 64) == 170
    assert BookSwiGLU(12).d_ff == int(8 / 3 * 12) == 32


def test_b63_param_count_formula():
    assert swiglu_param_count(64) == 3 * 64 * 170 == 32_640
    # for d_model divisible by 3 the floored count equals the ideal 8 d_model^2
    for d_model in (12, 30, 96):
        assert swiglu_param_count(d_model) == 8 * d_model ** 2


def test_b63_matches_conventional_ffn():
    # a gated block with d_ff = 8/3 d_model costs the same as a hidden-4x FFN
    for d_model in (12, 24, 48, 96):
        gated = swiglu_param_count(d_model)
        conventional = 2 * d_model * (4 * d_model)
        assert gated == conventional


def test_b63_packed_footprint():
    n = swiglu_param_count(64)
    assert swiglu_packed_bytes(64) == (n + 4) // 5 == 6_528


def test_swiglu_shapes_and_act_bits():
    torch.manual_seed(5)
    ffn = BookSwiGLU(16, act_bits=None)
    x = torch.randn(2, 5, 16)
    assert ffn(x).shape == (2, 5, 16)
    assert ffn.act_bits is None
    ffn8 = BookSwiGLU(16, act_bits=8)
    assert ffn8(x).shape == (2, 5, 16)


# ----------------------------------------------------------------------
# §6.5 — parity with the SDK
# ----------------------------------------------------------------------

def test_activation_quantizer_parity():
    from geoformer.quant.ste import AbsMaxQuantizer

    torch.manual_seed(6)
    x = torch.randn(2, 5, 16) * 3.0
    x_q, scale = int_activation_quantize(x, bits=8)
    x_q_sdk, scale_sdk = AbsMaxQuantizer.quantize_activation_int(x, bits=8)
    assert torch.allclose(x_q, x_q_sdk, atol=1e-6)
    assert torch.allclose(scale, scale_sdk, atol=1e-6)


def test_parity_rmsnorm():
    from geoformer.models.block import RMSNorm as SDKRMSNorm

    torch.manual_seed(7)
    sdk = SDKRMSNorm(16)
    book = BookRMSNorm(16)
    book.load_state_dict(sdk.state_dict())
    x = torch.randn(3, 5, 16)
    with torch.no_grad():
        assert torch.allclose(book(x), sdk(x), atol=1e-7)


def test_parity_swiglu_weight_only():
    from geoformer.layers.ffn import TernarySwiGLU as SDKFFN

    torch.manual_seed(8)
    sdk = SDKFFN(d_model=16, mode="ternary")
    for lin in (sdk.w_gate, sdk.w_up, sdk.w_down):
        lin.act_bits = None
    book = BookSwiGLU(16, act_bits=None)
    book.load_state_dict(sdk.state_dict())
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert torch.allclose(book(x), sdk(x), atol=1e-6)


def test_parity_swiglu_sdk_default_8bit():
    from geoformer.layers.ffn import TernarySwiGLU as SDKFFN

    torch.manual_seed(9)
    sdk = SDKFFN(d_model=16, mode="ternary")  # default act_bits = 8
    book = BookSwiGLU(16, act_bits=8)
    book.load_state_dict(sdk.state_dict())
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert torch.allclose(book(x), sdk(x), atol=1e-6)


def test_activation_quantization_is_not_identity():
    torch.manual_seed(10)
    book = BookSwiGLU(16, act_bits=8)
    book_wo = BookSwiGLU(16, act_bits=None)
    book_wo.load_state_dict(book.state_dict())
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert not torch.allclose(book(x), book_wo(x), atol=1e-4)
