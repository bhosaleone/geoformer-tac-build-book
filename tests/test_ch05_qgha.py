# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 5 tests — every theorem of the geodesic-attention chapter, pinned.

Claims guarded here, by name:

  B.5.1  Softmax attention is a row-stochastic kernel.
  B.5.2  Concentration law: E_a[d] is non-increasing in lambda_geo with
         derivative -Var_a(d); large lambda concentrates on argmin d. The
         entropy derivative is Cov_a(d,S) - lambda Var_a(d), and entropy is
         NOT monotone in general — pinned as a boundary, not hidden.
  B.5.3  Directed-Finsler asymmetry: (L_ij - L_ji) - (S_ij - S_ji) = 2 B_ij.
  §5.6   Parity with the production SDK QuantizedGHA (act_bits=None).
"""

import math

import pytest
import torch

from gftac.ch05_qgha import BookQGHA, attention_entropy


def chain_distance(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()


# ----------------------------------------------------------------------
# B.5.1 — row-stochastic attention
# ----------------------------------------------------------------------

def test_b51_row_stochastic():
    torch.manual_seed(0)
    mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.2)
    h = torch.randn(2, 6, 16)
    _, attn = mod(h, d_matrix=chain_distance(6))
    assert bool((attn >= 0).all())
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_multihead_shapes():
    torch.manual_seed(0)
    mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.1)
    h = torch.randn(3, 7, 16)
    out, attn = mod(h, d_matrix=chain_distance(7))
    assert out.shape == (3, 7, 16)
    assert attn.shape == (3, 4, 7, 7)


def test_d_model_must_divide_heads():
    with pytest.raises(AssertionError):
        BookQGHA(d_model=10, n_heads=4)


# ----------------------------------------------------------------------
# the geodesic penalty itself
# ----------------------------------------------------------------------

def test_geodesic_penalty_shifts_logits_exactly():
    torch.manual_seed(1)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.37)
    h = torch.randn(1, 5, 12)
    d = chain_distance(5)
    plain = mod.scores(h)
    shifted = mod.scores(h, d_matrix=d)
    assert torch.allclose(shifted, plain - 0.37 * d.unsqueeze(0).unsqueeze(0), atol=1e-6)


def test_lambda_zero_recovers_plain_attention():
    torch.manual_seed(2)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.0)
    h = torch.randn(1, 5, 12)
    logits = mod.scores(h, d_matrix=chain_distance(5))
    assert torch.allclose(logits, mod.scores(h), atol=1e-6)


# ----------------------------------------------------------------------
# B.5.2 — the concentration law
# ----------------------------------------------------------------------

def test_b52_expected_distance_is_monotone():
    torch.manual_seed(3)
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.1)
    h = torch.randn(1, 6, 16)
    d = chain_distance(6)
    dd = d.unsqueeze(0).unsqueeze(0)
    eds = []
    for lam in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam)
        mod.load_state_dict(ref.state_dict())
        with torch.no_grad():
            _, a = mod(h, d_matrix=d)
        eds.append(float((a * dd).sum(-1).mean()))
    assert all(eds[i + 1] <= eds[i] + 1e-9 for i in range(len(eds) - 1))
    assert eds[-1] < eds[0]


def test_b52_distance_derivative_is_minus_variance():
    torch.manual_seed(4)
    lam, eps = 0.3, 1e-4
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam)
    h = torch.randn(1, 6, 16)
    d = chain_distance(6)
    dd = d.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _, a = ref(h, d_matrix=d)
        p = a.clamp_min(1e-12)
        dmean = (p * dd).sum(-1)
        var_d = (p * (dd - dmean.unsqueeze(-1)) ** 2).sum(-1)
        analytic = float((-var_d).mean())
        mod2 = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam + eps)
        mod2.load_state_dict(ref.state_dict())
        _, a2 = mod2(h, d_matrix=d)
        fd = (float((a2 * dd).sum(-1).mean()) - float((a * dd).sum(-1).mean())) / eps
    assert abs(analytic - fd) < 0.02


def test_b52_entropy_derivative_formula():
    torch.manual_seed(5)
    lam, eps = 0.3, 1e-4
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam)
    h = torch.randn(1, 6, 16)
    d = chain_distance(6)
    dd = d.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _, a = ref(h, d_matrix=d)
        p = a.clamp_min(1e-12)
        s = ref.scores(h)  # pure QK term
        dmean = (p * dd).sum(-1)
        var_d = (p * (dd - dmean.unsqueeze(-1)) ** 2).sum(-1)
        smean = (p * s).sum(-1, keepdim=True)
        cov = (p * (dd - dmean.unsqueeze(-1)) * (s - smean)).sum(-1)
        analytic = float((cov - lam * var_d).mean())
        mod2 = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam + eps)
        mod2.load_state_dict(ref.state_dict())
        _, a2 = mod2(h, d_matrix=d)
        fd = (attention_entropy(a2) - attention_entropy(a)) / eps
    assert abs(analytic - fd) < 0.02


def test_b52_entropy_is_not_monotone_boundary():
    # The honest boundary: with a base score that offsets the distance tilt,
    # entropy can *rise* as lambda grows. s = [0, 3], d = [0, 1].
    s = torch.tensor([0.0, 3.0])
    d = torch.tensor([0.0, 1.0])
    h0 = float(-(torch.softmax(s, 0) * torch.log_softmax(s, 0)).sum())
    h3 = float(-(torch.softmax(s - 3.0 * d, 0) * torch.log_softmax(s - 3.0 * d, 0)).sum())
    assert h3 > h0  # entropy increased — monotonicity is not universal
    # but with a flat base score the entropy is monotone (Cov term = 0)
    s_flat = torch.zeros(2)
    h0f = float(-(torch.softmax(s_flat, 0) * torch.log_softmax(s_flat, 0)).sum())
    h3f = float(-(torch.softmax(s_flat - 3.0 * d, 0) * torch.log_softmax(s_flat - 3.0 * d, 0)).sum())
    assert h3f < h0f


def test_b52_large_lambda_concentrates_on_argmin():
    torch.manual_seed(6)
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.1)
    h = torch.randn(1, 6, 16)
    d = chain_distance(6)
    mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=80.0)
    mod.load_state_dict(ref.state_dict())
    with torch.no_grad():
        _, a = mod(h, d_matrix=d)
    mean_attn = a.mean(dim=(0, 1))  # [N, N]
    for i in range(6):
        assert int(mean_attn[i].argmax()) == int(d[i].argmin())


# ----------------------------------------------------------------------
# B.5.3 — directed Finsler asymmetry
# ----------------------------------------------------------------------

def test_b53_finsler_logit_shift_is_twice_bias():
    torch.manual_seed(7)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.1)
    h = torch.randn(1, 6, 12)
    d = chain_distance(6)
    B = torch.zeros(6, 6)
    for i in range(6):
        for j in range(6):
            B[i, j] = 0.5 * (j - i)
    assert torch.allclose(B + B.T, torch.zeros_like(B), atol=1e-9)  # antisymmetric
    Lb = mod.scores(h, d_matrix=d, directed_finsler=B)
    L0 = mod.scores(h, d_matrix=d)
    shift = (Lb - Lb.transpose(-2, -1)) - (L0 - L0.transpose(-2, -1))
    assert torch.allclose(shift, 2.0 * B.unsqueeze(0).unsqueeze(0), atol=1e-5)


def test_b53_symmetric_base_makes_symmetric_logits():
    torch.manual_seed(8)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.1)
    # make the base scores symmetric by setting K = Q
    with torch.no_grad():
        mod.k_proj.weight.copy_(mod.q_proj.weight)
    h = torch.randn(1, 6, 12)
    d = chain_distance(6)
    L0 = mod.scores(h, d_matrix=d)
    assert torch.allclose(L0, L0.transpose(-2, -1), atol=1e-5)
    B = torch.zeros(6, 6)
    for i in range(6):
        for j in range(6):
            B[i, j] = 0.4 * (j - i)
    LB = mod.scores(h, d_matrix=d, directed_finsler=B)
    assert not torch.allclose(LB, LB.transpose(-2, -1), atol=1e-3)
    # and the asymmetry is exactly twice the Finsler term
    assert torch.allclose(LB - LB.transpose(-2, -1),
                          2.0 * B.unsqueeze(0).unsqueeze(0), atol=1e-5)


def test_b53_attention_ratio_law():
    # Softmax of a symmetric matrix is NOT symmetric (row normalizers differ).
    # The exact invariant is the ratio law: for symmetric base scores,
    #     A_ij / A_ji = (Z_j / Z_i) * exp(2 B_ij).
    torch.manual_seed(8)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.1)
    with torch.no_grad():
        mod.k_proj.weight.copy_(mod.q_proj.weight)
    h = torch.randn(1, 6, 12)
    d = chain_distance(6)
    B = torch.zeros(6, 6)
    for i in range(6):
        for j in range(6):
            B[i, j] = 0.4 * (j - i)
    with torch.no_grad():
        L = mod.scores(h, d_matrix=d, directed_finsler=B)
        A = torch.softmax(L, dim=-1)
        Z = torch.exp(L).sum(dim=-1)  # [B, heads, N]
    ratio = A / A.transpose(-2, -1).clamp_min(1e-30)
    Zj = Z.unsqueeze(-2)  # Z_j, broadcast along the row axis i
    Zi = Z.unsqueeze(-1)  # Z_i, broadcast along the key axis j
    expected = (Zj / Zi) * torch.exp(2.0 * B.unsqueeze(0).unsqueeze(0))
    assert torch.allclose(ratio, expected, atol=1e-4)


# ----------------------------------------------------------------------
# the projections are ternary, and parity with the SDK
# ----------------------------------------------------------------------

def test_projections_are_ternary():
    from gftac.ch04_ternary import TernaryLinear, ternary_quantize

    mod = BookQGHA(d_model=16, n_heads=4)
    for proj in (mod.q_proj, mod.k_proj, mod.v_proj, mod.out_proj):
        assert isinstance(proj, TernaryLinear)
        w_q, gamma = ternary_quantize(proj.weight)
        w_eff = proj.effective_weight()
        assert torch.allclose(w_eff, torch.round(w_eff / gamma) * gamma, atol=1e-9)


def test_parity_with_sdk():
    from geoformer.layers.attention import QuantizedGHA

    torch.manual_seed(9)
    d_model, n_heads, n = 16, 4, 6
    sdk = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary",
                       lambda_geo=0.1, act_bits=None)
    book = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=0.1)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, n, d_model)
    d = chain_distance(n)
    with torch.no_grad():
        out_sdk, attn_sdk = sdk(h, d_matrix=d)
        out_book, attn_book = book(h, d_matrix=d)
    assert torch.allclose(attn_book, attn_sdk, atol=1e-6)
    assert torch.allclose(out_book, out_sdk, atol=1e-6)
