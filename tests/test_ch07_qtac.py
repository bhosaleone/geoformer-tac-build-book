# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 7 tests — the Q-TAC relaxation loop: Lyapunov dissipation, the
contraction/halting bound, the schedule as a polynomial filter, and parity
with the production SDK.

Claims guarded here, by name:

  B.7.1  Lyapunov dissipation. F is a convex quadratic with gradient
         A h - h_init; the flow dissipates exactly, and the Euler step
         decreases F iff delta_tau < 2 ||g||^2 / (g^T A g), in particular
         whenever delta_tau < 2 / lambda_max(A).
  B.7.2  Contraction and halting. ||h_tau - h*|| <= (1 - delta_tau lambda_min)^tau
         ||h_0 - h*||; the loop halts within its budget, and a larger budget
         never hurts.
  B.7.3  Schedule as a polynomial filter. The k-step propagator depends only on
         the multiset of step sizes (permutation invariance), and the Chebyshev
         step sizes attain the minimax contraction 1 / T_k((lmax+lmin)/(lmax-lmin)).
  §7.5   Parity vs geoformer.layers.tac.QuantizedTAC and
         geoformer.layers.cayley.S5CayleyProjector.
"""

import math

import torch

from gftac.ch07_qtac import (
    BookQuantizedTAC,
    BookS5Projector,
    chain_laplacian,
    chebyshev_schedule,
    make_relaxation_schedule,
    relax_variable,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _setup(seed=0, d_model=32, N=6, lam_dir=0.05, lam_s5=0.05):
    torch.manual_seed(seed)
    L = chain_laplacian(N)
    h_init = torch.randn(1, N, d_model)
    tac = BookQuantizedTAC(d_model=d_model, tau_max=200, delta_tau=0.1,
                           lambda_dir=lam_dir, lambda_s5=lam_s5)
    with torch.no_grad():
        A = tac.system_matrix(L)
    return tac, L, h_init, A.detach()


def _sum_energy(h, h_init, A):
    """The energy whose gradient is exactly A h - h_init (the unnormalized F)."""
    hf = h.reshape(h.shape[0], -1)
    hi = h_init.reshape(h_init.shape[0], -1)
    return (0.5 * (hf @ A * hf).sum() - (hf * hi).sum()
            + 0.5 * (hi * hi).sum()).detach()


def _step(h, h_init, L, tac, dt=None):
    dt = tac.delta_tau if dt is None else dt
    h_sym, _ = tac.s5_projector(h)
    dh = -((h - h_init) + tac.lambda_dir * torch.matmul(L.unsqueeze(0), h)
           + tac.lambda_s5 * (h - h_sym))
    return (h + dt * dh).detach()


def _run(tac, h, L, schedule=None):
    with torch.no_grad():
        return tac(h, L, schedule) if schedule is not None else tac(h, L)


# ----------------------------------------------------------------------
# B.7.1 — Lyapunov dissipation
# ----------------------------------------------------------------------

def test_b71_A_symmetric_positive_definite():
    _, _, _, A = _setup()
    assert torch.allclose(A, A.t(), atol=1e-6)
    eig = torch.linalg.eigvalsh(A)
    # A = I + lambda_s5 (I - P) + lambda_dir (L kron I): eigenvalues >= 1
    assert float(eig.min()) >= 1.0 - 1e-5
    assert float(eig.max()) > float(eig.min())


def test_b71_free_energy_matches_quadratic_form():
    tac, L, h_init, A = _setup()
    h = h_init + 0.3 * torch.randn_like(h_init)
    B, N, D = h.shape
    measured = tac.free_energy(h, h_init, L)
    expected = _sum_energy(h, h_init, A).item() / (B * N)
    assert abs(measured - expected) < 1e-5


def test_b71_gradient_is_Ah_minus_hinit():
    tac, L, h_init, A = _setup()
    h = h_init + 0.4 * torch.randn_like(h_init)
    B, N, D = h.shape
    v = torch.randn_like(h)
    eps = 1e-3
    fd = (tac.free_energy(h + eps * v, h_init, L)
          - tac.free_energy(h - eps * v, h_init, L)) / (2 * eps)
    grad = (h.reshape(B, -1) @ A - h_init.reshape(B, -1)).reshape(h.shape)
    analytic = float((grad * v).sum()) / (B * N)
    assert abs(fd - analytic) < 1e-4


def test_b71_continuous_flow_dissipates_exactly():
    # dF/dtau along dh/dtau = -(Ah - h_init) equals -||Ah - h_init||^2 <= 0.
    tac, L, h_init, A = _setup()
    h = h_init + 0.5 * torch.randn_like(h_init)
    B, N, D = h.shape
    g = (h.reshape(B, -1) @ A - h_init.reshape(B, -1)).reshape(h.shape)
    # direction of the SDK step is exactly -g
    h_sym, _ = tac.s5_projector(h)
    dh = -((h - h_init) + tac.lambda_dir * torch.matmul(L.unsqueeze(0), h)
           + tac.lambda_s5 * (h - h_sym))
    assert torch.allclose(dh, -g, atol=1e-5)
    rate = float((g * dh).sum().detach())
    assert rate <= 0.0


def test_b71_discrete_energy_decreases_for_stable_step():
    tac, L, h_init, A = _setup()
    h = h_init + 0.5 * torch.randn_like(h_init)
    F0 = _sum_energy(h, h_init, A)
    h1 = _step(h, h_init, L, tac)          # delta_tau = 0.1 < 2/lambda_max
    F1 = _sum_energy(h1, h_init, A)
    assert float(F1) < float(F0)


def test_b71_discrete_energy_increases_for_unstable_step():
    tac, L, h_init, A = _setup()
    eig = torch.linalg.eigvalsh(A)
    lam_max = float(eig.max())
    h = h_init + 0.5 * torch.randn_like(h_init)
    F0 = _sum_energy(h, h_init, A)
    h1 = _step(h, h_init, L, tac, dt=1.5 * 2.0 / lam_max)   # beyond stability
    F1 = _sum_energy(h1, h_init, A)
    assert float(F1) > float(F0)


def test_b71_exact_step_size_condition():
    tac, L, h_init, A = _setup()
    B, N, D = h_init.shape
    h = h_init + 0.5 * torch.randn_like(h_init)
    g = (h.reshape(B, -1) @ A - h_init.reshape(B, -1)).reshape(h.shape)
    gnorm2 = float((g * g).sum())
    gAg = float((g.reshape(B, -1) @ A * g.reshape(B, -1)).sum())
    dt_star = 2.0 * gnorm2 / gAg
    # just below the exact threshold F falls; just above it F rises
    F0 = _sum_energy(h, h_init, A)
    assert float(_sum_energy(_step(h, h_init, L, tac, dt=0.99 * dt_star), h_init, A)) < float(F0)
    assert float(_sum_energy(_step(h, h_init, L, tac, dt=1.01 * dt_star), h_init, A)) > float(F0)


def test_b71_energy_decreases_in_expectation_over_seeds():
    drops = []
    for seed in range(6):
        tac, L, h_init, A = _setup(seed=seed)
        h = h_init + 0.5 * torch.randn_like(h_init)
        prev = _sum_energy(h, h_init, A)
        ok = True
        for _ in range(50):
            h = _step(h, h_init, L, tac)
            cur = _sum_energy(h, h_init, A)
            if float(cur) > float(prev) + 1e-9:
                ok = False
            prev = cur
        drops.append(ok)
    assert all(drops)


# ----------------------------------------------------------------------
# B.7.2 — contraction and halting
# ----------------------------------------------------------------------

def test_b72_converges_to_the_fixed_point():
    tac, L, h_init, A = _setup()
    B = h_init.shape[0]
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    h = h_init.clone()
    for _ in range(400):
        h = _step(h, h_init, L, tac)
    assert float((h - h_star).norm()) < 1e-4


def test_b72_contraction_bound_holds():
    tac, L, h_init, A = _setup()
    B = h_init.shape[0]
    eig = torch.linalg.eigvalsh(A)
    lam_min = float(eig.min())
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    e0 = float((h_init - h_star).norm())
    h = h_init.clone()
    for step in range(1, 61):
        h = _step(h, h_init, L, tac)
        bound = (1 - tac.delta_tau * lam_min) ** step * e0
        assert float((h - h_star).norm()) <= bound + 1e-5


def test_b72_halting_step_within_budget():
    tac, L, h_init, _ = _setup()
    _, tau, _ = _run(tac, h_init, L)
    assert tau <= tac.tau_max


def test_b72_larger_budget_never_worse():
    tac, L, h_init, A = _setup()
    B = h_init.shape[0]
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    small, _, _ = _run(tac, h_init, L, schedule=[20])
    large, _, _ = _run(tac, h_init, L, schedule=[60])
    assert float((large - h_star).norm()) < float((small - h_star).norm())


def test_b72_drift_criterion_is_respected():
    # at the halting step the Euler increment is below 1e-4 (or the budget ran out)
    tac, L, h_init, _ = _setup()
    _, tau, _ = _run(tac, h_init, L)
    h = h_init.clone()
    for _ in range(tau - 1):
        h = _step(h, h_init, L, tac)
    increment = float(torch.norm(_step(h, h_init, L, tac) - h, dim=-1).mean())
    assert increment < 1e-4 or tau == tac.tau_max


def test_b72_zero_budget_is_identity():
    tac, L, h_init, _ = _setup()
    out, tau, K_S = _run(tac, h_init, L, schedule=[0])
    assert tau == 0
    assert torch.allclose(out, h_init)


# ----------------------------------------------------------------------
# B.7.3 — the schedule as a polynomial filter
# ----------------------------------------------------------------------

def test_b73_permutation_invariance_float64():
    tac, L, h_init, _ = _setup()
    import copy
    proj64 = copy.deepcopy(tac.s5_projector).double()
    base = [0.05, 0.1, 0.2, 0.4, 0.05, 0.1]
    perm = [0.4, 0.05, 0.1, 0.05, 0.2, 0.1]
    a = relax_variable(h_init.double(), L.double(), base, s5_projector=proj64)
    b = relax_variable(h_init.double(), L.double(), perm, s5_projector=proj64)
    assert float((a - b).abs().max().detach()) < 1e-12


def test_b73_permutation_invariance_float32():
    tac, L, h_init, _ = _setup()
    base = [0.05, 0.1, 0.2, 0.4, 0.05, 0.1]
    perm = [0.4, 0.05, 0.1, 0.05, 0.2, 0.1]
    a = relax_variable(h_init, L, base, s5_projector=tac.s5_projector)
    b = relax_variable(h_init, L, perm, s5_projector=tac.s5_projector)
    assert float((a - b).abs().max().detach()) < 1e-5


def test_b73_multiset_not_order():
    # a reversed schedule gives the same result as the original
    tac, L, h_init, _ = _setup()
    s = [0.03, 0.07, 0.15, 0.31, 0.6]
    a = relax_variable(h_init, L, s, s5_projector=tac.s5_projector)
    b = relax_variable(h_init, L, list(reversed(s)), s5_projector=tac.s5_projector)
    assert float((a - b).abs().max().detach()) < 1e-6


def test_b73_chebyshev_attains_minimax_bound():
    for (lmin, lmax) in [(1.0, 4.0), (1.0, 9.0), (0.5, 3.0)]:
        c = 0.5 * (lmin + lmax)
        r = 0.5 * (lmax - lmin)
        lam = torch.linspace(lmin, lmax, 20001, dtype=torch.float64)
        for k in (1, 2, 3, 4, 5):
            ds = chebyshev_schedule(k, lmin, lmax)
            p = torch.ones_like(lam)
            for d in ds:
                p = p * (1.0 - d * lam)
            worst = float(p.abs().max())
            pred = 1.0 / math.cosh(k * math.acosh(c / r))
            assert abs(worst - pred) < 1e-8


def test_b73_chebyshev_beats_best_fixed():
    for (lmin, lmax) in [(1.0, 4.0), (1.0, 9.0)]:
        c = 0.5 * (lmin + lmax)
        r = 0.5 * (lmax - lmin)
        for k in (2, 3, 4):
            fixed = ((lmax - lmin) / (lmax + lmin)) ** k
            cheb = 1.0 / math.cosh(k * math.acosh(c / r))
            assert cheb < fixed


def test_b73_k1_chebyshev_is_the_best_fixed_step():
    lmin, lmax = 1.0, 9.0
    ds = chebyshev_schedule(1, lmin, lmax)
    assert abs(ds[0] - 2.0 / (lmin + lmax)) < 1e-12


def test_b73_chebyshev_schedule_contracts_real_system():
    tac, L, h_init, A = _setup()
    B = h_init.shape[0]
    eig = torch.linalg.eigvalsh(A)
    lmin, lmax = float(eig.min()), float(eig.max())
    c = 0.5 * (lmin + lmax)
    r = 0.5 * (lmax - lmin)
    h_star = torch.linalg.solve(A, h_init.reshape(B, -1).T).T.reshape(h_init.shape)
    e0 = float((h_init - h_star).norm())
    for k in (2, 3, 4):
        ds = chebyshev_schedule(k, lmin, lmax)
        out = relax_variable(h_init, L, ds, s5_projector=tac.s5_projector)
        err = float((out - h_star).norm().detach())
        bound = (1.0 / math.cosh(k * math.acosh(c / r))) * e0
        assert err <= bound + 1e-4


def test_b73_schedule_kinds_well_formed():
    assert make_relaxation_schedule("fixed", 6, dt=0.2) == [0.2] * 6
    fib = make_relaxation_schedule("fibonacci", 6, dt=0.1)
    assert len(fib) == 6 and abs(fib[0] - 0.1) < 1e-12 and abs(fib[1] - 0.2) < 1e-12
    assert abs(fib[2] - 0.3) < 1e-12 and abs(fib[-1] - 1.3) < 1e-12
    conv = make_relaxation_schedule("convergents", 6, dt=0.1)
    assert len(conv) == 6 and all(v > 0 for v in conv)
    cheb = make_relaxation_schedule("chebyshev", 4, lambda_min=1.0, lambda_max=4.0)
    assert len(cheb) == 4 and all(v > 0 for v in cheb)
    try:
        make_relaxation_schedule("nonsense", 4)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown schedule kind should raise")


# ----------------------------------------------------------------------
# The S5 projector (used by the loop; Chapter 8 develops its theory)
# ----------------------------------------------------------------------

def test_s5_projector_is_an_orthogonal_projection():
    torch.manual_seed(0)
    proj = BookS5Projector(32)
    P = proj.projector_matrix()
    assert torch.allclose(P, P.t(), atol=1e-6)          # symmetric
    assert torch.allclose(P @ P, P, atol=1e-5)          # idempotent


def test_s5_forward_projects_and_reports_complexity():
    torch.manual_seed(1)
    proj = BookS5Projector(32)
    h = torch.randn(2, 5, 32)
    h_sym, K_S = proj(h)
    assert h_sym.shape == h.shape
    assert 0.0 <= float(K_S.detach()) <= 6.9069 + 1e-4


def test_s5_projector_parity_with_sdk():
    from geoformer.layers.cayley import S5CayleyProjector as SDKProj

    torch.manual_seed(2)
    sdk = SDKProj(d_model=32)
    book = BookS5Projector(d_model=32)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, 5, 32)
    with torch.no_grad():
        h_sdk, k_sdk = sdk(h)
        h_bk, k_bk = book(h)
    assert torch.allclose(h_bk, h_sdk, atol=0.0)          # bit-exact
    assert torch.allclose(k_bk, k_sdk, atol=0.0)


# ----------------------------------------------------------------------
# §7.5 — parity with the SDK Q-TAC
# ----------------------------------------------------------------------

def test_parity_qtac_bit_exact():
    from geoformer.layers.tac import QuantizedTAC as SDKTAC

    L = chain_laplacian(6)
    torch.manual_seed(3)
    sdk = SDKTAC(d_model=32, tau_max=8, delta_tau=0.1, lambda_dir=0.05, lambda_s5=0.05)
    book = BookQuantizedTAC(d_model=32, tau_max=8, delta_tau=0.1,
                            lambda_dir=0.05, lambda_s5=0.05)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, 6, 32)
    with torch.no_grad():
        h_sdk, tau_sdk, k_sdk = sdk(h, L)
        h_bk, tau_bk, k_bk = book(h, L)
    assert tau_bk == tau_sdk
    assert float((h_bk - h_sdk).abs().max()) == 0.0
    assert float((k_bk - k_sdk).abs()) == 0.0


def test_parity_qtac_with_explicit_schedule():
    from geoformer.layers.tac import QuantizedTAC as SDKTAC

    L = chain_laplacian(5)
    torch.manual_seed(4)
    sdk = SDKTAC(d_model=16, tau_max=4, delta_tau=0.2)
    book = BookQuantizedTAC(d_model=16, tau_max=4, delta_tau=0.2)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(1, 5, 16)
    sched = [1, 2, 3, 5, 8, 13]
    with torch.no_grad():
        h_sdk, tau_sdk, _ = sdk(h, L, schedule=sched)
        h_bk, tau_bk, _ = book(h, L, schedule=sched)
    assert tau_bk == tau_sdk
    assert float((h_bk - h_sdk).abs().max()) == 0.0


def test_schedule_last_element_is_the_budget():
    # the SDK's `schedule` is a step *budget*: it uses schedule[-1]
    L = chain_laplacian(5)
    tac = BookQuantizedTAC(d_model=16, tau_max=4, delta_tau=0.1)
    h = torch.randn(1, 5, 16)
    _, tau, _ = _run(tac, h, L, schedule=[1, 2, 3])
    assert tau <= 3
    _, tau_long, _ = _run(tac, h, L, schedule=[100])
    assert tau_long <= 100
