# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 3 tests — every theorem of the TAC chapter, pinned.

Claims guarded here, by name:

  B.3.1  Lyapunov dissipation: energy never rises under the flow.
  B.3.2  Groenwall timescale: exponential decay, and a step count that the
         discrete prediction matches to within one step.
  B.3.3  Double-well bistability: minima at +/-1, barrier at 0, basin
         selection by the sign of the initialization.
  §3.6   Dimension independence: identical step counts across N.
  §3.7   Temperature: equipartition on the harmonic well.
"""

import math

import torch

from gftac.ch03_tac import (
    DoubleWellPotential,
    QuadraticPotential,
    TacRelaxer,
    predicted_steps,
)


# ----------------------------------------------------------------------
# B.3.3 — the landscape itself
# ----------------------------------------------------------------------

def test_double_well_minima_and_barrier():
    v = DoubleWellPotential()
    # minima at +/-1: zero energy, zero gradient
    for m in (-1.0, 1.0):
        s = torch.tensor([m])
        assert abs(float(v.energy(s))) < 1e-12
        assert abs(float(v.grad(s)[0])) < 1e-12
    # barrier at 0: energy 1/4, zero gradient (a saddle), nonzero Hessian sign
    s0 = torch.zeros(1)
    assert abs(float(v.energy(s0)) - 0.25) < 1e-12
    assert abs(float(v.grad(s0)[0])) < 1e-12
    assert abs(v.barrier_height() - 0.25) < 1e-12
    # off the critical points the gradient is nonzero
    assert abs(float(v.grad(torch.tensor([0.5]))[0])) > 0.1


def test_b33_basin_selection_by_sign():
    v = DoubleWellPotential()
    for init, expected_sign in ((0.01, 1.0), (-0.01, -1.0)):
        res = TacRelaxer(v, kappa=1.0).relax(
            init * torch.ones(4), h=0.05, max_steps=2000, tol=1e-4
        )
        assert res.converged
        final = res.final_state
        assert bool((final > 0).all()) if expected_sign > 0 else bool((final < 0).all())
        assert torch.allclose(final.abs(), torch.ones(4), atol=1e-3)


def test_b33_separatrix_stays_put():
    # s = 0 exactly is the unstable fixed point: no drift, no motion.
    res = TacRelaxer(DoubleWellPotential(), kappa=1.0).relax(
        torch.zeros(4), h=0.05, max_steps=2000, tol=1e-6
    )
    assert res.steps == 0
    assert res.converged
    assert bool((res.final_state == 0).all())


# ----------------------------------------------------------------------
# B.3.1 — Lyapunov dissipation
# ----------------------------------------------------------------------

def test_b31_energy_monotone_double_well():
    s0 = 0.6 * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    res = TacRelaxer(DoubleWellPotential(), kappa=1.0).relax(
        s0, h=0.05, max_steps=2000, tol=1e-5
    )
    hist = res.energy_history
    # non-increasing at every single step, and strictly decreasing early on
    assert all(hist[k + 1] <= hist[k] + 1e-12 for k in range(len(hist) - 1))
    assert hist[1] < hist[0]
    assert hist[-1] < 1e-8


def test_b31_dissipation_identity_and_euler_descent():
    # continuous identity: grad F . (ds/dtau) = -kappa ||grad F||^2 <= 0
    v = DoubleWellPotential()
    s = torch.tensor([0.3, -0.7, 1.4])
    grad = v.grad(s)
    flow = -1.0 * grad  # kappa = 1
    assert float(grad @ flow) < 0.0
    assert abs(float(grad @ flow) + float(grad @ grad)) < 1e-12
    # one small Euler step descends
    h = 0.02
    assert float(v.energy(s - h * grad)) < float(v.energy(s))


def test_b31_energy_floor_is_numerical_not_exact():
    # Pushing the tolerance below the float32 resolution stalls the flow at a
    # numerical energy floor: it cannot descend forever in finite precision.
    res = TacRelaxer(DoubleWellPotential(), kappa=1.0).relax(
        torch.full((8,), 0.6), h=0.05, max_steps=400, tol=1e-12
    )
    assert not res.converged  # never reaches 1e-12 in float32
    assert res.steps == 400


# ----------------------------------------------------------------------
# B.3.2 — Groenwall timescale
# ----------------------------------------------------------------------

def test_b32_gronwall_continuous_bound():
    lam, kappa, h = 1.0, 1.0, 0.05
    q = QuadraticPotential.isotropic(16, curvature=lam)
    r = TacRelaxer(q, kappa=kappa)
    s0 = torch.ones(16) / math.sqrt(16.0)
    s = s0.clone()
    for k in range(1, 400):
        s = r.step(s, h)
        tau = k * h
        bound = math.exp(-kappa * lam * tau) * float(torch.linalg.vector_norm(s0))
        assert float(torch.linalg.vector_norm(s)) <= bound + 1e-9


def test_b32_step_prediction_matches_within_one():
    lam, kappa, h = 1.0, 1.0, 0.05
    q = QuadraticPotential.isotropic(16, curvature=lam)
    r = TacRelaxer(q, kappa=kappa)
    s0 = torch.ones(16) / math.sqrt(16.0)
    for tol in (1e-3, 1e-5, 1e-6):
        pred = predicted_steps(kappa, lam, h, float(torch.linalg.vector_norm(s0)), tol)
        actual = r.relax(s0, h=h, max_steps=5000, tol=tol).steps
        assert abs(pred - actual) <= 1


def test_b32_predicted_steps_rejects_unstable_step():
    # h kappa lam >= 2 leaves the stable range: no convergence predicted
    assert predicted_steps(1.0, 1.0, 2.0, 1.0, 1e-6) == -1
    assert predicted_steps(1.0, 1.0, 2.5, 1.0, 1e-6) == -1


def test_b32_laplacian_rate_is_the_spectral_gap():
    # For a Laplacian the constant mode is flat (min curvature 0); the rate
    # that governs relaxation on the mean-zero subspace is the Fiedler value.
    from gftac.ch02_metric_lattice import MetricLattice
    lat = MetricLattice(num_entities=5)
    lat.observe_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    q = QuadraticPotential.from_laplacian(lat.laplacian())
    assert abs(q.min_curvature()) < 1e-6
    assert abs(q.spectral_gap() - lat.fiedler_value()) < 1e-9
    assert q.spectral_gap() > 0.0


# ----------------------------------------------------------------------
# §3.6 — dimension independence
# ----------------------------------------------------------------------

def test_dimension_independence_of_step_count():
    counts = []
    for n in (10, 100, 1000):
        q = QuadraticPotential.isotropic(n, curvature=1.0)
        res = TacRelaxer(q, kappa=1.0).relax(
            torch.ones(n) / math.sqrt(n), h=0.05, max_steps=5000, tol=1e-6
        )
        counts.append(res.steps)
    assert counts[0] == counts[1] == counts[2]


# ----------------------------------------------------------------------
# The relaxer's contract
# ----------------------------------------------------------------------

def test_relax_result_shape_and_history():
    res = TacRelaxer(QuadraticPotential.isotropic(4)).relax(
        torch.ones(4), h=0.1, max_steps=100, tol=1e-3
    )
    final, steps, history, converged = res  # tuple-unpackable
    assert isinstance(steps, int)
    assert len(history) == steps + 1
    assert converged in (True, False)
    assert torch.linalg.vector_norm(final) <= torch.linalg.vector_norm(torch.ones(4))


def test_zero_temperature_is_deterministic():
    q = QuadraticPotential.isotropic(8)
    a = TacRelaxer(q, kappa=1.0, temperature=0.0).relax(torch.ones(8), h=0.05, max_steps=50)
    b = TacRelaxer(q, kappa=1.0, temperature=0.0).relax(torch.ones(8), h=0.05, max_steps=50)
    assert torch.equal(a.final_state, b.final_state)


def test_temperature_equipartition_on_harmonic_well():
    # On F = (1/2) lam ||s||^2 the stationary law has variance T/lam, so the
    # measured standard deviation must approach sqrt(T/lam) — the
    # fluctuation-dissipation preview of §3.7.
    lam, T = 1.0, 0.05
    q = QuadraticPotential.isotropic(1, curvature=lam)
    r = TacRelaxer(q, kappa=1.0, temperature=T, seed=7)
    samples = []
    for _ in range(400):
        s = torch.zeros(1)
        for _ in range(250):
            s = r.step(s, 0.02)
        samples.append(float(s[0]))
    measured = float(torch.tensor(samples).std())
    expected = math.sqrt(T / lam)
    assert abs(measured - expected) / expected < 0.15
