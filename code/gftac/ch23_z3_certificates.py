# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 23 — What the Z3 Certificates Do and Don't Say.

The research program ships machine-checked certificates: five TAC theorems
proved with Microsoft Z3, and the GeoFormer Q-series invariants (Q-1 geodesic
contraction, Q-2 Fiedler preservation, Q-3 soliton amplitude bound). This
module does three things.

1. **Reproduces** every certificate, in two columns. ``repo_method`` is what
   the shipped verifier does; ``status`` is the *negated-goal* status, which is
   the shape a proof must have (assert the negation of the goal; ``unsat`` means
   no counterexample exists). For four of the five TAC theorems these coincide;
   TAC-5 ships as a *SAT witness* — a consistency check, not a refutation — and
   we upgrade it to a genuine negated-goal refutation.

2. **Discloses the bounded domain.** Every one of these proofs quantifies over
   the *real* domain (Z3 ``Real``/``Int``), in exact arithmetic. None of them
   says anything about the floating-point routine that actually runs. The
   disclosure lists, per certificate, the exact premises it assumes and the
   things it does not cover.

3. **Upgrades the certificates by property-based randomized verification.**
   Each invariant's precondition ⇒ conclusion is sampled over graph ensembles
   and parameter draws, checked numerically in FP32 and FP64, and the FP32/FP64
   agreement is reported. A deliberately adversarial probe (an increment below
   the binary32 half-spacing) exhibits the exact boundary the real-domain
   certificate does not cross.

The chapter's new theorems, B.23.1 (soundness of the negated-goal method),
B.23.2 (the FP32 sub-ulp limitation), and B.23.3 (the statistical upgrade), are
pinned by ``tests/test_ch23_z3_certificates.py``.

Author: Atmabhan Pandit / Shrikant Bhosale (Eklavya Project)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import z3

__all__ = [
    "ProofRecord",
    "run_tac_theorems",
    "run_geoformer_theorems",
    "all_proof_records",
    "bounded_domain_disclosure",
    "property_check_q1",
    "property_check_q2",
    "build_certificate",
    "certificate_sha256",
]

#: Fixed seed for every randomized ensemble in this chapter. Determinism is a
#: contract: the same seed reproduces the same edges, the same pass counts, and
#: the same certificate hash.
PROPERTY_SEED = 23


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ProofRecord:
    """One reproduced certificate.

    ``status`` is the negated-goal status — the column that says "proved".
    ``repo_method``/``repo_status`` record what the shipped verifier actually
    did, so the chapter can point at the difference (TAC-5) rather than hide it.
    """

    proof_id: str
    name: str
    domain: str
    method: str
    status: str
    proved: bool
    repo_method: str
    repo_status: str
    premises: Tuple[str, ...]
    conclusion: str
    omits: Tuple[str, ...]
    detail: str


def _status(check: z3.CheckSatResult) -> str:
    """Z3's answer as a lower-case word: 'sat', 'unsat', or 'unknown'."""
    return str(check)


def _record(
    proof_id: str,
    name: str,
    status: str,
    premises: Tuple[str, ...],
    conclusion: str,
    omits: Tuple[str, ...],
    detail: str,
    repo_method: str = "negated-goal",
    repo_status: Optional[str] = None,
) -> ProofRecord:
    proved = status == "unsat"
    return ProofRecord(
        proof_id=proof_id,
        name=name,
        domain="real arithmetic (Z3 Real/Int), exact",
        method="negated-goal UNSAT",
        status=status,
        proved=proved,
        repo_method=repo_method,
        repo_status=repo_status if repo_status is not None else status,
        premises=premises,
        conclusion=conclusion,
        omits=omits,
        detail=detail,
    )


# ----------------------------------------------------------------------
# TAC theorems 1–5 (mirrors TAC/src/tac/proofs/z3_verifier.py)
# ----------------------------------------------------------------------

def _tac_1() -> ProofRecord:
    """Lyapunov free-energy dissipation: dF/dt = -kappa * ||grad F||^2 <= 0."""
    kappa, g0, g1, g2 = z3.Reals("kappa g0 g1 g2")
    grad_norm_sq = g0 * g0 + g1 * g1 + g2 * g2
    dF_dt = -kappa * grad_norm_sq

    # Refutation: kappa > 0 and dF/dt > 0.
    s1 = z3.Solver()
    s1.add(kappa > 0)
    s1.add(dF_dt > 0)
    r1 = _status(s1.check())

    # Refutation: dF/dt == 0 with a non-zero gradient.
    s2 = z3.Solver()
    s2.add(kappa > 0)
    s2.add(dF_dt == 0)
    s2.add(z3.Or(g0 != 0, g1 != 0, g2 != 0))
    r2 = _status(s2.check())

    status = "unsat" if (r1 == "unsat" and r2 == "unsat") else "sat"
    return _record(
        "TAC-1",
        "Lyapunov free-energy dissipation",
        status,
        premises=("kappa > 0", "grad F = (g0, g1, g2) in R^3"),
        conclusion="dF/dt = -kappa * ||grad F||^2 <= 0, with equality iff grad F = 0",
        omits=(
            "the discretized Euler-Maruyama update that actually runs",
            "floating-point rounding of the product -kappa * ||grad F||^2",
            "gradient dimensions N > 3",
        ),
        detail="UNSAT: no counterexample to monotone dissipation; equilibrium iff zero gradient.",
    )


def _tac_2() -> ProofRecord:
    """Dimension-independent relaxation bound (quadratic surface)."""
    kappa, lambda_min, s_norm_sq, dissipation = z3.Reals(
        "kappa lambda_min s_norm_sq dissipation"
    )
    s = z3.Solver()
    s.add(kappa > 0)
    s.add(lambda_min > 0)
    s.add(s_norm_sq > 0)
    s.add(dissipation >= 2 * kappa * lambda_min * s_norm_sq)
    s.add(dissipation <= 0)  # refutation
    status = _status(s.check())
    return _record(
        "TAC-2",
        "Dimension-independent relaxation",
        status,
        premises=(
            "kappa > 0",
            "lambda_min > 0",
            "||s||^2 > 0",
            "dissipation >= 2 * kappa * lambda_min * ||s||^2",
        ),
        conclusion="dissipation > 0; relaxation timescale governed only by lambda_min",
        omits=(
            "A >= lambda_min * I is taken as a premise, not derived",
            "finite-precision evaluation of the quadratic form",
        ),
        detail="UNSAT: quadratic dissipation cannot stall while the state is non-zero.",
    )


def _tac_3() -> ProofRecord:
    """Bistable double-well soft-spin invariant."""
    s = z3.Real("s")
    v_prime = s * (s * s - 1)

    # Refutation 1: a stationary point outside {-1, 0, +1}.
    r1solver = z3.Solver()
    r1solver.add(v_prime == 0)
    r1solver.add(s != -1, s != 0, s != 1)
    r1 = _status(r1solver.check())

    # Refutation 2: restoring force fails for |s| > 1.
    r2solver = z3.Solver()
    r2solver.add(z3.Or(s > 1, s < -1))
    r2solver.add(s * v_prime <= 0)
    r2 = _status(r2solver.check())

    # Curvature signs at the three stationary points (exact integers).
    curvature = (3 * (-1) * (-1) - 1 > 0) and (3 * 1 * 1 - 1 > 0) and (3 * 0 * 0 - 1 < 0)
    status = "unsat" if (r1 == "unsat" and r2 == "unsat" and curvature) else "sat"
    return _record(
        "TAC-3",
        "Bistable double-well soft-spin",
        status,
        premises=("s in R", "V(s) = 1/4 * (s^2 - 1)^2"),
        conclusion="stationary points exactly {-1, 0, +1}; +/-1 stable, 0 unstable; confinement for |s| > 1",
        omits=(
            "Langevin noise (the stochastic term of the relaxer)",
            "multi-dimensional soft-spin fields",
        ),
        detail="UNSAT: the stationary set is exactly {-1, 0, +1}, with the stated stability.",
    )


def _tac_4() -> ProofRecord:
    """Exact-cover ground-state equivalence on the shipped micro-instance."""
    s0, s1, beta = z3.Reals("s0 s1 beta")
    cov_res = (s0 + s1 - 1) * (s0 + s1 - 1)
    bin0 = (s0 * (1 - s0)) * (s0 * (1 - s0))
    bin1 = (s1 * (1 - s1)) * (s1 * (1 - s1))
    F = cov_res + beta * (bin0 + bin1)

    s = z3.Solver()
    s.add(beta > 0)
    s.add(F == 0)
    s.add(z3.Not(z3.Or(z3.And(s0 == 1, s1 == 0), z3.And(s0 == 0, s1 == 1))))
    status = _status(s.check())
    return _record(
        "TAC-4",
        "Exact-cover ground state",
        status,
        premises=("beta > 0", "A = [[1], [1]] (a 2-set, 1-element micro-instance)"),
        conclusion="F(s) = 0 iff (s0, s1) in {(1, 0), (0, 1)}",
        omits=(
            "general set systems: only the 2x1 instance is encoded",
            "floating-point evaluation of F",
        ),
        detail="UNSAT: zero energy is equivalent to the discrete exact cover on the encoded instance.",
    )


def _tac_5() -> ProofRecord:
    """S5 representation invariant. Shipped as a SAT witness; upgraded here."""
    fact5 = z3.IntVal(5 * 4 * 3 * 2 * 1)
    dims = [1, 1, 4, 4, 5, 5, 6]
    sum_squares = sum(d * d for d in dims)
    denom = fact5 + 4 * 4 + 1
    goal = z3.And(fact5 == 120, sum_squares == 120, denom == 137)

    # What the shipped verifier does: exhibit a model of the goal (SAT).
    s_sat = z3.Solver()
    s_sat.add(goal)
    repo_status = _status(s_sat.check())

    # The proof-shaped upgrade: the negation is unsatisfiable.
    s_ref = z3.Solver()
    s_ref.add(z3.Not(goal))
    status = _status(s_ref.check())

    return _record(
        "TAC-5",
        "S5 representation invariant (137)",
        status,
        premises=("|S5| = 5! = 120", "irrep dimensions [1, 1, 4, 4, 5, 5, 6]"),
        conclusion="sum(d_i^2) = 120 and 120 + 4^2 + 1 = 137",
        omits=(
            "that the listed integers are the actual irreps (taken from the classification)",
            "meaning for the floating-point Cayley projector",
        ),
        detail=(
            "Shipped as SAT (a consistency witness). The negated-goal upgrade asserts "
            "NOT(120=120 AND 120=120 AND 137=137) and returns UNSAT, which is a genuine "
            "refutation of the arithmetic identity."
        ),
        repo_method="SAT-witness",
        repo_status=repo_status,
    )


def run_tac_theorems() -> List[ProofRecord]:
    """Reproduce and record TAC Theorems 1–5."""
    return [_tac_1(), _tac_2(), _tac_3(), _tac_4(), _tac_5()]


# ----------------------------------------------------------------------
# GeoFormer Q-series (mirrors geoformer/tests/test_z3_invariants.py)
# ----------------------------------------------------------------------

def _geo_q1() -> ProofRecord:
    """Q-1: discrete geodesic contraction."""
    W_t, delta_w, eps = z3.Reals("W_t delta_w eps")
    s = z3.Solver()
    s.add(W_t >= delta_w)
    s.add(delta_w > 0)
    s.add(eps > 0)
    d_t = 1.0 / (W_t + eps)
    d_next = 1.0 / (W_t + delta_w + eps)
    s.add(d_next >= d_t)  # refutation of strict contraction
    status = _status(s.check())
    return _record(
        "Q-1",
        "Discrete geodesic contraction",
        status,
        premises=("W_t >= delta_w", "delta_w > 0", "eps > 0"),
        conclusion="1/(W_t + delta_w + eps) < 1/(W_t + eps): the edge distance strictly contracts",
        omits=(
            "floating-point rounding: in FP32 a small delta_w is absorbed by W_t",
            "the graph-level path distance (this certificate is about one edge)",
        ),
        detail="UNSAT: no real counterexample to strict edge-resistance contraction.",
    )


def _geo_q2() -> ProofRecord:
    """Q-2: quantized Fiedler preservation."""
    N, delta_w, lambda_2_min = z3.Reals("N delta_w lambda_2_min")
    s = z3.Solver()
    s.add(N >= 4)
    s.add(delta_w > 0)
    s.add(lambda_2_min == N * delta_w)
    s.add(lambda_2_min <= 0)  # refutation
    status = _status(s.check())
    return _record(
        "Q-2",
        "Quantized Fiedler preservation",
        status,
        premises=("N >= 4", "delta_w > 0", "lambda_2_min == N * delta_w"),
        conclusion="lambda_2_min > 0: the spectral gap never collapses",
        omits=(
            "the graph-theoretic derivation lambda_2(L) >= N * min W_ij (taken here as the definition)",
            "the FP32/FP64 Laplacian eigensolve",
        ),
        detail="UNSAT: the algebraic-connectivity floor cannot be non-positive.",
    )


def _geo_q3() -> ProofRecord:
    """Q-3: soliton amplitude lower bound (32 hops)."""
    alpha_0, d_avg, alpha_pow_gamma, rho = z3.Reals(
        "alpha_0 d_avg alpha_pow_gamma rho"
    )
    s = z3.Solver()
    s.add(alpha_0 >= 2)
    s.add(d_avg >= 1)
    s.add(d_avg <= 4)
    s.add(alpha_pow_gamma >= alpha_0 * alpha_0)
    s.add(rho == d_avg / alpha_pow_gamma)
    s.add(alpha_pow_gamma > d_avg)
    s.add(rho >= 1.0)  # refutation: leakage does not contract
    status = _status(s.check())
    return _record(
        "Q-3",
        "Soliton amplitude lower bound",
        status,
        premises=(
            "alpha_0 >= 2",
            "1 <= d_avg <= 4",
            "alpha_0^gamma >= alpha_0^2",
            "alpha_0^gamma > d_avg",
        ),
        conclusion="rho = d_avg / alpha_0^gamma < 1: the distractor ratio contracts",
        omits=(
            "that the 32-hop benchmark actually satisfies these premises",
            "the induction over hops: only the per-hop ratio is bounded",
        ),
        detail="UNSAT: the contraction ratio cannot reach 1 under the stated premises.",
    )


def run_geoformer_theorems() -> List[ProofRecord]:
    """Reproduce and record the GeoFormer Q-1/Q-2/Q-3 invariants."""
    return [_geo_q1(), _geo_q2(), _geo_q3()]


def all_proof_records() -> List[ProofRecord]:
    """Every certificate the chapter reproduces: five TAC, three GeoFormer."""
    return run_tac_theorems() + run_geoformer_theorems()


def bounded_domain_disclosure(
    records: Optional[List[ProofRecord]] = None,
) -> List[ProofRecord]:
    """Return the records whose ``premises``/``omits`` form the disclosure.

    The disclosure is not a separate data structure: it is the same records,
    read for what they assume (``premises``) and what they leave out
    (``omits``). Every one is over the real domain.
    """
    return list(records) if records is not None else all_proof_records()


# ----------------------------------------------------------------------
# Property-based randomized verification
# ----------------------------------------------------------------------

def property_check_q1(
    num_graphs: int = 200,
    edges_per_graph: int = 64,
    seed: int = PROPERTY_SEED,
    w_max: float = 8.0,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Sample ensembles and check Q-1 (contraction) in FP32 and FP64.

    Each of ``num_graphs * edges_per_graph`` edges draws ``W >= 1`` and
    ``0 < delta_w <= W`` — the precondition ``W_t >= delta_w > 0`` — and we test
    the conclusion ``d_next < d_t`` with ``d = 1/(W + eps)``. The same edge is
    evaluated once in FP64 and once in FP32, and the two verdicts are compared.

    A second, adversarial probe draws ``delta_w = 2^-25 * W``. That increment
    lies strictly below the binary32 half-spacing, so FP32 rounds ``W + delta_w``
    back to ``W`` and the strict contraction fails, while FP64 still contracts.
    This is the exact bounded-domain boundary of Theorem B.23.2.
    """
    rng = np.random.default_rng(seed)
    total = num_graphs * edges_per_graph

    W = rng.uniform(1.0, w_max, size=total)             # W >= 1 > 0
    delta = rng.uniform(1e-3, 1.0, size=total) * W      # 0 < delta <= W

    # FP64 verdict.
    d_before64 = 1.0 / (W + eps)
    d_after64 = 1.0 / (W + delta + eps)
    hold64 = d_after64 < d_before64

    # FP32 verdict (same edges, cast down).
    W32 = W.astype(np.float32)
    delta32 = delta.astype(np.float32)
    eps32 = np.float32(eps)
    one32 = np.float32(1.0)
    d_before32 = one32 / (W32 + eps32)
    d_after32 = one32 / (W32 + delta32 + eps32)
    hold32 = d_after32 < d_before32

    fp32_passed = int(np.count_nonzero(hold32))
    fp64_passed = int(np.count_nonzero(hold64))
    agreement = bool(np.array_equal(hold32, hold64))

    # Adversarial boundary probe: increment below the FP32 half-spacing.
    probe = _q1_boundary_probe(seed=seed + 1, eps=eps)
    return {
        "num_graphs": num_graphs,
        "edges_per_graph": edges_per_graph,
        "total": total,
        "fp32_passed": fp32_passed,
        "fp64_passed": fp64_passed,
        "agreement": agreement,
        "boundary_total": probe["total"],
        "boundary_fp32_passed": probe["fp32_passed"],
        "boundary_fp64_passed": probe["fp64_passed"],
        "boundary_agreement": probe["agreement"],
    }


def _q1_boundary_probe(
    num_edges: int = 50,
    seed: int = PROPERTY_SEED + 1,
    w_max: float = 8.0,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """FP32 absorbs a sub-half-spacing increment; FP64 does not (B.23.2)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(1.0, w_max, size=num_edges)
    delta = (2.0 ** -25) * W  # strictly below the binary32 half-spacing

    d_before64 = 1.0 / (W + eps)
    d_after64 = 1.0 / (W + delta + eps)
    hold64 = d_after64 < d_before64

    W32 = W.astype(np.float32)
    delta32 = delta.astype(np.float32)
    eps32 = np.float32(eps)
    one32 = np.float32(1.0)
    d_before32 = one32 / (W32 + eps32)
    d_after32 = one32 / (W32 + delta32 + eps32)
    hold32 = d_after32 < d_before32

    return {
        "total": num_edges,
        "fp32_passed": int(np.count_nonzero(hold32)),
        "fp64_passed": int(np.count_nonzero(hold64)),
        "agreement": bool(np.array_equal(hold32, hold64)),
    }


def property_check_q2(
    num_lattices: int = 50,
    n: int = 12,
    seed: int = PROPERTY_SEED,
    w_min: float = 0.5,
    w_max: float = 8.0,
) -> Dict[str, Any]:
    """Sample dense lattices and check Q-2 (Fiedler floor) in FP32 and FP64.

    Each lattice draws directed conductances in ``[w_min, w_max]``, symmetrizes
    to ``A = W + W^T``, and sets ``delta_w`` to the minimum off-diagonal entry of
    ``A`` — so the hypothesis of Theorem B.2.3 (every pair carries symmetrized
    conductance at least ``delta_w``) holds by construction. The check is the
    conclusion ``lambda_2(L) >= N * delta_w``, evaluated on the Laplacian
    ``L = D - A`` in both precisions.
    """
    rng = np.random.default_rng(seed)
    mask = ~np.eye(n, dtype=bool)

    fp32_passed = 0
    fp64_passed = 0
    agreement = True
    min_margin = None

    for _ in range(num_lattices):
        W = rng.uniform(w_min, w_max, size=(n, n))
        np.fill_diagonal(W, 0.0)
        A = W + W.T
        delta_w = float(np.min(A[mask]))

        L64 = np.diag(A.sum(axis=1)) - A
        lambda2_64 = float(np.linalg.eigvalsh(L64)[1])
        hold64 = lambda2_64 >= n * delta_w

        A32 = A.astype(np.float32)
        L32 = np.diag(A32.sum(axis=1).astype(np.float32)) - A32
        lambda2_32 = float(np.linalg.eigvalsh(L32)[1])
        hold32 = lambda2_32 >= np.float32(n) * np.float32(delta_w)

        fp64_passed += int(hold64)
        fp32_passed += int(hold32)
        agreement = agreement and (hold64 == hold32)

        margin = lambda2_64 - n * delta_w
        min_margin = margin if min_margin is None else min(min_margin, margin)

    return {
        "num_lattices": num_lattices,
        "n": n,
        "fp32_passed": fp32_passed,
        "fp64_passed": fp64_passed,
        "agreement": bool(agreement),
        "min_margin": float(min_margin) if min_margin is not None else 0.0,
    }


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------

def _canonical(payload: Dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def certificate_sha256(payload: Dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of ``payload`` (the 'sha256' field is ignored)."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def build_certificate(
    records: Optional[List[ProofRecord]] = None,
) -> Dict[str, Any]:
    """Assemble the chapter's certificate: proofs + property-based results + hash.

    The payload contains no timestamps and no wall-clock values, so its SHA-256
    is stable run to run. The digest is over the canonical JSON of the payload
    with the ``sha256`` field removed.
    """
    recs = list(records) if records is not None else all_proof_records()
    q1 = property_check_q1()
    q2 = property_check_q2()

    payload: Dict[str, Any] = {
        "chapter": "23",
        "title": "What the Z3 Certificates Do and Don't Say",
        "z3_version": z3.get_version_string(),
        "proofs": {r.proof_id: r.status for r in recs},
        "repo_sat_witnesses": [r.proof_id for r in recs if r.repo_method == "SAT-witness"],
        "all_goals_proved": bool(all(r.proved for r in recs)),
        "disclosure": {
            "certificates": len(recs),
            "over_reals": len(recs),
            "cover_floating_point": 0,
        },
        "property_based": {
            "q1": {
                "graphs": q1["num_graphs"],
                "edges_per_graph": q1["edges_per_graph"],
                "edges": q1["total"],
                "fp32_passed": q1["fp32_passed"],
                "fp64_passed": q1["fp64_passed"],
                "agreement": q1["agreement"],
                "boundary_fp32_passed": q1["boundary_fp32_passed"],
                "boundary_fp64_passed": q1["boundary_fp64_passed"],
                "boundary_total": q1["boundary_total"],
            },
            "q2": {
                "lattices": q2["num_lattices"],
                "nodes": q2["n"],
                "fp32_passed": q2["fp32_passed"],
                "fp64_passed": q2["fp64_passed"],
                "agreement": q2["agreement"],
                "min_margin": round(q2["min_margin"], 3),
            },
        },
    }
    payload["sha256"] = certificate_sha256(payload)
    return payload


def _time_proofs(fn) -> Tuple[List[ProofRecord], float]:
    """Run a proof suite and report its wall time (the only run-dependent value)."""
    t0 = time.perf_counter()
    records = fn()
    return records, time.perf_counter() - t0
