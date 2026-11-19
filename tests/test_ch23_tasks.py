# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch23_tasks.py — Pin every printed line of the Chapter 23 tasks.

Each task's stdout is compared line-for-line against the transcript printed in
``chapters/ch23_z3_certificates.md``. The single run-dependent line (the Z3 wall
time in Task 1) is matched by a regular expression, exactly as the book wraps it
in «...».
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

from gftac import ch23_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _lines(out: str):
    return [line.rstrip() for line in out.splitlines()]


def _assert_exact(out: str, expected):
    actual = _lines(out)
    assert actual == expected, "\n".join(
        f"line {i}: got {a!r} want {e!r}"
        for i, (a, e) in enumerate(zip(actual, expected))
    ) or f"length {len(actual)} != {len(expected)}"


# ----------------------------------------------------------------------
# Task 1
# ----------------------------------------------------------------------

TASK1_EXPECTED = [
    "TAC Z3 Certificate — five theorems:",
    "  • TAC-1  Lyapunov free-energy dissipation     status=unsat proved=True  repo=negated-goal",
    "  • TAC-2  Dimension-independent relaxation     status=unsat proved=True  repo=negated-goal",
    "  • TAC-3  Bistable double-well soft-spin       status=unsat proved=True  repo=negated-goal",
    "  • TAC-4  Exact-cover ground state             status=unsat proved=True  repo=negated-goal",
    "  • TAC-5  S5 representation invariant (137)    status=unsat proved=True  repo=SAT-witness",
    "  • All five goals proved:  True",
    # The wall-time line is run-dependent; see VOLATILE_LINE_RE.
    "  • Z3 wall time:           0.0000 s (run-dependent)",
]

VOLATILE_LINE_RE = re.compile(r"^  • Z3 wall time:\s+\d+\.\d{4} s \(run-dependent\)$")


def test_task_1():
    actual = _lines(_capture(ch23_tasks.task_1))
    assert len(actual) == len(TASK1_EXPECTED)
    for line, expected in zip(actual, TASK1_EXPECTED):
        if expected == TASK1_EXPECTED[-1]:
            assert VOLATILE_LINE_RE.fullmatch(line), line
        else:
            assert line == expected


# ----------------------------------------------------------------------
# Task 2
# ----------------------------------------------------------------------

TASK2_EXPECTED = [
    "GeoFormer Z3 Certificate — Q-series invariants:",
    "  • Q-1  Discrete geodesic contraction        status=unsat proved=True  repo=negated-goal",
    "  • Q-2  Quantized Fiedler preservation       status=unsat proved=True  repo=negated-goal",
    "  • Q-3  Soliton amplitude lower bound        status=unsat proved=True  repo=negated-goal",
    "  • All three goals proved:  True",
]


def test_task_2():
    _assert_exact(_capture(ch23_tasks.task_2), TASK2_EXPECTED)


# ----------------------------------------------------------------------
# Task 3
# ----------------------------------------------------------------------

TASK3_EXPECTED = [
    "Bounded-Domain Disclosure — what each certificate assumes and omits:",
    "  [TAC-1] Lyapunov free-energy dissipation",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  kappa > 0; grad F = (g0, g1, g2) in R^3",
    "      proves:   dF/dt = -kappa * ||grad F||^2 <= 0, with equality iff grad F = 0",
    "      omits:    the discretized Euler-Maruyama update that actually runs; floating-point rounding of the product -kappa * ||grad F||^2; gradient dimensions N > 3",
    "  [TAC-2] Dimension-independent relaxation",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  kappa > 0; lambda_min > 0; ||s||^2 > 0; dissipation >= 2 * kappa * lambda_min * ||s||^2",
    "      proves:   dissipation > 0; relaxation timescale governed only by lambda_min",
    "      omits:    A >= lambda_min * I is taken as a premise, not derived; finite-precision evaluation of the quadratic form",
    "  [TAC-3] Bistable double-well soft-spin",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  s in R; V(s) = 1/4 * (s^2 - 1)^2",
    "      proves:   stationary points exactly {-1, 0, +1}; +/-1 stable, 0 unstable; confinement for |s| > 1",
    "      omits:    Langevin noise (the stochastic term of the relaxer); multi-dimensional soft-spin fields",
    "  [TAC-4] Exact-cover ground state",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  beta > 0; A = [[1], [1]] (a 2-set, 1-element micro-instance)",
    "      proves:   F(s) = 0 iff (s0, s1) in {(1, 0), (0, 1)}",
    "      omits:    general set systems: only the 2x1 instance is encoded; floating-point evaluation of F",
    "  [TAC-5] S5 representation invariant (137)",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  |S5| = 5! = 120; irrep dimensions [1, 1, 4, 4, 5, 5, 6]",
    "      proves:   sum(d_i^2) = 120 and 120 + 4^2 + 1 = 137",
    "      omits:    that the listed integers are the actual irreps (taken from the classification); meaning for the floating-point Cayley projector",
    "  [Q-1] Discrete geodesic contraction",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  W_t >= delta_w; delta_w > 0; eps > 0",
    "      proves:   1/(W_t + delta_w + eps) < 1/(W_t + eps): the edge distance strictly contracts",
    "      omits:    floating-point rounding: in FP32 a small delta_w is absorbed by W_t; the graph-level path distance (this certificate is about one edge)",
    "  [Q-2] Quantized Fiedler preservation",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  N >= 4; delta_w > 0; lambda_2_min == N * delta_w",
    "      proves:   lambda_2_min > 0: the spectral gap never collapses",
    "      omits:    the graph-theoretic derivation lambda_2(L) >= N * min W_ij (taken here as the definition); the FP32/FP64 Laplacian eigensolve",
    "  [Q-3] Soliton amplitude lower bound",
    "      domain:   real arithmetic (Z3 Real/Int), exact",
    "      assumes:  alpha_0 >= 2; 1 <= d_avg <= 4; alpha_0^gamma >= alpha_0^2; alpha_0^gamma > d_avg",
    "      proves:   rho = d_avg / alpha_0^gamma < 1: the distractor ratio contracts",
    "      omits:    that the 32-hop benchmark actually satisfies these premises; the induction over hops: only the per-hop ratio is bounded",
    "",
    "  Disclosure summary: 8 certificates over the real domain; 0 cover floating-point rounding.",
]


def test_task_3():
    _assert_exact(_capture(ch23_tasks.task_3), TASK3_EXPECTED)


# ----------------------------------------------------------------------
# Task 4
# ----------------------------------------------------------------------

TASK4_EXPECTED = [
    "Property-Based Check of Q-1 (discrete geodesic contraction):",
    "  Ensembles:                 200 random graphs × 64 edges = 12800 edges",
    "  Domain:                    W_t >= delta_w > 0, eps > 0",
    "  FP32 contractions held:    12800/12800  (100.00%)",
    "  FP64 contractions held:    12800/12800  (100.00%)",
    "  FP32 == FP64 on all edges: True",
    "  Bounded-domain probe (delta_w = 2^-25 * W, W >= 1):",
    "    FP32 contractions held:  0/50",
    "    FP64 contractions held:  50/50",
    "    FP32 == FP64:            False",
]


def test_task_4():
    _assert_exact(_capture(ch23_tasks.task_4), TASK4_EXPECTED)


# ----------------------------------------------------------------------
# Task 5
# ----------------------------------------------------------------------

TASK5_EXPECTED = [
    "Property-Based Check of Q-2 (Fiedler floor lambda_2 >= N * delta_w):",
    "  Ensembles:                 50 random dense lattices of N = 12 nodes",
    "  Domain:                    every off-diagonal symmetrized conductance >= delta_w > 0",
    "  FP32 bound held:           50/50",
    "  FP64 bound held:           50/50",
    "  FP32 == FP64 on all lattices: True",
    "  Minimum margin (lambda_2 - N*delta_w): 36.660",
]


def test_task_5():
    _assert_exact(_capture(ch23_tasks.task_5), TASK5_EXPECTED)


# ----------------------------------------------------------------------
# Task 6
# ----------------------------------------------------------------------

TASK6_EXPECTED = [
    "Z3 + Property-Based Verification Certificate (canonical JSON):",
    "{",
    '  "all_goals_proved": true,',
    '  "chapter": "23",',
    '  "disclosure": {',
    '    "certificates": 8,',
    '    "cover_floating_point": 0,',
    '    "over_reals": 8',
    "  },",
    '  "proofs": {',
    '    "Q-1": "unsat",',
    '    "Q-2": "unsat",',
    '    "Q-3": "unsat",',
    '    "TAC-1": "unsat",',
    '    "TAC-2": "unsat",',
    '    "TAC-3": "unsat",',
    '    "TAC-4": "unsat",',
    '    "TAC-5": "unsat"',
    "  },",
    '  "property_based": {',
    '    "q1": {',
    '      "agreement": true,',
    '      "boundary_fp32_passed": 0,',
    '      "boundary_fp64_passed": 50,',
    '      "boundary_total": 50,',
    '      "edges": 12800,',
    '      "edges_per_graph": 64,',
    '      "fp32_passed": 12800,',
    '      "fp64_passed": 12800,',
    '      "graphs": 200',
    "    },",
    '    "q2": {',
    '      "agreement": true,',
    '      "fp32_passed": 50,',
    '      "fp64_passed": 50,',
    '      "lattices": 50,',
    '      "min_margin": 36.66,',
    '      "nodes": 12',
    "    }",
    "  },",
    '  "repo_sat_witnesses": [',
    '    "TAC-5"',
    "  ],",
    '  "sha256": "eb730262d0cf0d0d4f500f207a9536396eda32591605787300a9b44027f63c1e",',
    '  "title": "What the Z3 Certificates Do and Don\'t Say",',
    '  "z3_version": "5.1.0"',
    "}",
    "",
    "  SHA-256 of the canonical payload (this field excluded): eb730262d0cf0d0d4f500f207a9536396eda32591605787300a9b44027f63c1e",
]


def test_task_6():
    _assert_exact(_capture(ch23_tasks.task_6), TASK6_EXPECTED)
