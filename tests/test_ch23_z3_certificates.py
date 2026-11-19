# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch23_z3_certificates.py — Unit tests for Chapter 23.

Assertions:
  * every reproduced Z3 goal (TAC 1–5, GeoFormer Q-1/Q-2/Q-3) is proved, i.e.
    its negated goal returns ``z3.unsat``;
  * the shipped TAC-5 SAT witness is recorded as such and upgraded to UNSAT;
  * the property-based checks hold on the sampled ensembles, in FP32 and FP64,
    and the adversarial boundary probe exhibits the FP32 limitation;
  * the certificate JSON is well-formed and its SHA-256 is stable.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import z3

from gftac.ch23_z3_certificates import (
    PROPERTY_SEED,
    all_proof_records,
    bounded_domain_disclosure,
    build_certificate,
    certificate_sha256,
    property_check_q1,
    property_check_q2,
    run_geoformer_theorems,
    run_tac_theorems,
)


# ----------------------------------------------------------------------
# Reproduced certificates
# ----------------------------------------------------------------------

def test_tac_theorems_all_unsat():
    records = run_tac_theorems()
    assert [r.proof_id for r in records] == ["TAC-1", "TAC-2", "TAC-3", "TAC-4", "TAC-5"]
    for r in records:
        assert r.status == "unsat", f"{r.proof_id} not proved: {r.status}"
        assert r.proved is True


def test_geoformer_theorems_all_unsat():
    records = run_geoformer_theorems()
    assert [r.proof_id for r in records] == ["Q-1", "Q-2", "Q-3"]
    for r in records:
        assert r.status == "unsat"
        assert r.proved is True


def test_all_goals_proved():
    records = all_proof_records()
    assert len(records) == 8
    assert all(r.proved for r in records)
    assert all(r.status == "unsat" for r in records)


def test_tac5_shipped_as_sat_witness_and_upgraded():
    tac5 = [r for r in run_tac_theorems() if r.proof_id == "TAC-5"][0]
    # The shipped verifier asked for a model (SAT); our negated goal is UNSAT.
    assert tac5.repo_method == "SAT-witness"
    assert tac5.repo_status == "sat"
    assert tac5.status == "unsat"
    assert tac5.proved is True


def test_repo_tac_verifier_cross_check():
    """The repo's shipped TAC verifier agrees that all five theorems hold."""
    from tac.proofs.z3_verifier import TACZ3Verifier

    cert = TACZ3Verifier().run_all_proofs()
    assert cert["all_theorems_verified"] is True
    assert len(cert["theorems"]) == 5
    assert len(cert["sha256_fingerprint"]) == 64


def test_repo_geoformer_suite_cross_check():
    """Re-run the repo's actual Q-1/Q-2/Q-3 tests; they must pass green."""
    import importlib.util
    import os

    repo_test = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "geoformer", "tests", "test_z3_invariants.py",
    )
    if not os.path.exists(repo_test):
        pytest.skip("geoformer repo suite not present")

    spec = importlib.util.spec_from_file_location("_repo_test_z3_invariants", repo_test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Each of these is the repo's own negated-goal refutation; running them
    # re-executes the shipped encodings and asserts their UNSAT.
    module.test_z3_theorem_q1_discrete_geodesic_contraction()
    module.test_z3_theorem_q2_fiedler_preservation()
    module.test_z3_theorem_q3_soliton_amplitude_lower_bound()


# ----------------------------------------------------------------------
# Bounded-domain disclosure
# ----------------------------------------------------------------------

def test_disclosure_covers_all_and_none_is_float():
    records = bounded_domain_disclosure()
    assert len(records) == 8
    for r in records:
        assert "real" in r.domain
        assert r.premises and r.omits


def test_b23_2_fp32_absorbs_sub_half_spacing_increment():
    """Theorem B.23.2: FP32 rounds W + delta_w back to W for a sub-ulp increment."""
    W = np.float32(8.0)
    delta = np.float32(2.0 ** -25 * 8.0)
    eps = np.float32(1e-8)

    # The increment is strictly below the binary32 half-spacing at W.
    assert np.float32(W + delta) == W

    d_before = np.float32(1.0) / (W + eps)
    d_after = np.float32(1.0) / (W + delta + eps)
    assert not (d_after < d_before), "strict contraction must fail in FP32 here"

    # The same increment contracts cleanly in FP64.
    W64 = 8.0
    delta64 = 2.0 ** -25 * 8.0
    assert 1.0 / (W64 + delta64 + 1e-8) < 1.0 / (W64 + 1e-8)


# ----------------------------------------------------------------------
# Property-based checks
# ----------------------------------------------------------------------

def test_property_q1_holds_and_agrees():
    res = property_check_q1()
    assert res["total"] == res["num_graphs"] * res["edges_per_graph"]
    assert res["fp32_passed"] == res["total"]
    assert res["fp64_passed"] == res["total"]
    assert res["agreement"] is True


def test_property_q1_boundary_probe_shows_fp32_gap():
    res = property_check_q1()
    assert res["boundary_fp32_passed"] == 0
    assert res["boundary_fp64_passed"] == res["boundary_total"]
    assert res["boundary_agreement"] is False


def test_property_q1_is_deterministic():
    a = property_check_q1()
    b = property_check_q1()
    assert a == b


def test_property_q2_holds_and_agrees():
    res = property_check_q2()
    assert res["fp32_passed"] == res["num_lattices"]
    assert res["fp64_passed"] == res["num_lattices"]
    assert res["agreement"] is True
    assert res["min_margin"] > 0.0


def test_property_q2_is_deterministic():
    assert property_check_q2() == property_check_q2()


def test_seed_is_fixed():
    assert PROPERTY_SEED == 23


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------

def test_certificate_well_formed():
    cert = build_certificate()
    assert cert["chapter"] == "23"
    assert cert["z3_version"] == z3.get_version_string()
    assert cert["all_goals_proved"] is True
    assert set(cert["proofs"]) == {"TAC-1", "TAC-2", "TAC-3", "TAC-4", "TAC-5", "Q-1", "Q-2", "Q-3"}
    assert all(v == "unsat" for v in cert["proofs"].values())
    assert cert["repo_sat_witnesses"] == ["TAC-5"]
    assert cert["disclosure"] == {"certificates": 8, "over_reals": 8, "cover_floating_point": 0}
    # JSON round-trips.
    assert json.loads(json.dumps(cert)) == cert


def test_certificate_hash_is_stable_and_verifiable():
    cert = build_certificate()
    digest = cert["sha256"]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)

    # Recomputing from the payload (with the hash field removed) matches.
    assert certificate_sha256(cert) == digest

    # Two independent assemblies agree byte for byte.
    assert build_certificate()["sha256"] == digest


def test_certificate_hash_changes_with_payload():
    cert = build_certificate()
    tampered = dict(cert)
    tampered["all_goals_proved"] = False
    assert certificate_sha256(tampered) != cert["sha256"]
