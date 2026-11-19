# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch22_claims.py — Unit tests for the Chapter 22 claims ledger.
"""

from __future__ import annotations

import os

from gftac.ch22_claims import (
    BOOK_ROOT,
    CLAIMS,
    VALID_LABELS,
    audit_claims,
    claims_sha256,
    count_by_label,
    render_claims_md,
)


def test_registry_is_nonempty():
    assert len(CLAIMS) >= 20


def test_every_claim_carries_a_valid_label():
    for claim in CLAIMS:
        assert claim.label in VALID_LABELS, claim.cid


def test_claim_ids_are_unique():
    ids = [c.cid for c in CLAIMS]
    assert len(ids) == len(set(ids))


def test_every_cited_artifact_exists():
    for claim in CLAIMS:
        assert os.path.isfile(os.path.join(BOOK_ROOT, claim.artifact)), claim.cid


def test_every_cited_test_exists():
    for claim in CLAIMS:
        assert os.path.isfile(os.path.join(BOOK_ROOT, claim.test)), claim.cid


def test_every_pinned_value_is_asserted_in_its_test():
    for claim in CLAIMS:
        if claim.pinned is not None:
            text = open(os.path.join(BOOK_ROOT, claim.test), encoding="utf-8").read()
            assert claim.pinned in text, (claim.cid, claim.pinned)


def test_full_audit_is_green():
    results = audit_claims()
    failures = [r for r in results if not r[1]]
    assert failures == []
    assert len(results) == len(CLAIMS)


def test_count_by_label_sums_to_registry_size():
    counts = count_by_label()
    assert sum(counts.values()) == len(CLAIMS)
    assert set(counts) == set(VALID_LABELS)


def test_render_contains_one_row_per_claim():
    text = render_claims_md()
    rows = [line for line in text.splitlines() if line.startswith("| CL-")]
    assert len(rows) == len(CLAIMS)
    assert "MEASURED" in text and "MODELED" in text and "RESEARCH DEMO" in text


def test_ledger_hash_is_stable():
    assert claims_sha256() == claims_sha256()
    assert len(claims_sha256()) == 64
