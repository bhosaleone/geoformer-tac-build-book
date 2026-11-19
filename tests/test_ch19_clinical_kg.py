# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch19_clinical_kg.py — Unit tests for Chapter 19 clinical KG engine.

Everything asserted here is the behaviour of a small graph-propagation demo
(RESEARCH DEMO): no test asserts a clinical claim, only a structural or numeric
property of the compiled ontology and its sparse engine.
"""

from __future__ import annotations

import pytest
import torch

from gftac.ch19_clinical_kg import (
    ClinicalDiagnosticEngine,
    MedicalOntology,
    RELATION_IDS,
    build_clinical_knowledge_graph,
)

STEMI_CASE = [
    "Retrosternal_Crushing_Chest_Pain",
    "Diaphoresis",
    "ST_Segment_Elevation",
    "Tearing_Interscapular_Back_Pain",
]
DISSECTION_CASE = [
    "Tearing_Interscapular_Back_Pain",
    "Pulse_Deficit_Between_Arms",
    "Mediastinal_Widening_on_CXR",
]


def test_ontology_entity_and_relation_counts():
    ont = build_clinical_knowledge_graph()
    assert ont.num_entities == 128
    assert ont.num_relations == 6
    assert set(ont.relation2id) == set(RELATION_IDS)


def test_ontology_edges_per_relation():
    ont = build_clinical_knowledge_graph()
    counts = {ont.id2relation[r]: len(e) for r, e in ont.edges_by_rel.items()}
    assert counts == {
        "presents_with": 42,
        "caused_by": 25,
        "indicated_biomarker": 24,
        "treated_by": 25,
        "contraindicated_in": 1,
        "complication_of": 1,
    }
    assert sum(counts.values()) == 118


def test_ontology_entity_types():
    ont = build_clinical_knowledge_graph()
    census = ont.census()
    assert census["entities_by_type"] == {
        "disease": 12,
        "symptom": 42,
        "biomarker": 24,
        "pathophysiology": 24,
        "drug": 26,
    }


def test_entity_ids_are_insertion_ordered():
    ont = build_clinical_knowledge_graph()
    # The first entity ever added is the head of the first triple.
    assert ont.id2entity[0] == "Acute_STEMI"
    assert ont.entity2id["Acute_STEMI"] == 0


def test_engine_compiles_forward_and_reverse_relations():
    engine = ClinicalDiagnosticEngine()
    compiled = engine.compiled_relations
    assert compiled == [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105]
    assert engine.memory.assert_invariants() is True


def test_engine_sparse_footprint_matches_law():
    engine = ClinicalDiagnosticEngine()
    nnz = sum(t._nnz() for t in engine.memory.sparse_edges.values())
    # 118 forward edges + 118 reverse edges, all distinct coordinates.
    assert nnz == 236
    assert engine.memory.memory_footprint_bytes() == 236 * 20


def test_stemi_case_ranks_acute_stemi_first():
    engine = ClinicalDiagnosticEngine()
    result = engine.diagnose_patient(STEMI_CASE, top_k=3)
    assert result["primary_diagnosis"] == "Acute_STEMI"
    assert result["differentials"][0]["disease"] == "Acute_STEMI"
    assert result["differentials"][0]["probability"] > 70.0
    assert result["differentials"][1]["disease"] == "Aortic_Dissection"


def test_contraindication_alert_fires_on_dissection():
    engine = ClinicalDiagnosticEngine()
    alerts = engine.check_contraindications("Aortic_Dissection")
    assert len(alerts) == 1
    assert alerts[0] == (
        "CRITICAL WARNING: Thrombolysis_tPA is strictly CONTRAINDICATED "
        "in Aortic_Dissection!"
    )


def test_contraindication_clears_on_stemi():
    engine = ClinicalDiagnosticEngine()
    assert engine.check_contraindications("Acute_STEMI") == []


def test_dissection_case_triggers_alert_end_to_end():
    engine = ClinicalDiagnosticEngine()
    result = engine.diagnose_patient(DISSECTION_CASE, top_k=1)
    assert result["primary_diagnosis"] == "Aortic_Dissection"
    assert len(result["safety_contraindications"]) == 1
    assert "Thrombolysis_tPA" in result["safety_contraindications"][0]


def test_causal_chain_depth_at_least_three():
    engine = ClinicalDiagnosticEngine()
    chain = engine.trace_causal_path("Acute_STEMI")
    assert chain[0] == "Acute_STEMI"
    assert len(chain) - 1 >= 3
    assert chain[-1] == "Myocyte_Coagulative_Necrosis"


def test_causal_chain_terminates_on_disease_without_mechanism():
    engine = ClinicalDiagnosticEngine()
    assert engine.trace_causal_path("Deep_Vein_Thrombosis") == ["Deep_Vein_Thrombosis"]


def test_recommended_interventions_are_treatments():
    engine = ClinicalDiagnosticEngine()
    treatments = engine.recommended_interventions("Acute_STEMI")
    assert treatments == [
        "Primary_Percutaneous_Coronary_Intervention",
        "Aspirin_Dual_Antiplatelet",
        "Unfractionated_Heparin",
        "High_Intensity_Atorvastatin",
    ]


def test_unknown_findings_reported():
    engine = ClinicalDiagnosticEngine()
    result = engine.diagnose_patient(["Not_A_Real_Finding"], top_k=1)
    assert "error" in result
    assert result["unknown"] == ["Not_A_Real_Finding"]


def test_sdk_parity_ranking_and_alert():
    from geoformer.clinical.engine import GeoFormerClinicalDiagnosticEngine

    book = ClinicalDiagnosticEngine()
    sdk = GeoFormerClinicalDiagnosticEngine()

    book_stemi = book.diagnose_patient(STEMI_CASE, top_k=3)
    sdk_stemi = sdk.diagnose_patient(STEMI_CASE, top_k=3)
    assert [d["disease"] for d in book_stemi["differentials"]] == [
        d["disease"] for d in sdk_stemi["differentials"]
    ]
    for b, s in zip(book_stemi["differentials"], sdk_stemi["differentials"]):
        assert abs(b["probability"] - s["probability"]) < 1e-9

    book_dis = book.diagnose_patient(DISSECTION_CASE, top_k=1)
    sdk_dis = sdk.diagnose_patient(DISSECTION_CASE, top_k=1)
    assert book_dis["safety_contraindications"] == sdk_dis["safety_contraindications"]


def test_sdk_ontology_graph_parity():
    from geoformer.clinical.ontology import (
        build_clinical_knowledge_graph as sdk_build,
    )

    book = build_clinical_knowledge_graph()
    sdk = sdk_build()
    assert book.entity2id == sdk.entity2id
    assert book.entity_type == sdk.entity_type
    assert book.edges_by_rel == sdk.edges_by_rel


def test_determinism_of_diagnosis():
    a = ClinicalDiagnosticEngine().diagnose_patient(STEMI_CASE, top_k=3)
    b = ClinicalDiagnosticEngine().diagnose_patient(STEMI_CASE, top_k=3)
    assert a["differentials"] == b["differentials"]


def test_reverse_store_support_is_transpose():
    """Theorem B.19.1: the abductive store mirrors the deductive support."""
    engine = ClinicalDiagnosticEngine()
    for r in range(6):
        fwd = engine.memory.sparse_edges[r].indices()
        rev = engine.memory.sparse_edges[100 + r].indices()
        fwd_pairs = set(zip(fwd[0].tolist(), fwd[1].tolist()))
        rev_pairs = set(zip(rev[0].tolist(), rev[1].tolist()))
        assert rev_pairs == {(b, a) for (a, b) in fwd_pairs}


def test_caused_by_is_functional_and_acyclic():
    """Theorem B.19.4 hypothesis: out-degree <= 1 and no directed cycle."""
    ont = build_clinical_knowledge_graph()
    rel = ont.relation2id["caused_by"]
    successor = {}
    for head, tail in ont.edges_by_rel[rel]:
        assert head not in successor  # out-degree at most one
        successor[head] = tail
    for start in successor:
        seen = set()
        cur = start
        while cur in successor:
            assert cur not in seen  # a cycle would repeat a vertex
            seen.add(cur)
            cur = successor[cur]


def test_differential_probabilities_sum_below_100():
    """Theorem B.19.2: entries are positive, sorted, and sum short of 100."""
    engine = ClinicalDiagnosticEngine()
    result = engine.diagnose_patient(STEMI_CASE, top_k=10)
    probs = [d["probability"] for d in result["differentials"]]
    assert all(0.0 < p < 100.0 for p in probs)
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) < 100.0
    assert 100.0 - sum(probs) < 1e-4

