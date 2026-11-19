# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 19 — runnable tasks (RESEARCH DEMO).

Each task is a self-contained experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch19_tasks.py 1        # a single task
    python3 code/gftac/ch19_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch19_tasks.py.

Everything printed here is the output of a graph propagation over a small
hand-built ontology.  It is a **RESEARCH DEMO**, not medical advice.
"""

from __future__ import annotations

import os
import sys
import time
import torch

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

_SDK = os.path.join(os.path.dirname(_CODE), "geoformer", "src")
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

from gftac.ch19_clinical_kg import (
    ClinicalDiagnosticEngine,
    build_clinical_knowledge_graph,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]

# The canonical STEMI-versus-dissection emergency, used by tasks 3, 4 and 6.
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


def task_1() -> None:
    """§19.3 — Census of the clinical knowledge graph."""
    ont = build_clinical_knowledge_graph()
    census = ont.census()

    print("Clinical Knowledge Graph Census — RESEARCH DEMO:")
    print(f"  • Entities:                       {census['num_entities']}")
    print(f"  • Relation Types:                 {census['num_relations']}")
    print(f"  • Directed Edges:                 {census['num_edges']}")
    print("  Entities by Type:")
    for etype, count in sorted(
        census["entities_by_type"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"      {etype:<24} {count}")
    print("  Edges by Relation:")
    for rel, count in census["edges_by_relation"].items():
        print(f"      {rel:<24} {count}")


def task_2() -> None:
    """§19.4 — Compile the ontology into sparse hippocampal memory."""
    ont = build_clinical_knowledge_graph()
    engine = ClinicalDiagnosticEngine(ontology=ont)
    memory = engine.memory

    forward = [r for r in engine.compiled_relations if r < 100]
    reverse = [r for r in engine.compiled_relations if r >= 100]
    nnz = sum(t._nnz() for t in memory.sparse_edges.values())
    sparse_bytes = memory.memory_footprint_bytes()
    dense_bytes = memory.dense_footprint_bytes()
    compression = dense_bytes / max(1, sparse_bytes)

    print("Compiling the Clinical Ontology into Sparse Hippocampal Memory — RESEARCH DEMO:")
    print(f"  • Entities:                       {engine.num_entities}")
    print(f"  • Forward Relations:              {len(forward)}")
    print(f"  • Reverse Relations:              {len(reverse)}")
    print(f"  • Stored Nonzeros:                {nnz}")
    print(f"  • Sparse Physical Footprint:      {sparse_bytes:,} bytes "
          f"({sparse_bytes / (1024 ** 2):.4f} MB)")
    print(f"  • Dense Equivalent:               {dense_bytes:,} bytes "
          f"({dense_bytes / (1024 ** 2):.2f} MB)")
    print(f"  • Compression Factor:             {compression:.1f}x")
    print(f"  • Invariant Battery Audit:        "
          f"{'PASSED' if memory.assert_invariants() else 'FAILED'}")


def task_3() -> None:
    """§19.5 — Differential diagnosis on a canonical STEMI case."""
    engine = ClinicalDiagnosticEngine()
    result = engine.diagnose_patient(STEMI_CASE, top_k=3)

    print("Differential Diagnosis — Canonical STEMI vs Aortic Dissection (RESEARCH DEMO):")
    print(f"  Observed findings ({len(STEMI_CASE)}): {', '.join(STEMI_CASE)}")
    print("  Ranked differentials:")
    for rank, entry in enumerate(result["differentials"], start=1):
        print(f"    {rank}. {entry['disease']:<26} {entry['probability']:6.2f}%  "
              f"(raw score {entry['confidence_score']:.6f})")
    print(f"  Primary diagnosis:   {result['primary_diagnosis']}")
    chain = result["pathophysiological_causal_chain"][0]
    print(f"  Causal chain:        {chain}")
    print(f"  Recommended (DEMO):  {', '.join(result['recommended_interventions'])}")


def task_4() -> None:
    """§19.6 — Contraindication alert firing and clearing."""
    engine = ClinicalDiagnosticEngine()

    dissection = engine.diagnose_patient(DISSECTION_CASE, top_k=1)
    stemi = engine.diagnose_patient(STEMI_CASE, top_k=1)

    dissection_alerts = dissection["safety_contraindications"]
    stemi_alerts = stemi["safety_contraindications"]

    print("Contraindication Safety Check — RESEARCH DEMO:")
    print(f"  Case A (dissection findings) -> primary {dissection['primary_diagnosis']}")
    print(f"    Alert fired:   {bool(dissection_alerts)}")
    print(f"    Alert:         {dissection_alerts[0] if dissection_alerts else 'none'}")
    print(f"  Case B (STEMI findings)      -> primary {stemi['primary_diagnosis']}")
    print(f"    Alert fired:   {bool(stemi_alerts)}")
    print(f"    Alert:         {stemi_alerts[0] if stemi_alerts else 'none'}")


def task_5() -> None:
    """§19.7 — Multi-hop causal path tracing."""
    ont = build_clinical_knowledge_graph()
    engine = ClinicalDiagnosticEngine(ontology=ont)

    chains = [
        (name, engine.trace_causal_path(name))
        for name in ont.entities_of_type("disease")
    ]
    chains.sort(key=lambda item: (-(len(item[1]) - 1), item[0]))

    print("Multi-Hop Pathophysiological Causal Trace — RESEARCH DEMO:")
    for name, chain in chains:
        depth = len(chain) - 1
        print(f"  {name:<28} depth {depth}  | {' -> '.join(chain)}")
    longest_name, longest_chain = chains[0]
    print(f"  Longest causal chain: depth {len(longest_chain) - 1} ({longest_name})")


def task_6() -> None:
    """§19.8 — SDK parity audit vs GeoFormerClinicalDiagnosticEngine."""
    from geoformer.clinical.engine import GeoFormerClinicalDiagnosticEngine

    book = ClinicalDiagnosticEngine()
    sdk = GeoFormerClinicalDiagnosticEngine()

    book_case1 = book.diagnose_patient(STEMI_CASE, top_k=3)
    sdk_case1 = sdk.diagnose_patient(STEMI_CASE, top_k=3)
    book_rank = [d["disease"] for d in book_case1["differentials"]]
    sdk_rank = [d["disease"] for d in sdk_case1["differentials"]]
    ranking_match = book_rank == sdk_rank

    book_case2 = book.diagnose_patient(DISSECTION_CASE, top_k=1)
    sdk_case2 = sdk.diagnose_patient(DISSECTION_CASE, top_k=1)
    book_alert = book_case2["safety_contraindications"]
    sdk_alert = sdk_case2["safety_contraindications"]
    alert_match = book_alert == sdk_alert

    max_diff = 0.0
    for b, s in zip(book_case1["differentials"], sdk_case1["differentials"]):
        max_diff = max(max_diff, abs(b["probability"] - s["probability"]))

    print("SDK Parity Audit — Book Engine vs GeoFormerClinicalDiagnosticEngine (RESEARCH DEMO):")
    print(f"  Case 1 ranking (STEMI vs dissection): {book_rank}")
    print(f"  SDK    ranking (STEMI vs dissection): {sdk_rank}")
    print(f"  Rankings match:                       {ranking_match}")
    print(f"  Case 2 contraindication alert match:  {alert_match}")
    print(f"  Max differential probability diff:    {max_diff:.2e}")
    print(f"  Full Parity Verified:                 "
          f"{ranking_match and alert_match and max_diff < 1e-9}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        print("Running all Chapter 19 tasks:\n")
        task_1()
        print("\n" + "-" * 60 + "\n")
        task_2()
        print("\n" + "-" * 60 + "\n")
        task_3()
        print("\n" + "-" * 60 + "\n")
        task_4()
        print("\n" + "-" * 60 + "\n")
        task_5()
        print("\n" + "-" * 60 + "\n")
        task_6()
    else:
        task_num = sys.argv[1]
        fn = globals().get(f"task_{task_num}")
        if fn is None:
            print(f"Unknown task: {task_num}")
            sys.exit(1)
        fn()


if __name__ == "__main__":
    main()
