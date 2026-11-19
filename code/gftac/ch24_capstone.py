# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 24 — Capstone: The Whole Stack in One Pipeline.

Twenty-three chapters built twenty-three components. This module runs them as
one machine. The capstone is not new theory; it is composition. Each stage of
the pipeline is a component from an earlier chapter, called exactly as that
chapter's tests call it, and each stage reports a small set of deterministic
metrics. The metrics are folded into a single SHA-256 certificate, so the whole
stack has one reproducible digest.

The stages, in order:

  1. Metric lattice      (Chapter 2)  — a quantized conductance world.
  2. Hebbian plasticity  (Chapter 10) — writes that contract resistance.
  3. Model forward       (Chapter 9)  — a GeoFormer block stack, counted.
  4. Soliton propagation (Chapter 16) — a certified multi-hop chain.
  5. Packed export       (Chapter 4)  — trits to five-per-byte.
  6. Code-security transfer (Chapter 18) — RESEARCH DEMO, not a claim.

Stage 6 is the book's one deliberate projection. The same sparse relational
machinery that propagates state across a clinical ontology propagates *taint*
across a code property graph: a source of untrusted data, propagator nodes, a
sanitizer that absorbs, and sink nodes where taint is a vulnerability. The
demonstration is a toy — eight nodes, two programs — and it is labeled RESEARCH
DEMO everywhere it appears. Its purpose is to show that the paradigm transfers,
not to detect real vulnerabilities.

Run the capstone ledger::

    python3 code/gftac/ch24_capstone.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch02_metric_lattice import MetricLattice
from gftac.ch04_ternary import pack_trits, packed_bits_per_parameter, packed_bytes
from gftac.ch09_model import BookGeoFormer
from gftac.ch10_plasticity import PlasticConductanceEngine, PlasticityConfig
from gftac.ch16_soliton import simulate_multihop_soliton_chain
from gftac.ch18_sparse_memory import SparseHippocampalMemory, SparseMemoryConfig

__all__ = [
    "StageResult",
    "stage_1_lattice",
    "stage_2_plasticity",
    "stage_3_model",
    "stage_4_soliton",
    "stage_5_export",
    "stage_6_code_security",
    "run_capstone",
    "certificate_sha256",
    "main",
]


@dataclass
class StageResult:
    """One stage's name and its deterministic metrics."""

    name: str
    metrics: Dict[str, object] = field(default_factory=dict)


def _fmt(v: object) -> str:
    """Stable string form for hashing and printing."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.12g}"
    if isinstance(v, (tuple, list)):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    return str(v)


# ----------------------------------------------------------------------
# Stage 1 — the metric lattice (Chapter 2)
# ----------------------------------------------------------------------

def stage_1_lattice(n: int = 32) -> StageResult:
    """Build a quantized chain world and read its Fiedler value."""
    lat = MetricLattice(num_entities=n, bits=4, w_max=1.0, seed=0)
    for i in range(n - 1):
        lat.observe_edge(i, i + 1, weight=1.0)
    edges = int((lat.W > 0).sum().item())
    return StageResult(
        "lattice",
        {
            "nodes": n,
            "edges": edges,
            "fiedler_lambda2": lat.fiedler_value(),
            "quantized": lat.is_quantized(),
        },
    )


# ----------------------------------------------------------------------
# Stage 2 — Hebbian plasticity (Chapter 10)
# ----------------------------------------------------------------------

def stage_2_plasticity(n: int = 32) -> StageResult:
    """Reinforce every edge of the chain; watch resistance contract."""
    eng = PlasticConductanceEngine(PlasticityConfig(num_entities=n, seed=42))
    d_before = eng.resistance(0, 1)
    updated = eng.update_conductance(
        list(range(n - 1)), list(range(1, n)), weight_inc=8.0
    )
    d_after = eng.resistance(0, 1)
    return StageResult(
        "plasticity",
        {
            "edges_updated": updated,
            "resistance_before": d_before,
            "resistance_after": d_after,
            "contracted": d_after < d_before,
            "invariants": eng.assert_invariants(),
        },
    )


# ----------------------------------------------------------------------
# Stage 3 — model forward (Chapter 9)
# ----------------------------------------------------------------------

def stage_3_model(d_model: int = 32) -> StageResult:
    """Instantiate a two-block GeoFormer, run a batch, count the parameters."""
    torch.manual_seed(0)
    model = BookGeoFormer(
        in_features=d_model,
        num_classes=4,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
    )
    x = torch.randn(3, d_model)
    with torch.no_grad():
        logits = model(x)
    params, packed_kb = model.count_parameters()
    return StageResult(
        "model",
        {
            "params": params,
            "packed_kb": packed_kb,
            "batch": 3,
            "num_classes": 4,
            "logits_sum": float(logits.sum().item()),
        },
    )


# ----------------------------------------------------------------------
# Stage 4 — soliton propagation (Chapter 16)
# ----------------------------------------------------------------------

def stage_4_soliton(hops: int = 32) -> StageResult:
    """Run the certified multi-hop soliton chain at depth 32."""
    res = simulate_multihop_soliton_chain(
        num_hops=hops, alpha_0=2.5, gamma=2.5, d_avg=2.0
    )
    return StageResult(
        "soliton",
        {
            "hops": hops,
            "rho": res["rho"],
            "gamma_crit": res["gamma_crit"],
            "supercritical": res["is_supercritical"],
            "final_peak_mass": res["final_peak_mass"],
            "satisfies_bound": res["satisfies_bound"],
        },
    )


# ----------------------------------------------------------------------
# Stage 5 — packed export (Chapter 4)
# ----------------------------------------------------------------------

def stage_5_export(num_trits: int = 1024) -> StageResult:
    """Pack a deterministic trit vector five to a byte."""
    trits = [(i % 3) - 1 for i in range(num_trits)]
    packed = pack_trits(trits)
    return StageResult(
        "export",
        {
            "trits": num_trits,
            "packed_bytes": int(packed.numel()),
            "theoretical_bytes": packed_bytes(num_trits),
            "bits_per_parameter": packed_bits_per_parameter(),
        },
    )


# ----------------------------------------------------------------------
# Stage 6 — code-security transfer (Chapter 18) — RESEARCH DEMO
# ----------------------------------------------------------------------

# A tiny code property graph. Node 0 is the untrusted source; nodes 4 and 6 are
# sinks (a database query and a format string); node 5 is a sanitizer that
# absorbs taint because it has no outgoing data-flow edge.
_CPG_NODES = 8
_CPG_SINKS = (4, 6)
_CPG_VULNERABLE = ((0, 1), (1, 3), (3, 4), (0, 6))   # taint reaches 4 and 6
_CPG_SAFE = ((0, 5),)                                 # sanitizer absorbs it


def _cpg_memory(num_entities: int, edges: Sequence[Tuple[int, int]]):
    mem = SparseHippocampalMemory(
        SparseMemoryConfig(num_entities=num_entities, d_model=8, mode="fp32")
    )
    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        tgt = torch.tensor([e[1] for e in edges], dtype=torch.long)
        # edge_index rows are (target, source).
        mem.register_sparse_relation(0, torch.stack([tgt, src]))
    return mem


def _taint_reaches(edges: Sequence[Tuple[int, int]], hops: int = 8) -> List[int]:
    """Sinks that ever receive taint mass over ``hops`` of propagation."""
    mem = _cpg_memory(_CPG_NODES, edges)
    h = torch.zeros(_CPG_NODES, dtype=torch.float32)
    h[0] = 1.0
    reached: List[int] = []
    for _ in range(hops):
        h = mem.propagate_sparse_relation(0, h)
        for sink in _CPG_SINKS:
            if float(h[sink].item()) > 1e-6 and sink not in reached:
                reached.append(sink)
    return sorted(reached)


def stage_6_code_security() -> StageResult:
    """Propagate taint across a toy code property graph (RESEARCH DEMO)."""
    vuln = _taint_reaches(_CPG_VULNERABLE)
    safe = _taint_reaches(_CPG_SAFE)
    return StageResult(
        "code_security",
        {
            "vulnerable_sinks_reached": len(vuln),
            "safe_sinks_reached": len(safe),
            "taint_reached_sink": len(vuln) > 0,
            "sanitizer_blocked_taint": len(safe) == 0,
        },
    )


# ----------------------------------------------------------------------
# The whole stack
# ----------------------------------------------------------------------

def run_capstone() -> List[StageResult]:
    """Run every stage in order and return the results."""
    return [
        stage_1_lattice(),
        stage_2_plasticity(),
        stage_3_model(),
        stage_4_soliton(),
        stage_5_export(),
        stage_6_code_security(),
    ]


def certificate_sha256(results: Sequence[StageResult]) -> str:
    """A single digest over every stage's metrics."""
    payload = "\n".join(
        f"{r.name}|"
        + ",".join(f"{k}={_fmt(v)}" for k, v in sorted(r.metrics.items()))
        for r in results
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _all_green(results: Sequence[StageResult]) -> bool:
    """Every stage's own pass-condition holds."""
    m = {r.name: r.metrics for r in results}
    return bool(
        m["lattice"]["quantized"]
        and m["plasticity"]["invariants"]
        and m["plasticity"]["contracted"]
        and m["soliton"]["satisfies_bound"]
        and m["code_security"]["taint_reached_sink"]
        and m["code_security"]["sanitizer_blocked_taint"]
    )


def main() -> None:
    results = run_capstone()
    for r in results:
        print(f"[{r.name}]")
        for k, v in sorted(r.metrics.items()):
            print(f"  {k} = {_fmt(v)}")
    print(f"[certificate] sha256 = {certificate_sha256(results)}")
    print(f"[certificate] all_green = {_all_green(results)}")


if __name__ == "__main__":
    main()
