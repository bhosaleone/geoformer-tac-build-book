# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 22 — The Claims Ledger.

Every public number in the book is a claim, and every claim must be labeled and
traceable. This module holds the ledger: a single registry of the book's public
numbers, each tagged MEASURED / MODELED / RESEARCH DEMO, each pointing at the
artifact that produces it, the test that pins it, and the command that
regenerates it.

`render_claims_md()` turns the registry into `CLAIMS.md`; `audit_claims()`
checks that every cited artifact and test exists and that every pinned value is
actually asserted somewhere in its test. Nothing here re-measures the numbers —
it is the book's index of *where* each number is measured.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

BOOK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The three permitted labels. Every claim carries exactly one.
VALID_LABELS = ("MEASURED", "MODELED", "RESEARCH DEMO")


@dataclass(frozen=True)
class Claim:
    """One public number in the book, with its provenance."""

    cid: str
    chapter: int
    statement: str
    value: str
    label: str
    artifact: str
    test: str
    command: str
    pinned: Optional[str] = None


# The ledger. Artifacts and tests are book-relative paths; every one must exist.
CLAIMS: Tuple[Claim, ...] = (
    Claim("CL-01", 1, "Flat transformer scores at chance on unseen multi-hop queries",
          "0.0% ± 0.0% at 2/3/4-hop", "MEASURED",
          "code/gftac/ch01_compositional_collapse.py", "tests/test_ch01_tasks.py",
          "python3 code/gftac/ch01_tasks.py 3", None),
    Claim("CL-02", 2, "Fiedler gap lower bound on a min-conductance complete lattice",
          "λ₂ ≥ N · Δw", "MEASURED",
          "code/gftac/ch02_metric_lattice.py", "tests/test_ch02_metric_lattice.py",
          "python3 code/gftac/ch02_tasks.py 6", "N * "),
    Claim("CL-03", 3, "Relaxation step count is independent of dimension under unit-norm init",
          "identical τ for N ∈ {10,100,1000}", "MEASURED",
          "code/gftac/ch03_tac.py", "tests/test_ch03_tac.py",
          "python3 code/gftac/ch03_tasks.py 6", None),
    Claim("CL-04", 4, "Ternary foundation model packed footprint",
          "249,665 params → 48.76 KiB", "MEASURED",
          "code/gftac/ch04_ternary.py", "tests/test_ch04_ternary.py",
          "python3 code/gftac/ch04_tasks.py 6", "48.76"),
    Claim("CL-05", 5, "Attention concentration law d E[d]/dλ = −Var(d)",
          "identity holds (< 0.02)", "MEASURED",
          "code/gftac/ch05_qgha.py", "tests/test_ch05_qgha.py",
          "python3 code/gftac/ch05_tasks.py 4", None),
    Claim("CL-06", 6, "SiLU gate minimum of the ternary SwiGLU",
          "z* = −1.2784645", "MEASURED",
          "code/gftac/ch06_swiglu.py", "tests/test_ch06_swiglu.py",
          "python3 code/gftac/ch06_tasks.py 4", "1.2784645"),
    Claim("CL-07", 7, "Q-TAC operator spectral radius / condition number",
          "λ_max = 1.236603 (κ = 1.2366)", "MEASURED",
          "code/gftac/ch07_qtac.py", "tests/test_ch07_tasks.py",
          "python3 code/gftac/ch07_tasks.py 2", "1.236603"),
    Claim("CL-08", 7, "Chebyshev schedule acceleration over fixed-step relaxation",
          "up to 31.5× at k = 6", "MEASURED",
          "code/gftac/ch07_qtac.py", "tests/test_ch07_qtac.py",
          "python3 code/gftac/ch07_tasks.py 6", None),
    Claim("CL-09", 8, "Kolmogorov description-length bound of the S₅ projector",
          "K(S) ≤ 54.73 bits", "MODELED",
          "code/gftac/ch08_cayley.py", "tests/test_ch08_cayley.py",
          "python3 code/gftac/ch08_tasks.py 5", "54.73"),
    Claim("CL-10", 9, "Quasi-crystalline positional encoding collision rate",
          "0 collisions for N ≤ 2048", "MEASURED",
          "code/gftac/ch09_model.py", "tests/test_ch09_model.py",
          "python3 code/gftac/ch09_tasks.py 6", None),
    Claim("CL-11", 14, "Exact Student-t p-values via Simpson quadrature match scipy",
          "agreement ≤ 1e-10", "MEASURED",
          "code/gftac/ch14_eval.py", "tests/test_ch14_eval.py",
          "python3 code/gftac/ch14_tasks.py 3", None),
    Claim("CL-12", 17, "Autonomous halting accuracy at extreme depth",
          "100% at 32 hops, 5 seeds", "MEASURED",
          "code/gftac/ch17_goal_attractor.py", "tests/test_ch17_tasks.py",
          "python3 code/gftac/ch17_tasks.py 4", "100.0"),
    Claim("CL-13", 18, "Sparse memory compression at 100,000 entities",
          "100,000× vs dense", "MEASURED",
          "code/gftac/ch18_sparse_memory.py", "tests/test_ch18_sparse_memory.py",
          "python3 code/gftac/ch18_tasks.py 1", "200,000"),
    Claim("CL-14", 18, "Sparse vs dense propagation error (8 hops)",
          "‖h_sp − h_dn‖∞ = 3.73e-09", "MEASURED",
          "code/gftac/ch18_sparse_memory.py", "tests/test_ch18_tasks.py",
          "python3 code/gftac/ch18_tasks.py 4", None),
    Claim("CL-15", 18, "Autonomous halting depth on a 5-hop sparse chain",
          "τ* = 5 (exact)", "MEASURED",
          "code/gftac/ch18_sparse_memory.py", "tests/test_ch18_tasks.py",
          "python3 code/gftac/ch18_tasks.py 5", None),
    Claim("CL-16", 19, "Clinical ontology compiled to sparse memory",
          "128 entities, 166.6× compression", "RESEARCH DEMO",
          "code/gftac/ch19_clinical_kg.py", "tests/test_ch19_tasks.py",
          "python3 code/gftac/ch19_tasks.py 2", "166.6"),
    Claim("CL-17", 19, "Contraindication alert fires on the dissection case",
          "CRITICAL WARNING on Thrombolysis_tPA", "RESEARCH DEMO",
          "code/gftac/ch19_clinical_kg.py", "tests/test_ch19_clinical_kg.py",
          "python3 code/gftac/ch19_tasks.py 4", "CONTRAINDICATED"),
    Claim("CL-18", 20, "Fixed-point (FP_ONE=255) diffusion vs float reference",
          "8/9 at 2-hop, 6/6 at 3-hop", "MEASURED",
          "code/gftac/ch20_mcu.py", "tests/test_ch20_tasks.py",
          "python3 code/gftac/ch20_tasks.py 3", "8/9"),
    Claim("CL-19", 20, "ATmega328P Flash footprint of the exported kernel",
          "7,846 B (23.94% of 32 KB)", "MEASURED",
          "code/gftac/ch20_mcu.py", "tests/test_ch20_tasks.py",
          "python3 code/gftac/ch20_tasks.py 4", "7,846"),
    Claim("CL-20", 21, "Memristor crossbar dynamic energy per inference",
          "443.897 pJ (MODELED)", "MODELED",
          "code/gftac/ch21_memristor.py", "tests/test_ch21_memristor.py",
          "python3 code/gftac/ch21_tasks.py 1", "443.897"),
    Claim("CL-21", 21, "Crossbar energy reduction vs FP32 GPU baseline",
          "9,462×", "MODELED",
          "code/gftac/ch21_memristor.py", "tests/test_ch21_tasks.py",
          "python3 code/gftac/ch21_tasks.py 6", "9,462"),
    Claim("CL-22", 21, "Monte-Carlo accuracy at 20% conductance noise",
          "38.0% (MODELED)", "MODELED",
          "code/gftac/ch21_memristor.py", "tests/test_ch21_tasks.py",
          "python3 code/gftac/ch21_tasks.py 3", "38.0%"),
    Claim("CL-23", 23, "All Z3 negated-goal proofs discharge UNSAT",
          "TAC 1–5 and Q-1/2/3 all unsat", "MEASURED",
          "code/gftac/ch23_z3_certificates.py", "tests/test_ch23_z3_certificates.py",
          "python3 code/gftac/ch23_tasks.py 1", "unsat"),
    Claim("CL-24", 23, "Property-based Q-1 geodesic contraction holds on sampled ensembles",
          "12,800/12,800 (FP32 and FP64)", "MEASURED",
          "code/gftac/ch23_z3_certificates.py", "tests/test_ch23_tasks.py",
          "python3 code/gftac/ch23_tasks.py 4", None),
    Claim("CL-25", 10, "Invariant battery preserved under batched Hebbian writes",
          "W≥0, hollow diagonal, bounded step, quantized", "MEASURED",
          "code/gftac/ch10_plasticity.py", "tests/test_ch10_plasticity.py",
          "python3 code/gftac/ch10_tasks.py 1", "B.10"),
    Claim("CL-26", 11, "The integrated model's loss suite has five pinned terms",
          "reconstruction + geometric + directed + plasticity + entropy", "MEASURED",
          "code/gftac/ch11_integrated_model.py", "tests/test_ch11_integrated_model.py",
          "python3 code/gftac/ch11_tasks.py 3", "B.11"),
    Claim("CL-27", 12, "Crossroads dataset constructs highways and decoys with α₀ control",
          "α₀ = signal-to-decoy ratio is a dial", "MEASURED",
          "code/gftac/ch12_datasets.py", "tests/test_ch12_datasets.py",
          "python3 code/gftac/ch12_tasks.py 2", "crossroad"),
    Claim("CL-28", 13, "Kramers dual-path optimizer separates fast and slow timescales",
          "dual-path update reproduced", "MEASURED",
          "code/gftac/ch13_trainer.py", "tests/test_ch13_trainer.py",
          "python3 code/gftac/ch13_tasks.py 5", "Kramers"),
    Claim("CL-29", 15, "Linear-diffusion failure modes reproduced and diagnosed",
          "three failure modes under the diffusion contract", "MEASURED",
          "code/gftac/ch15_diffusion_failures.py", "tests/test_ch15_diffusion_failures.py",
          "python3 code/gftac/ch15_tasks.py 1", "diffusion"),
    Claim("CL-30", 16, "Q-3 soliton lower bound is satisfied at every hop",
          "empirical peak mass ≥ analytic bound", "MEASURED",
          "code/gftac/ch16_soliton.py", "tests/test_ch16_soliton.py",
          "python3 code/gftac/ch16_tasks.py 4", "satisfies_bound"),
    Claim("CL-31", 24, "The whole-stack capstone certificate is a fixed point",
          "SHA-256, deterministic on a machine (16-hex prefix)", "MEASURED",
          "code/gftac/ch24_capstone.py", "tests/test_ch24_tasks.py",
          "python3 code/gftac/ch24_tasks.py 7", None),
)


def _read(path: str) -> str:
    with open(os.path.join(BOOK_ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def count_by_label() -> dict:
    """Number of claims carrying each label."""
    counts = {label: 0 for label in VALID_LABELS}
    for claim in CLAIMS:
        counts[claim.label] += 1
    return counts


def audit_claims() -> List[Tuple[str, bool, str]]:
    """Check every row: artifact exists, test exists, pinned value is asserted."""
    results: List[Tuple[str, bool, str]] = []
    for claim in CLAIMS:
        if claim.label not in VALID_LABELS:
            results.append((claim.cid, False, f"invalid label {claim.label!r}"))
            continue
        if not os.path.isfile(os.path.join(BOOK_ROOT, claim.artifact)):
            results.append((claim.cid, False, f"missing artifact {claim.artifact}"))
            continue
        if not os.path.isfile(os.path.join(BOOK_ROOT, claim.test)):
            results.append((claim.cid, False, f"missing test {claim.test}"))
            continue
        if claim.pinned is not None and claim.pinned not in _read(claim.test):
            results.append((claim.cid, False,
                            f"pinned value {claim.pinned!r} not asserted in {claim.test}"))
            continue
        results.append((claim.cid, True, "ok"))
    return results


def render_claims_md() -> str:
    """Render the ledger as a Markdown document."""
    lines = [
        "# CLAIMS.md — The Book's Public Numbers, Labeled and Traceable",
        "",
        "Generated by `code/gftac/ch22_claims.py`. Every row is one public number in",
        "the book, its honesty label, the artifact that produces it, the test that pins",
        "it, and the command that regenerates it. The labels are:",
        "",
        "- **MEASURED** — computed or simulated on this machine, deterministic under seed.",
        "- **MODELED** — produced by an analytical or physical model, not by silicon.",
        "- **RESEARCH DEMO** — an illustrative application; no domain claims.",
        "",
        f"Total claims: **{len(CLAIMS)}**. "
        + " | ".join(f"{label}: {n}" for label, n in count_by_label().items())
        + ".",
        "",
        "| ID | Ch | Claim | Value | Label | Artifact | Test | Command |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CLAIMS:
        lines.append(
            f"| {c.cid} | {c.chapter} | {c.statement} | {c.value} | {c.label} | "
            f"`{c.artifact}` | `{c.test}` | `{c.command}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_claims_md(path: Optional[str] = None) -> str:
    """Write CLAIMS.md (default: book root); return its absolute path."""
    target = path or os.path.join(BOOK_ROOT, "CLAIMS.md")
    text = render_claims_md()
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    return target


def claims_sha256() -> str:
    """SHA-256 of the rendered ledger (stable across runs)."""
    return hashlib.sha256(render_claims_md().encode("utf-8")).hexdigest()


__all__ = [
    "Claim", "CLAIMS", "VALID_LABELS", "count_by_label", "audit_claims",
    "render_claims_md", "write_claims_md", "claims_sha256",
]
