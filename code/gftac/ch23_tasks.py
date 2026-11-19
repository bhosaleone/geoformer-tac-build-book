# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 23 — runnable tasks.

Each task is a self-contained experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch23_tasks.py 1        # a single task
    python3 code/gftac/ch23_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch23_tasks.py. The single line that reports Z3 wall time is
run-dependent and is shown in the book wrapped in «...».
"""

from __future__ import annotations

import json
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch23_z3_certificates import (
    _time_proofs,
    build_certificate,
    bounded_domain_disclosure,
    property_check_q1,
    property_check_q2,
    run_geoformer_theorems,
    run_tac_theorems,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§23.1 — Reproduce the five TAC Z3 theorems and print their status."""
    records, elapsed = _time_proofs(run_tac_theorems)

    print("TAC Z3 Certificate — five theorems:")
    for r in records:
        print(
            f"  • {r.proof_id}  {r.name:<36} status={r.status:<5} "
            f"proved={str(r.proved):<5} repo={r.repo_method}"
        )
    print(f"  • All five goals proved:  {all(r.proved for r in records)}")
    print(f"  • Z3 wall time:           {elapsed:.4f} s (run-dependent)")


def task_2() -> None:
    """§23.1 — Reproduce the GeoFormer Q-1/Q-2/Q-3 invariants."""
    records = run_geoformer_theorems()

    print("GeoFormer Z3 Certificate — Q-series invariants:")
    for r in records:
        print(
            f"  • {r.proof_id}  {r.name:<36} status={r.status:<5} "
            f"proved={str(r.proved):<5} repo={r.repo_method}"
        )
    print(f"  • All three goals proved:  {all(r.proved for r in records)}")


def task_3() -> None:
    """§23.2 — The bounded-domain disclosure: premises assumed, coverage omitted."""
    records = bounded_domain_disclosure()

    print("Bounded-Domain Disclosure — what each certificate assumes and omits:")
    for r in records:
        print(f"  [{r.proof_id}] {r.name}")
        print(f"      domain:   {r.domain}")
        print(f"      assumes:  {'; '.join(r.premises)}")
        print(f"      proves:   {r.conclusion}")
        print(f"      omits:    {'; '.join(r.omits)}")
    print("")
    print(
        f"  Disclosure summary: {len(records)} certificates over the real domain; "
        f"0 cover floating-point rounding."
    )


def task_4() -> None:
    """§23.3 — Property-based randomized check of Q-1 in FP32 and FP64."""
    res = property_check_q1()

    pct32 = 100.0 * res["fp32_passed"] / res["total"]
    pct64 = 100.0 * res["fp64_passed"] / res["total"]
    print("Property-Based Check of Q-1 (discrete geodesic contraction):")
    print(
        f"  Ensembles:                 {res['num_graphs']} random graphs × "
        f"{res['edges_per_graph']} edges = {res['total']} edges"
    )
    print("  Domain:                    W_t >= delta_w > 0, eps > 0")
    print(f"  FP32 contractions held:    {res['fp32_passed']}/{res['total']}  ({pct32:.2f}%)")
    print(f"  FP64 contractions held:    {res['fp64_passed']}/{res['total']}  ({pct64:.2f}%)")
    print(f"  FP32 == FP64 on all edges: {res['agreement']}")
    print("  Bounded-domain probe (delta_w = 2^-25 * W, W >= 1):")
    print(
        f"    FP32 contractions held:  {res['boundary_fp32_passed']}/{res['boundary_total']}"
    )
    print(
        f"    FP64 contractions held:  {res['boundary_fp64_passed']}/{res['boundary_total']}"
    )
    print(f"    FP32 == FP64:            {res['boundary_agreement']}")


def task_5() -> None:
    """§23.4 — Property-based randomized check of Q-2 over dense lattices."""
    res = property_check_q2()

    print("Property-Based Check of Q-2 (Fiedler floor lambda_2 >= N * delta_w):")
    print(
        f"  Ensembles:                 {res['num_lattices']} random dense lattices "
        f"of N = {res['n']} nodes"
    )
    print("  Domain:                    every off-diagonal symmetrized conductance >= delta_w > 0")
    print(f"  FP32 bound held:           {res['fp32_passed']}/{res['num_lattices']}")
    print(f"  FP64 bound held:           {res['fp64_passed']}/{res['num_lattices']}")
    print(f"  FP32 == FP64 on all lattices: {res['agreement']}")
    print(f"  Minimum margin (lambda_2 - N*delta_w): {res['min_margin']:.3f}")


def task_6() -> None:
    """§23.5 — Emit the JSON certificate with its stable SHA-256."""
    cert = build_certificate()

    print("Z3 + Property-Based Verification Certificate (canonical JSON):")
    print(json.dumps(cert, indent=2, sort_keys=True))
    print("")
    print(f"  SHA-256 of the canonical payload (this field excluded): {cert['sha256']}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        print("Running all Chapter 23 tasks:\n")
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
