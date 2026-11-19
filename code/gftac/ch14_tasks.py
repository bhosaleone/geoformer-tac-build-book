# SPDX-License-Identifier: AGPL-3.0-only
"""
code/gftac/ch14_tasks.py — Runnable tasks for Chapter 14.
"""

from __future__ import annotations

import math
import os
import sys
import torch

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch14_eval import (
    compute_hits_at_k,
    compute_mrr,
    compute_target_modal_mass,
    compute_spectral_entropy,
    compute_effective_depth,
    student_t_pdf,
    simpson_student_t_cdf,
    welch_t_test,
    audit_zero_shot_disjointness,
)


def task_1() -> None:
    """Task 1: Retrieval Metrics Verification on Synthetic Logits."""
    print("\n--- Task 1 ---")
    # 4 samples, 5 classes
    # Sample 0: Target 2 is rank 1 (logits: [0.1, 0.2, 0.9, 0.3, 0.4]) -> rank 1
    # Sample 1: Target 0 is rank 2 (logits: [0.7, 0.8, 0.1, 0.2, 0.3]) -> rank 2
    # Sample 2: Target 4 is rank 3 (logits: [0.5, 0.6, 0.2, 0.1, 0.4]) -> rank 3
    # Sample 3: Target 1 is rank 4 (logits: [0.9, 0.3, 0.8, 0.7, 0.1]) -> rank 4
    logits = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],
        [0.7, 0.8, 0.1, 0.2, 0.3],
        [0.5, 0.6, 0.2, 0.1, 0.4],
        [0.9, 0.3, 0.8, 0.7, 0.1],
    ])
    targets = torch.tensor([2, 0, 4, 1])

    hits1 = compute_hits_at_k(logits, targets, k=1)
    hits3 = compute_hits_at_k(logits, targets, k=3)
    hits5 = compute_hits_at_k(logits, targets, k=5)
    mrr = compute_mrr(logits, targets)

    print("Retrieval Metrics Audit:")
    print(f"  Hits@1:  {hits1 * 100:.1f}%  (Target: 25.0%)")
    print(f"  Hits@3:  {hits3 * 100:.1f}%  (Target: 75.0%)")
    print(f"  Hits@5:  {hits5 * 100:.1f}%  (Target: 100.0%)")
    print(f"  MRR:     {mrr:.4f}   (Expected: (1/1 + 1/2 + 1/3 + 1/4)/4 = 0.5208)")
    print(f"  Hits@k Monotonicity Preserved: {hits1 <= hits3 <= hits5}")


def task_2() -> None:
    """Task 2: Attractor Modal Mass & Entropy Scaling."""
    print("\n--- Task 2 ---")
    # Peaked attractor distribution: 95% on target
    z_peaked = torch.tensor([[0.01, 0.02, 0.95, 0.01, 0.01]])
    # Uniform attractor distribution: 20% on each
    z_uniform = torch.full((1, 5), 0.20)
    targets = torch.tensor([2])

    p_peak = compute_target_modal_mass(z_peaked, targets)
    h_peak = compute_spectral_entropy(z_peaked)

    p_unif = compute_target_modal_mass(z_uniform, targets)
    h_unif = compute_spectral_entropy(z_uniform)

    print("Attractor State Diagnostics:")
    print(f"  Peaked State:  Modal Mass P(y) = {p_peak:.4f} | Entropy H(z) = {h_peak:.4f}")
    print(f"  Uniform State: Modal Mass P(y) = {p_unif:.4f} | Entropy H(z) = {h_unif:.4f}")
    print(f"  Sharp Soliton Signature Verified: {p_peak > p_unif and h_peak < h_unif}")


def task_3() -> None:
    """Task 3: Exact Simpson Quadrature vs. Scipy Stats Precision Audit."""
    print("\n--- Task 3 ---")
    import scipy.stats as stats

    nu = 8
    test_points = [0.5, 1.2, 2.306004, 3.5, 5.0]
    max_err = 0.0

    for t_val in test_points:
        p_simpson = simpson_student_t_cdf(t_val, df=nu, n=2000)
        p_scipy = stats.t.cdf(t_val, df=nu)
        err = abs(p_simpson - p_scipy)
        max_err = max(max_err, err)

    print("Simpson Quadrature Precision Audit (df = 8):")
    print(f"  Maximum Absolute Discrepancy: {max_err:.2e}")
    print(f"  Precision Criterion (< 1e-12): {max_err < 1e-12}")


def task_4() -> None:
    """Task 4: Welch's t-test on SUPPLIED values — a worked example, not a benchmark.

    The two lists below are illustrative inputs written into the task so the
    statistic can be reproduced exactly. They are NOT five independently
    executed benchmark runs. This task pins the correctness of Welch's test,
    not the empirical provenance of the numbers; see LIMITATIONS.md.
    """
    print("\n--- Task 4 ---")
    # Illustrative 5-value inputs (%), supplied to demonstrate the calculation.
    # Not measured in this chapter; see LIMITATIONS.md and RESEARCH_BRIEF.md.
    geoformer_tac = [96.8, 97.2, 95.9, 98.1, 96.5]
    flat_baseline = [12.5, 14.0, 11.8, 13.2, 12.0]

    t_stat, df, p_val = welch_t_test(geoformer_tac, flat_baseline)
    m_gf = sum(geoformer_tac) / len(geoformer_tac)
    m_base = sum(flat_baseline) / len(flat_baseline)

    print("Statistical Hypothesis Testing (ILLUSTRATIVE supplied values, NOT a measurement):")
    print(f"  GeoFormer+TAC Mean Hits@1: {m_gf:.2f}%")
    print(f"  Flat Baseline Mean Hits@1: {m_base:.2f}%")
    print(f"  Welch t-statistic:         {t_stat:.4f}")
    print(f"  Degrees of Freedom (ν):    {df:.2f}")
    print(f"  Two-Tailed p-value:        {p_val:.2e}")
    print(f"  Statistically Decisive (p < 0.001): {p_val < 1e-3}")
    print("  Provenance: values hardcoded in ch14_tasks.py; statistic only.")


def task_5() -> None:
    """Task 5: Zero-Shot Disjointness Invariant Audit."""
    print("\n--- Task 5 ---")
    train_facts = [(0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4)]
    test_queries = {
        2: [(0, 1, 2), (1, 1, 3)],  # 2-hop pairs: (0, 2), (1, 3)
        3: [(0, 1, 3), (1, 1, 4)],  # 3-hop pairs: (0, 3), (1, 4)
        4: [(0, 1, 4)],              # 4-hop pairs: (0, 4)
    }

    overlaps = audit_zero_shot_disjointness(train_facts, test_queries)
    print("Zero-Shot Disjointness Audit:")
    for depth, count in overlaps.items():
        print(f"  Depth {depth}-hop Overlaps with 1-hop Train: {count}")
    all_disjoint = all(count == 0 for count in overlaps.values())
    print(f"  Zero-Shot Disjointness Invariant: {all_disjoint} ✔")


def task_6() -> None:
    """Task 6: Autonomous Halting Depth Detection (tau*)."""
    print("\n--- Task 6 ---")
    # Simulated drift decay: ||z_{t+1} - z_t||
    drift_history = [0.45, 0.22, 0.08, 0.015, 0.002, 0.0003, 0.00005, 0.00001]
    tau_star = compute_effective_depth(drift_history, tol=1e-4)

    print("Emergent Attractor Halting:")
    print(f"  Drift History: {drift_history[:tau_star]}")
    print(f"  Emergent Halting Depth τ*: {tau_star} steps (tol = 1e-4)")
    print(f"  Halting strictly terminates within finite schedule: {tau_star <= len(drift_history)}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        for task in (task_1, task_2, task_3, task_4, task_5, task_6):
            task()
    else:
        {1: task_1, 2: task_2, 3: task_3, 4: task_4,
         5: task_5, 6: task_6}[int(sys.argv[1])]()


if __name__ == "__main__":
    main()
