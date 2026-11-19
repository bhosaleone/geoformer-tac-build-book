# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch14_eval.py — Unit tests for Chapter 14 evaluation protocol and quadrature.
"""

from __future__ import annotations

import math
import pytest
import scipy.stats as stats
import torch

from gftac.ch14_eval import (
    audit_zero_shot_disjointness,
    compute_effective_depth,
    compute_hits_at_k,
    compute_mrr,
    compute_spectral_entropy,
    compute_target_modal_mass,
    simpson_student_t_cdf,
    student_t_pdf,
    welch_t_test,
)


def test_hits_at_k_and_mrr_exact():
    # 4 queries, 5 classes
    logits = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],  # tgt 2 is rank 1
        [0.7, 0.8, 0.1, 0.2, 0.3],  # tgt 0 is rank 2
        [0.5, 0.6, 0.2, 0.1, 0.4],  # tgt 4 is rank 3
        [0.9, 0.3, 0.8, 0.7, 0.1],  # tgt 1 is rank 4
    ])
    targets = torch.tensor([2, 0, 4, 1])

    hits1 = compute_hits_at_k(logits, targets, k=1)
    hits2 = compute_hits_at_k(logits, targets, k=2)
    hits3 = compute_hits_at_k(logits, targets, k=3)
    hits4 = compute_hits_at_k(logits, targets, k=4)
    hits5 = compute_hits_at_k(logits, targets, k=5)
    mrr = compute_mrr(logits, targets)

    assert hits1 == 0.25
    assert hits2 == 0.50
    assert hits3 == 0.75
    assert hits4 == 1.00
    assert hits5 == 1.00

    # Expected MRR: (1/1 + 1/2 + 1/3 + 1/4) / 4 = (25/12) / 4 = 25/48 ≈ 0.5208333
    assert abs(mrr - (25.0 / 48.0)) < 1e-6
    # Theorem B.14.1 bounds: Hits@1 <= MRR <= (1 + Hits@1)/2
    assert hits1 <= mrr <= (1.0 + hits1) / 2.0


def test_attractor_modal_mass_and_entropy():
    z_peaked = torch.tensor([[0.01, 0.02, 0.95, 0.01, 0.01]])
    z_uniform = torch.full((1, 5), 0.20)
    target = torch.tensor([2])

    p_peak = compute_target_modal_mass(z_peaked, target)
    p_unif = compute_target_modal_mass(z_uniform, target)
    h_peak = compute_spectral_entropy(z_peaked)
    h_unif = compute_spectral_entropy(z_uniform)

    assert abs(p_peak - 0.95) < 1e-6
    assert abs(p_unif - 0.20) < 1e-6
    assert h_peak < 0.30
    assert abs(h_unif - math.log(5.0)) < 1e-4


def test_effective_depth():
    drifts = [0.1, 0.05, 0.001, 0.00005, 0.00001]
    tau = compute_effective_depth(drifts, tol=1e-4)
    assert tau == 4

    tau_never = compute_effective_depth([0.5, 0.4, 0.3], tol=1e-4)
    assert tau_never == 3


def test_student_t_pdf():
    # Check PDF vs scipy.stats.t.pdf
    for df in [1.0, 3.5, 8.0, 30.0]:
        for t in [-3.0, -1.0, 0.0, 0.8, 2.5]:
            val_us = student_t_pdf(t, df)
            val_sp = stats.t.pdf(t, df)
            assert abs(val_us - val_sp) < 1e-12


def test_simpson_student_t_cdf_precision():
    # Simpson quadrature should match scipy.stats.t.cdf to within 1e-11
    for df in [2.0, 5.0, 8.0, 20.0]:
        for t in [-4.0, -2.306, -0.5, 0.0, 0.5, 1.96, 3.0]:
            p_us = simpson_student_t_cdf(t, df=df, n=2000)
            p_sp = stats.t.cdf(t, df=df)
            assert abs(p_us - p_sp) < 1e-11, f"Failed at df={df}, t={t}: diff={abs(p_us - p_sp)}"


def test_welch_t_test_against_scipy():
    sample1 = [96.8, 97.2, 95.9, 98.1, 96.5]
    sample2 = [12.5, 14.0, 11.8, 13.2, 12.0]

    t_us, df_us, p_us = welch_t_test(sample1, sample2)
    res = stats.ttest_ind(sample1, sample2, equal_var=False)

    assert abs(t_us - res.statistic) < 1e-6
    assert abs(p_us - res.pvalue) < 1e-10
    assert p_us < 1e-10


def test_zero_shot_disjointness():
    train_facts = [(0, 1, 1), (1, 1, 2), (2, 1, 3)]
    test_queries = {
        2: [(0, 1, 2), (1, 1, 3)],  # Disjoint from 1-hop facts
        3: [(0, 1, 3)],
    }
    report = audit_zero_shot_disjointness(train_facts, test_queries)
    assert report[2] == 0
    assert report[3] == 0

    # Leak intentional fact to test detection
    test_queries_leaked = {
        2: [(0, 1, 1)],  # Overlap!
    }
    leak_report = audit_zero_shot_disjointness(train_facts, test_queries_leaked)
    assert leak_report[2] == 1
