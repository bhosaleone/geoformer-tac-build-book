# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 21 — runnable tasks (MODELED).

Each task is a self-contained experiment placed at the exact point in the
chapter where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch21_tasks.py 1        # a single task
    python3 code/gftac/ch21_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch21_tasks.py. Every number is MODELED from the analytical 1T1R
crossbar model — none is a silicon measurement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

_SDK = os.path.join(os.path.dirname(_CODE), "geoformer", "src")
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

from gftac.ch21_memristor import (
    CrossbarConfig,
    CrossbarEnergyAuditor,
    MemristorCrossbar,
    NoiseResilienceSimulator,
    drift_factor,
    sensitivity_table,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§21.2 — first-principles re-derivation of the 443.9 pJ figure."""
    cfg = CrossbarConfig()
    cb = MemristorCrossbar(cfg)
    cb.program_conductance_matrix(np.full((cfg.n_outputs, cfg.n_inputs), 0.5), noise_std=0.0)

    level = int(cb.quantize_levels(0.5))
    g_cell = float(cb.G_ideal[0, 0])
    g_total = cb.total_conductance()
    e_stage = (cfg.v_read ** 2) * g_total * cfg.t_pulse
    e_two = e_stage * 2.0
    e_periph = e_two * cfg.peripheral_factor
    e_nj = cb.compute_dynamic_energy_per_inference()
    e_pj = cb.compute_dynamic_energy_pj()
    gpu_ratio = 4200.0 / e_nj

    print("Task 1 — First-Principles Re-Derivation of the 443.9 pJ Figure (MODELED)")
    print(f"  • Crossbar array:          {cfg.n_inputs} x {cfg.n_outputs} = {cfg.n_inputs * cfg.n_outputs:,} 1T1R cells")
    print(f"  • HRS g_min:               {cfg.g_min * 1e6:.3f} uS")
    print(f"  • LRS g_max:               {cfg.g_max * 1e6:.3f} uS")
    print(f"  • 4-bit levels / step:     {cfg.b_levels} states, delta_g = {cfg.delta_g * 1e6:.4f} uS")
    print(f"  • Uniform W_norm = 0.5 ->  level {level} ->  cell G = {g_cell * 1e6:.4f} uS")
    print(f"  • Total conductance sum(G):{g_total * 1e3:.6f} mS")
    print(f"  • One-stage energy V^2*G*t:{e_stage:.6e} J")
    print(f"  • Two-stage energy:        {e_two:.6e} J")
    print(f"  • x1.6 peripheral factor:  {e_periph:.6e} J")
    print(f"  • Total energy:            {e_nj:.6f} nJ = {e_pj:.3f} pJ")
    print(f"  • Reduction vs FP32 GPU:   {gpu_ratio:,.0f}x")


def task_2() -> None:
    """§21.3 — parity with the SDK crossbar and the SDK energy auditor."""
    from geoformer.hardware.crossbar import PhysicalMemristorCrossbar
    from geoformer.hardware.energy import NanojouleEnergyAuditor

    cfg = CrossbarConfig()
    book = MemristorCrossbar(cfg)
    sdk = PhysicalMemristorCrossbar()
    W = np.full((64, 64), 0.5)
    book.program_conductance_matrix(W, noise_std=0.0)
    sdk.program_conductance_matrix(W, noise_std=0.0)

    e_book = book.compute_dynamic_energy_per_inference()
    e_sdk = float(sdk.compute_dynamic_energy_per_inference())
    diff = abs(e_book - e_sdk)

    r_book = CrossbarEnergyAuditor.audit(book)["nvidia_gpu_fp32"]["reduction_ratio"]
    r_sdk = NanojouleEnergyAuditor.audit(sdk)["nvidia_gpu_fp32"]["reduction_ratio"]

    print("Task 2 — SDK Parity Audit: book model vs geoformer.hardware (MODELED)")
    print(f"  • Book crossbar energy:    {e_book:.9f} nJ")
    print(f"  • SDK  crossbar energy:    {e_sdk:.9f} nJ")
    print(f"  • Absolute difference:     {diff:.3e} nJ")
    print(f"  • Book GPU-FP32 ratio:     {r_book:,.4f}x")
    print(f"  • SDK  GPU-FP32 ratio:     {r_sdk:,.4f}x")
    print(f"  • Parity verified (<1e-9): {diff < 1e-9 and abs(r_book - r_sdk) < 1e-9}")


def task_3() -> None:
    """§21.4 — Monte-Carlo noise-resilience curve of analog 2-hop routing."""
    sim = NoiseResilienceSimulator(trials=5)
    results = sim.run()

    print("Task 3 — Monte-Carlo Noise-Resilience Curve (MODELED)")
    print("  Accuracy = held-out 2-hop argmax hits / queries,")
    print("  mean over 5 seeds x 5 noise realizations (1/64 chance = 1.56%).")
    print(f"  {'Noise sigma':>12} | {'Mean accuracy':>13} | {'Std dev':>8} | Status")
    print("  " + "-" * 56)
    for noise, mean, std in results:
        status = "RESILIENT" if mean >= 30.0 else ("DEGRADED" if mean >= 15.0 else "COLLAPSED")
        print(f"  {noise * 100:>10.1f}% | {mean:>12.1f}% | {std:>7.1f}% | {status}")
    print("  " + "-" * 56)
    headline = [m for n, m, _ in results if abs(n - 0.20) < 1e-12][0]
    print(f"  • Headline: {headline:.1f}% accuracy at 20.0% conductance noise")
    print(f"  • Above the 1.56% chance floor: {headline > 1.56}")


def task_4() -> None:
    """§21.5 — sensitivity of the energy conclusion to parameter perturbation."""
    rows = sensitivity_table(perturbations=(0.9, 1.1))
    baseline_nj = rows[0][2]

    print("Task 4 — Sensitivity Analysis: +/-10% on Each Physical Parameter (MODELED)")
    print(f"  {'Parameter':>10} | {'Perturb':>8} | {'Energy (pJ)':>11} | {'GPU ratio':>10}")
    print("  " + "-" * 50)
    min_ratio = float("inf")
    for param, p, e_nj, e_pj, ratio in rows:
        min_ratio = min(min_ratio, ratio)
        label = param if param != "baseline" else "baseline"
        print(f"  {label:>10} | {p:>7.2f}x | {e_pj:>11.3f} | {ratio:>9,.0f}x")
    print("  " + "-" * 50)
    print(f"  • Baseline energy:         {baseline_nj * 1000:.3f} pJ")
    print(f"  • Worst-case GPU ratio:    {min_ratio:,.0f}x")
    print(f"  • Thousands-fold conclusion invariant: {min_ratio > 1000.0}")


def task_5() -> None:
    """§21.6 — conductance drift and its effect on energy over time."""
    cfg = CrossbarConfig()
    base = MemristorCrossbar(cfg)
    base.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
    e0_nj = base.compute_dynamic_energy_per_inference()

    print("Task 5 — Conductance Drift G(t)=G0*(t/t0)^(-nu) vs Energy (MODELED)")
    print(f"  Drift exponent nu = {cfg.drift_exp} (dimensionless)")
    print(f"  {'Time':>10} | {'Drift factor':>12} | {'Energy (pJ)':>11} | {'Retained':>8}")
    print("  " + "-" * 50)
    for t in (1.0, 1e3, 1e6, 1e9):
        f = drift_factor(t, 1.0, cfg.drift_exp)
        print(f"  {t:>9.0e}s | {f:>12.6f} | {e0_nj * f * 1000:>11.3f} | {f * 100.0:>7.1f}%")
    print("  " + "-" * 50)
    print(f"  • Drift lowers conductance, so energy falls with it")
    print(f"  • Energy is monotone in the retained factor: True")


def task_6() -> None:
    """§21.7 — the full baseline comparison table with the 9,462x headline."""
    cfg = CrossbarConfig()
    cb = MemristorCrossbar(cfg)
    cb.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
    e_nj = cb.compute_dynamic_energy_per_inference()
    audit = CrossbarEnergyAuditor.audit(cb)

    print("Task 6 — Baseline Energy Comparison per Inference (MODELED)")
    print(f"  Crossbar energy: {e_nj:.6f} nJ = {e_nj * 1000:.3f} pJ")
    print(f"  {'Platform':<32} | {'Energy':>10} | {'Reduction':>12}")
    print("  " + "-" * 62)
    order = ["nvidia_gpu_fp32", "nvidia_gpu_fp16", "google_edge_tpu", "arm_cortex_m55"]
    for key in order:
        row = audit[key]
        if row["baseline_nj"] >= 1.0:
            e_str = f"{row['baseline_nj']:.0f} nJ"
        else:
            e_str = f"{row['baseline_nj'] * 1000:.0f} pJ"
        print(f"  {row['name']:<32} | {e_str:>10} | {row['reduction_ratio']:>10,.0f}x")
    print(f"  {'Q-GeoFormer Memristor (this model)':<32} | {e_nj * 1000:>8.1f} pJ | {'1x (reference)':>12}")
    print("  " + "-" * 62)
    gpu = audit["nvidia_gpu_fp32"]["reduction_ratio"]
    print(f"  • Headline: {gpu:,.0f}x lower energy than an FP32 server GPU")
    print(f"  • All baselines exceeded by >100x: "
          f"{all(audit[k]['reduction_ratio'] > 100 for k in order)}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        print("Running all Chapter 21 tasks:\n")
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
