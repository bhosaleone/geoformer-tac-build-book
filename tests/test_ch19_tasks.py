# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch19_tasks.py — Pins every printed line of every Chapter 19 task.

Each assertion is a substring of the task's stdout, so no transcript in the
chapter can drift from the artifact that produces it.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from gftac import ch19_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_task_1_ontology_census():
    out = _capture(ch19_tasks.task_1)
    assert "Clinical Knowledge Graph Census — RESEARCH DEMO:" in out
    assert "  • Entities:                       128" in out
    assert "  • Relation Types:                 6" in out
    assert "  • Directed Edges:                 118" in out
    assert "  Entities by Type:" in out
    assert "      symptom                  42" in out
    assert "      drug                     26" in out
    assert "      biomarker                24" in out
    assert "      pathophysiology          24" in out
    assert "      disease                  12" in out
    assert "  Edges by Relation:" in out
    assert "      presents_with            42" in out
    assert "      caused_by                25" in out
    assert "      indicated_biomarker      24" in out
    assert "      treated_by               25" in out
    assert "      contraindicated_in       1" in out
    assert "      complication_of          1" in out


def test_task_2_compile_footprint():
    out = _capture(ch19_tasks.task_2)
    assert "Compiling the Clinical Ontology into Sparse Hippocampal Memory — RESEARCH DEMO:" in out
    assert "  • Entities:                       128" in out
    assert "  • Forward Relations:              6" in out
    assert "  • Reverse Relations:              6" in out
    assert "  • Stored Nonzeros:                236" in out
    assert "  • Sparse Physical Footprint:      4,720 bytes (0.0045 MB)" in out
    assert "  • Dense Equivalent:               786,432 bytes (0.75 MB)" in out
    assert "  • Compression Factor:             166.6x" in out
    assert "  • Invariant Battery Audit:        PASSED" in out


def test_task_3_differential_diagnosis():
    out = _capture(ch19_tasks.task_3)
    assert "Differential Diagnosis — Canonical STEMI vs Aortic Dissection (RESEARCH DEMO):" in out
    assert (
        "  Observed findings (4): Retrosternal_Crushing_Chest_Pain, Diaphoresis, "
        "ST_Segment_Elevation, Tearing_Interscapular_Back_Pain" in out
    )
    assert "  Ranked differentials:" in out
    assert "    1. Acute_STEMI                 75.00%  (raw score 0.750000)" in out
    assert "    2. Aortic_Dissection           25.00%  (raw score 0.250000)" in out
    assert "  Primary diagnosis:   Acute_STEMI" in out
    assert (
        "  Causal chain:        Acute_STEMI -> Coronary_Atherosclerotic_Plaque_Rupture "
        "-> Acute_Occlusive_Thrombosis -> Transmural_Myocardial_Ischemia "
        "-> Myocyte_Coagulative_Necrosis" in out
    )
    assert (
        "  Recommended (DEMO):  Primary_Percutaneous_Coronary_Intervention, "
        "Aspirin_Dual_Antiplatelet, Unfractionated_Heparin, High_Intensity_Atorvastatin"
        in out
    )


def test_task_4_contraindication_alert():
    out = _capture(ch19_tasks.task_4)
    assert "Contraindication Safety Check — RESEARCH DEMO:" in out
    assert "  Case A (dissection findings) -> primary Aortic_Dissection" in out
    assert "    Alert fired:   True" in out
    assert (
        "    Alert:         CRITICAL WARNING: Thrombolysis_tPA is strictly "
        "CONTRAINDICATED in Aortic_Dissection!" in out
    )
    assert "  Case B (STEMI findings)      -> primary Acute_STEMI" in out
    assert "    Alert fired:   False" in out
    assert "    Alert:         none" in out


def test_task_5_causal_trace():
    out = _capture(ch19_tasks.task_5)
    assert "Multi-Hop Pathophysiological Causal Trace — RESEARCH DEMO:" in out
    assert (
        "  Acute_STEMI                  depth 4  | Acute_STEMI -> "
        "Coronary_Atherosclerotic_Plaque_Rupture -> Acute_Occlusive_Thrombosis -> "
        "Transmural_Myocardial_Ischemia -> Myocyte_Coagulative_Necrosis" in out
    )
    assert (
        "  Acute_Appendicitis           depth 3  | Acute_Appendicitis -> "
        "Appendiceal_Lumen_Obstruction_by_Fecalith -> "
        "Intraluminal_Bacterial_Overgrowth_and_Ischemia -> "
        "Gangrenous_Transmural_Perforation" in out
    )
    assert (
        "  Diabetic_Ketoacidosis        depth 3  | Diabetic_Ketoacidosis -> "
        "Absolute_Insulin_Deficiency -> Unchecked_Lipolysis_and_Free_Fatty_Acids -> "
        "Hepatic_Ketogenesis_Acetoacetate_BetaHB" in out
    )
    assert (
        "  Pulmonary_Embolism           depth 3  | Pulmonary_Embolism -> "
        "Thrombus_Embolization_to_Pulmonary_Artery -> Ventilation_Perfusion_Mismatch -> "
        "Acute_Right_Ventricular_Strain" in out
    )
    assert (
        "  Acute_Ischemic_Stroke_MCA    depth 2  | Acute_Ischemic_Stroke_MCA -> "
        "Thromboembolic_Occlusion_of_Middle_Cerebral_Artery -> "
        "Ischemic_Penumbra_and_Cytotoxic_Edema" in out
    )
    assert (
        "  Acute_Pancreatitis           depth 2  | Acute_Pancreatitis -> "
        "Premature_Intraacinar_Trypsinogen_Activation -> "
        "Autodigestion_and_Fat_Necrosis" in out
    )
    assert (
        "  Bacterial_Meningitis         depth 2  | Bacterial_Meningitis -> "
        "Hematogenous_Seeding_to_Subarachnoid_Space -> "
        "Meningeal_Neutrophilic_Inflammation_Exudate" in out
    )
    assert (
        "  Tension_Pneumothorax         depth 2  | Tension_Pneumothorax -> "
        "One_Way_Valve_Air_Entrapment -> Impaired_Venous_Return_Preload_Collapse"
        in out
    )
    assert (
        "  Acute_Pericarditis           depth 1  | Acute_Pericarditis -> "
        "Pericardial_Serofibrinous_Inflammation" in out
    )
    assert (
        "  Aortic_Dissection            depth 1  | Aortic_Dissection -> "
        "Cystic_Medial_Necrosis_and_Intimal_Tear" in out
    )
    assert (
        "  Bacterial_Pneumonia          depth 1  | Bacterial_Pneumonia -> "
        "Streptococcus_Pneumoniae_Alveolar_Infiltration" in out
    )
    assert "  Deep_Vein_Thrombosis         depth 0  | Deep_Vein_Thrombosis" in out
    assert "  Longest causal chain: depth 4 (Acute_STEMI)" in out


def test_task_6_sdk_parity():
    out = _capture(ch19_tasks.task_6)
    assert "SDK Parity Audit — Book Engine vs GeoFormerClinicalDiagnosticEngine (RESEARCH DEMO):" in out
    assert "  Case 1 ranking (STEMI vs dissection): ['Acute_STEMI', 'Aortic_Dissection']" in out
    assert "  SDK    ranking (STEMI vs dissection): ['Acute_STEMI', 'Aortic_Dissection']" in out
    assert "  Rankings match:                       True" in out
    assert "  Case 2 contraindication alert match:  True" in out
    assert "  Max differential probability diff:    0.00e+00" in out
    assert "  Full Parity Verified:                 True" in out
