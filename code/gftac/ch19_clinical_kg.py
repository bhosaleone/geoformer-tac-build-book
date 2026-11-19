# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 19 — Clinical Knowledge-Graph Integration (RESEARCH DEMO).

This module compiles a small clinical ontology — diseases, symptoms, biomarkers,
pathophysiological mechanisms, drugs and contraindications — into the sparse
hippocampal memory of Chapter 18, and builds a diagnostic engine on top of it
that performs three relational inferences:

    * differential diagnosis      (findings  ->  ranked diseases)
    * causal path tracing         (disease   ->  pathophysiology chain)
    * contraindication checking   (drug      ->  disease safety alert)

The ontology reproduced here is the *same* graph as the production SDK's
``geoformer.clinical.ontology.build_clinical_knowledge_graph()`` — the same
entities, the same six relation types, and the same directed edges — so the
chapter's Task 6 parity audit can compare this engine against
``geoformer.clinical.engine.GeoFormerClinicalDiagnosticEngine`` edge for edge and
ranking for ranking.

**RESEARCH DEMO.** Nothing in this module is a medical device, a clinical
decision support tool, or medical advice. It is a demonstration that the sparse
relational machinery of Part V can carry genuine domain structure. The
"diagnoses" it prints are the outputs of a graph propagation, not the outputs of
a clinician.

Author: Atmabhan Pandit / Shrikant Bhosale (Eklavya Project)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from gftac.ch18_sparse_memory import (
    SparseHippocampalMemory,
    SparseMemoryConfig,
)

__all__ = [
    "RELATION_IDS",
    "MedicalOntology",
    "build_clinical_knowledge_graph",
    "ClinicalDiagnosticEngine",
]


# ---------------------------------------------------------------------------
# The relation vocabulary.  These ids are load-bearing: the diagnostic engine
# hard-codes the reverse ids 100 + r for abduction, exactly as the SDK does.
# ---------------------------------------------------------------------------
RELATION_IDS: Dict[str, int] = {
    "presents_with": 0,        # Disease  -> Symptom
    "caused_by": 1,            # Symptom/Disease -> Pathophysiological mechanism
    "indicated_biomarker": 2,  # Disease  -> Lab / ECG / Imaging finding
    "treated_by": 3,           # Disease  -> Drug / Intervention
    "contraindicated_in": 4,   # Drug     -> Condition
    "complication_of": 5,      # Disease A -> Disease B
}

# Reverse-relation offset used to compile the abductive direction
# (symptom -> disease, biomarker -> disease, ...) into the same sparse store.
REVERSE_OFFSET = 100


# ---------------------------------------------------------------------------
# The clinical knowledge graph, as an ordered list of triples.  The order is
# the insertion order of the SDK builder, so rebuilding from this list yields
# the identical integer ids the SDK assigns.
#
# RESEARCH DEMO — ontology content adapted from standard internal-medicine
# teaching material; it is a graph, not a guideline.
# ---------------------------------------------------------------------------
_CLINICAL_TRIPLES: List[Tuple[str, str, str, str, str]] = [
    # 1. CARDIOLOGY
    ("Acute_STEMI", "presents_with", "Retrosternal_Crushing_Chest_Pain", "disease", "symptom"),
    ("Acute_STEMI", "presents_with", "Diaphoresis", "disease", "symptom"),
    ("Acute_STEMI", "presents_with", "Radiation_to_Left_Arm", "disease", "symptom"),
    ("Acute_STEMI", "presents_with", "Dyspnea", "disease", "symptom"),
    ("Acute_STEMI", "indicated_biomarker", "Elevated_Cardiac_Troponin_I", "disease", "biomarker"),
    ("Acute_STEMI", "indicated_biomarker", "ST_Segment_Elevation", "disease", "biomarker"),
    ("Acute_STEMI", "caused_by", "Coronary_Atherosclerotic_Plaque_Rupture", "disease", "pathophysiology"),
    ("Coronary_Atherosclerotic_Plaque_Rupture", "caused_by", "Acute_Occlusive_Thrombosis", "pathophysiology", "pathophysiology"),
    ("Acute_Occlusive_Thrombosis", "caused_by", "Transmural_Myocardial_Ischemia", "pathophysiology", "pathophysiology"),
    ("Transmural_Myocardial_Ischemia", "caused_by", "Myocyte_Coagulative_Necrosis", "pathophysiology", "pathophysiology"),
    ("Myocyte_Coagulative_Necrosis", "caused_by", "Elevated_Cardiac_Troponin_I", "pathophysiology", "biomarker"),
    ("Acute_STEMI", "treated_by", "Primary_Percutaneous_Coronary_Intervention", "disease", "drug"),
    ("Acute_STEMI", "treated_by", "Aspirin_Dual_Antiplatelet", "disease", "drug"),
    ("Acute_STEMI", "treated_by", "Unfractionated_Heparin", "disease", "drug"),
    ("Acute_STEMI", "treated_by", "High_Intensity_Atorvastatin", "disease", "drug"),

    ("Acute_Pericarditis", "presents_with", "Sharp_Pleuritic_Chest_Pain", "disease", "symptom"),
    ("Acute_Pericarditis", "presents_with", "Pain_Relieved_Sitting_Forward", "disease", "symptom"),
    ("Acute_Pericarditis", "presents_with", "Pericardial_Friction_Rub", "disease", "symptom"),
    ("Acute_Pericarditis", "indicated_biomarker", "Diffuse_Concave_ST_Elevation", "disease", "biomarker"),
    ("Acute_Pericarditis", "indicated_biomarker", "PR_Segment_Depression", "disease", "biomarker"),
    ("Acute_Pericarditis", "caused_by", "Pericardial_Serofibrinous_Inflammation", "disease", "pathophysiology"),
    ("Acute_Pericarditis", "treated_by", "Colchicine", "disease", "drug"),
    ("Acute_Pericarditis", "treated_by", "High_Dose_Ibuprofen", "disease", "drug"),

    ("Aortic_Dissection", "presents_with", "Tearing_Interscapular_Back_Pain", "disease", "symptom"),
    ("Aortic_Dissection", "presents_with", "Pulse_Deficit_Between_Arms", "disease", "symptom"),
    ("Aortic_Dissection", "presents_with", "Severe_Refractory_Hypertension", "disease", "symptom"),
    ("Aortic_Dissection", "indicated_biomarker", "Mediastinal_Widening_on_CXR", "disease", "biomarker"),
    ("Aortic_Dissection", "indicated_biomarker", "Intimal_Tear_on_CT_Angiography", "disease", "biomarker"),
    ("Aortic_Dissection", "caused_by", "Cystic_Medial_Necrosis_and_Intimal_Tear", "disease", "pathophysiology"),
    ("Aortic_Dissection", "treated_by", "Intravenous_Esmolol_Beta_Blockade", "disease", "drug"),
    ("Aortic_Dissection", "treated_by", "Emergency_Surgical_Graft_Repair", "disease", "drug"),
    ("Thrombolysis_tPA", "contraindicated_in", "Aortic_Dissection", "drug", "disease"),

    # 2. PULMONOLOGY
    ("Pulmonary_Embolism", "presents_with", "Sudden_Onset_Dyspnea", "disease", "symptom"),
    ("Pulmonary_Embolism", "presents_with", "Pleuritic_Chest_Pain", "disease", "symptom"),
    ("Pulmonary_Embolism", "presents_with", "Hemoptysis", "disease", "symptom"),
    ("Pulmonary_Embolism", "presents_with", "Tachycardia", "disease", "symptom"),
    ("Pulmonary_Embolism", "presents_with", "Unilateral_Calf_Swelling_DVT", "disease", "symptom"),
    ("Pulmonary_Embolism", "indicated_biomarker", "Markedly_Elevated_D_Dimer", "disease", "biomarker"),
    ("Pulmonary_Embolism", "indicated_biomarker", "CT_Pulmonary_Angiogram_Filling_Defect", "disease", "biomarker"),
    ("Pulmonary_Embolism", "indicated_biomarker", "S1Q3T3_Pattern_on_ECG", "disease", "biomarker"),
    ("Deep_Vein_Thrombosis", "complication_of", "Pulmonary_Embolism", "disease", "disease"),
    ("Pulmonary_Embolism", "caused_by", "Thrombus_Embolization_to_Pulmonary_Artery", "disease", "pathophysiology"),
    ("Thrombus_Embolization_to_Pulmonary_Artery", "caused_by", "Ventilation_Perfusion_Mismatch", "pathophysiology", "pathophysiology"),
    ("Ventilation_Perfusion_Mismatch", "caused_by", "Acute_Right_Ventricular_Strain", "pathophysiology", "pathophysiology"),
    ("Pulmonary_Embolism", "treated_by", "Low_Molecular_Weight_Heparin", "disease", "drug"),
    ("Pulmonary_Embolism", "treated_by", "Direct_Oral_Anticoagulants_Apixaban", "disease", "drug"),
    ("Pulmonary_Embolism", "treated_by", "Catheter_Directed_Thrombolysis", "disease", "drug"),

    ("Bacterial_Pneumonia", "presents_with", "Fever_with_Rigors", "disease", "symptom"),
    ("Bacterial_Pneumonia", "presents_with", "Productive_Cough_with_Rusty_Sputum", "disease", "symptom"),
    ("Bacterial_Pneumonia", "presents_with", "Bronchial_Breath_Sounds_and_Crackles", "disease", "symptom"),
    ("Bacterial_Pneumonia", "indicated_biomarker", "Lobar_Consolidation_on_Chest_XRay", "disease", "biomarker"),
    ("Bacterial_Pneumonia", "indicated_biomarker", "Elevated_C_Reactive_Protein_and_PCT", "disease", "biomarker"),
    ("Bacterial_Pneumonia", "indicated_biomarker", "Neutrophilic_Leukocytosis", "disease", "biomarker"),
    ("Bacterial_Pneumonia", "caused_by", "Streptococcus_Pneumoniae_Alveolar_Infiltration", "disease", "pathophysiology"),
    ("Bacterial_Pneumonia", "treated_by", "Ceftriaxone_plus_Azithromycin", "disease", "drug"),

    ("Tension_Pneumothorax", "presents_with", "Hyperresonance_to_Percussion", "disease", "symptom"),
    ("Tension_Pneumothorax", "presents_with", "Absent_Unilateral_Breath_Sounds", "disease", "symptom"),
    ("Tension_Pneumothorax", "presents_with", "Tracheal_Deviation_to_Contralateral_Side", "disease", "symptom"),
    ("Tension_Pneumothorax", "presents_with", "Severe_Hypotension_and_Shock", "disease", "symptom"),
    ("Tension_Pneumothorax", "caused_by", "One_Way_Valve_Air_Entrapment", "disease", "pathophysiology"),
    ("One_Way_Valve_Air_Entrapment", "caused_by", "Impaired_Venous_Return_Preload_Collapse", "pathophysiology", "pathophysiology"),
    ("Tension_Pneumothorax", "treated_by", "Immediate_Needle_Decompression_2nd_ICS", "disease", "drug"),
    ("Tension_Pneumothorax", "treated_by", "Tube_Thoracostomy_Intercostal_Drain", "disease", "drug"),

    # 3. GASTROENTEROLOGY & ACUTE SURGERY
    ("Acute_Appendicitis", "presents_with", "Periumbilical_Pain_Migrating_to_RLQ", "disease", "symptom"),
    ("Acute_Appendicitis", "presents_with", "Tenderness_at_McBurney_Point", "disease", "symptom"),
    ("Acute_Appendicitis", "presents_with", "Positive_Rovsing_and_Psoas_Signs", "disease", "symptom"),
    ("Acute_Appendicitis", "presents_with", "Low_Grade_Fever_and_Anorexia", "disease", "symptom"),
    ("Acute_Appendicitis", "indicated_biomarker", "Non_Compressible_Appendix_USG_over_6mm", "disease", "biomarker"),
    ("Acute_Appendicitis", "indicated_biomarker", "Target_Sign_on_Abdominal_CT", "disease", "biomarker"),
    ("Acute_Appendicitis", "caused_by", "Appendiceal_Lumen_Obstruction_by_Fecalith", "disease", "pathophysiology"),
    ("Appendiceal_Lumen_Obstruction_by_Fecalith", "caused_by", "Intraluminal_Bacterial_Overgrowth_and_Ischemia", "pathophysiology", "pathophysiology"),
    ("Intraluminal_Bacterial_Overgrowth_and_Ischemia", "caused_by", "Gangrenous_Transmural_Perforation", "pathophysiology", "pathophysiology"),
    ("Acute_Appendicitis", "treated_by", "Emergency_Laparoscopic_Appendectomy", "disease", "drug"),
    ("Acute_Appendicitis", "treated_by", "Intravenous_Ceftriaxone_Metronidazole", "disease", "drug"),

    ("Acute_Pancreatitis", "presents_with", "Epigastric_Pain_Radiating_to_Back", "disease", "symptom"),
    ("Acute_Pancreatitis", "presents_with", "Relief_on_Leaning_Forward", "disease", "symptom"),
    ("Acute_Pancreatitis", "presents_with", "Nausea_and_Bilious_Vomiting", "disease", "symptom"),
    ("Acute_Pancreatitis", "indicated_biomarker", "Serum_Lipase_Greater_Than_3x_Normal", "disease", "biomarker"),
    ("Acute_Pancreatitis", "indicated_biomarker", "Peripancreatic_Fluid_Collections_on_CT", "disease", "biomarker"),
    ("Acute_Pancreatitis", "caused_by", "Premature_Intraacinar_Trypsinogen_Activation", "disease", "pathophysiology"),
    ("Premature_Intraacinar_Trypsinogen_Activation", "caused_by", "Autodigestion_and_Fat_Necrosis", "pathophysiology", "pathophysiology"),
    ("Acute_Pancreatitis", "treated_by", "Aggressive_Intravenous_Ringer_Lactate", "disease", "drug"),
    ("Acute_Pancreatitis", "treated_by", "Early_Enteral_Nutrition", "disease", "drug"),

    # 4. ENDOCRINOLOGY & METABOLISM
    ("Diabetic_Ketoacidosis", "presents_with", "Kussmaul_Deep_Rapid_Breathing", "disease", "symptom"),
    ("Diabetic_Ketoacidosis", "presents_with", "Fruity_Acetone_Breath_Odor", "disease", "symptom"),
    ("Diabetic_Ketoacidosis", "presents_with", "Polyuria_Polydipsia_Dehydration", "disease", "symptom"),
    ("Diabetic_Ketoacidosis", "presents_with", "Abdominal_Pain_and_Altered_Sensorium", "disease", "symptom"),
    ("Diabetic_Ketoacidosis", "indicated_biomarker", "Hyperglycemia_over_250mg_dL", "disease", "biomarker"),
    ("Diabetic_Ketoacidosis", "indicated_biomarker", "High_Anion_Gap_Metabolic_Acidosis_pH_under_7_3", "disease", "biomarker"),
    ("Diabetic_Ketoacidosis", "indicated_biomarker", "Positive_Serum_Beta_Hydroxybutyrate", "disease", "biomarker"),
    ("Diabetic_Ketoacidosis", "caused_by", "Absolute_Insulin_Deficiency", "disease", "pathophysiology"),
    ("Absolute_Insulin_Deficiency", "caused_by", "Unchecked_Lipolysis_and_Free_Fatty_Acids", "pathophysiology", "pathophysiology"),
    ("Unchecked_Lipolysis_and_Free_Fatty_Acids", "caused_by", "Hepatic_Ketogenesis_Acetoacetate_BetaHB", "pathophysiology", "pathophysiology"),
    ("Diabetic_Ketoacidosis", "treated_by", "Continuous_IV_Regular_Insulin_Infusion", "disease", "drug"),
    ("Diabetic_Ketoacidosis", "treated_by", "Intravenous_Isotonic_0_9_Saline_Hydration", "disease", "drug"),
    ("Diabetic_Ketoacidosis", "treated_by", "Potassium_Replacement_Protocol", "disease", "drug"),

    # 5. NEUROLOGY
    ("Bacterial_Meningitis", "presents_with", "Fever_and_Chills", "disease", "symptom"),
    ("Bacterial_Meningitis", "presents_with", "Severe_Throbbing_Headache", "disease", "symptom"),
    ("Bacterial_Meningitis", "presents_with", "Nuchal_Rigidity_Neck_Stiffness", "disease", "symptom"),
    ("Bacterial_Meningitis", "presents_with", "Positive_Kernig_and_Brudzinski_Signs", "disease", "symptom"),
    ("Bacterial_Meningitis", "presents_with", "Photophobia", "disease", "symptom"),
    ("Bacterial_Meningitis", "indicated_biomarker", "CSF_Neutrophilic_Pleocytosis", "disease", "biomarker"),
    ("Bacterial_Meningitis", "indicated_biomarker", "Markedly_Decreased_CSF_Glucose_under_40mg", "disease", "biomarker"),
    ("Bacterial_Meningitis", "indicated_biomarker", "Elevated_CSF_Protein_over_100mg", "disease", "biomarker"),
    ("Bacterial_Meningitis", "caused_by", "Hematogenous_Seeding_to_Subarachnoid_Space", "disease", "pathophysiology"),
    ("Hematogenous_Seeding_to_Subarachnoid_Space", "caused_by", "Meningeal_Neutrophilic_Inflammation_Exudate", "pathophysiology", "pathophysiology"),
    ("Bacterial_Meningitis", "treated_by", "IV_Ceftriaxone_plus_Vancomycin", "disease", "drug"),
    ("Bacterial_Meningitis", "treated_by", "Adjuvant_IV_Dexamethasone", "disease", "drug"),

    ("Acute_Ischemic_Stroke_MCA", "presents_with", "Contralateral_Hemiparesis_Face_Arm_over_Leg", "disease", "symptom"),
    ("Acute_Ischemic_Stroke_MCA", "presents_with", "Expressive_or_Receptive_Aphasia", "disease", "symptom"),
    ("Acute_Ischemic_Stroke_MCA", "presents_with", "Facial_Droop", "disease", "symptom"),
    ("Acute_Ischemic_Stroke_MCA", "presents_with", "Dysarthria_Slurred_Speech", "disease", "symptom"),
    ("Acute_Ischemic_Stroke_MCA", "indicated_biomarker", "Diffusion_Restriction_on_Brain_MRI", "disease", "biomarker"),
    ("Acute_Ischemic_Stroke_MCA", "indicated_biomarker", "Hyperdense_MCA_Sign_on_NonContrast_CT", "disease", "biomarker"),
    ("Acute_Ischemic_Stroke_MCA", "caused_by", "Thromboembolic_Occlusion_of_Middle_Cerebral_Artery", "disease", "pathophysiology"),
    ("Thromboembolic_Occlusion_of_Middle_Cerebral_Artery", "caused_by", "Ischemic_Penumbra_and_Cytotoxic_Edema", "pathophysiology", "pathophysiology"),
    ("Acute_Ischemic_Stroke_MCA", "treated_by", "Intravenous_Alteplase_tPA_under_4_5h", "disease", "drug"),
    ("Acute_Ischemic_Stroke_MCA", "treated_by", "Endovascular_Mechanical_Thrombectomy", "disease", "drug"),
]


class MedicalOntology:
    """
    Medical entity vocabulary plus multi-relational edges.

    Entities receive integer ids in first-appearance order, exactly as the SDK's
    ``MedicalOntology`` does, so the two stores are index-compatible.
    """

    def __init__(self) -> None:
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.entity_type: Dict[str, str] = {}

        self.relation2id: Dict[str, int] = dict(RELATION_IDS)
        self.id2relation: Dict[int, str] = {v: k for k, v in self.relation2id.items()}

        # rel_id -> list of (source_id, target_id)
        self.edges_by_rel: Dict[int, List[Tuple[int, int]]] = {
            r: [] for r in self.relation2id.values()
        }

    def add_entity(self, name: str, etype: str) -> int:
        if name not in self.entity2id:
            eid = len(self.entity2id)
            self.entity2id[name] = eid
            self.id2entity[eid] = name
            self.entity_type[name] = etype
        return self.entity2id[name]

    def add_triple(
        self,
        head: str,
        rel: str,
        tail: str,
        head_type: str = "concept",
        tail_type: str = "concept",
    ) -> None:
        hid = self.add_entity(head, head_type)
        tid = self.add_entity(tail, tail_type)
        self.edges_by_rel[self.relation2id[rel]].append((hid, tid))

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    def entities_of_type(self, etype: str) -> List[str]:
        return [name for name, t in self.entity_type.items() if t == etype]

    def census(self) -> Dict[str, object]:
        """A deterministic census of the ontology (counts only)."""
        type_counts = Counter(self.entity_type.values())
        rel_counts = {
            self.id2relation[r]: len(self.edges_by_rel[r])
            for r in sorted(self.edges_by_rel)
        }
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_edges": sum(rel_counts.values()),
            "entities_by_type": dict(type_counts),
            "edges_by_relation": rel_counts,
        }


def build_clinical_knowledge_graph() -> MedicalOntology:
    """
    Construct the clinical knowledge graph.

    Reproduces ``geoformer.clinical.ontology.build_clinical_knowledge_graph``:
    128 entities across 5 types and 118 directed edges across 6 relations.
    **RESEARCH DEMO** — not a clinical resource.
    """
    ont = MedicalOntology()
    for head, rel, tail, head_type, tail_type in _CLINICAL_TRIPLES:
        ont.add_triple(head, rel, tail, head_type, tail_type)
    return ont


class ClinicalDiagnosticEngine:
    """
    Compile a clinical ontology into a :class:`SparseHippocampalMemory` and run
    relational clinical inference on it.

    The compile step registers, for every relation ``r``, both the forward
    conductance (``src -> tgt``) and the reverse conductance under id
    ``100 + r`` (``tgt -> src``).  The forward direction carries deductive
    queries (disease -> treatment, disease -> mechanism); the reverse direction
    carries the abductive queries that a differential needs (finding -> disease).

    **RESEARCH DEMO.**  All outputs are graph-propagation scores, not diagnoses.
    """

    def __init__(
        self,
        ontology: Optional[MedicalOntology] = None,
        d_model: int = 64,
        mode: str = "fp32",
    ) -> None:
        self.ont = ontology if ontology is not None else build_clinical_knowledge_graph()
        self.d_model = d_model
        self.mode = mode
        self.num_entities = self.ont.num_entities

        config = SparseMemoryConfig(
            num_entities=self.num_entities, d_model=d_model, mode=mode
        )
        self.memory = SparseHippocampalMemory(config)
        self._compile_ontology_into_relational_memory()

    # -- compilation --------------------------------------------------------
    def _compile_ontology_into_relational_memory(self) -> None:
        """Compile forward and reverse conductances into the sparse store."""
        for r_id, edge_list in self.ont.edges_by_rel.items():
            if not edge_list:
                continue
            src_nodes = [e[0] for e in edge_list]
            tgt_nodes = [e[1] for e in edge_list]

            # Forward conductance: source -> target.
            edge_index_fwd = torch.tensor([tgt_nodes, src_nodes], dtype=torch.long)
            self.memory.register_sparse_relation(r_id, edge_index_fwd)

            # Reverse conductance (abduction): target -> source.
            rev_r_id = REVERSE_OFFSET + r_id
            edge_index_rev = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
            self.memory.register_sparse_relation(rev_r_id, edge_index_rev)

    @property
    def compiled_relations(self) -> List[int]:
        return sorted(self.memory.sparse_edges.keys())

    # -- the three inferences ----------------------------------------------
    def recommended_interventions(self, disease_name: str) -> List[str]:
        """Disease -> treated_by -> Drug, in edge order."""
        dis_id = self.ont.entity2id[disease_name]
        rel = self.ont.relation2id["treated_by"]
        return [
            self.ont.id2entity[tid]
            for hid, tid in self.ont.edges_by_rel[rel]
            if hid == dis_id
        ]

    def trace_causal_path(self, disease_name: str, max_depth: int = 4) -> List[str]:
        """
        Walk the ``caused_by`` chain from a disease.

        Returns the ordered node list ``[disease, mechanism_1, ...]``; its
        length minus one is the hop depth of the traced pathophysiology.
        """
        dis_id = self.ont.entity2id[disease_name]
        rel = self.ont.relation2id["caused_by"]
        chain = [disease_name]
        curr_id = dis_id
        for _ in range(max_depth):
            next_id = None
            for hid, tid in self.ont.edges_by_rel[rel]:
                if hid == curr_id:
                    next_id = tid
                    break
            if next_id is None:
                break
            chain.append(self.ont.id2entity[next_id])
            curr_id = next_id
        return chain

    def check_contraindications(self, disease_name: str) -> List[str]:
        """Drug -> contraindicated_in -> Disease, reported as safety alerts."""
        dis_id = self.ont.entity2id[disease_name]
        rel = self.ont.relation2id["contraindicated_in"]
        alerts = []
        for hid, tid in self.ont.edges_by_rel[rel]:
            if tid == dis_id:
                alerts.append(
                    f"CRITICAL WARNING: {self.ont.id2entity[hid]} is strictly "
                    f"CONTRAINDICATED in {disease_name}!"
                )
        return alerts

    def _match_findings(
        self, observed_findings: List[str]
    ) -> Tuple[List[int], List[str]]:
        """Map free-text findings to entity ids (exact, then fuzzy fallback)."""
        matched_ids: List[int] = []
        unknown_findings: List[str] = []
        for finding in observed_findings:
            if finding in self.ont.entity2id:
                matched_ids.append(self.ont.entity2id[finding])
                continue
            found = False
            needle = finding.lower().replace(" ", "_")
            for ent_name, eid in self.ont.entity2id.items():
                if needle in ent_name.lower() or ent_name.lower() in needle:
                    matched_ids.append(eid)
                    found = True
                    break
            if not found:
                unknown_findings.append(finding)
        return matched_ids, unknown_findings

    def diagnose_patient(
        self, observed_findings: List[str], top_k: int = 3
    ) -> Dict:
        """
        Multi-hop differential diagnosis from observed signs and biomarkers.

        The scoring is the SDK's: a one-hop abductive push along the reverse
        ``presents_with`` (100) and reverse ``indicated_biomarker`` (102)
        relations, plus a two-hop causal resonance along the reverse
        ``caused_by`` relation (101) weighted at 0.6.  Disease scores are
        renormalized to percentages.

        **RESEARCH DEMO** — a graph score, not a diagnosis.
        """
        matched_ids, unknown_findings = self._match_findings(observed_findings)

        if not matched_ids:
            return {
                "error": "No recognizable clinical findings provided.",
                "unknown": unknown_findings,
            }

        h_symptom = torch.zeros(self.num_entities)
        share = 1.0 / len(matched_ids)
        for eid in matched_ids:
            h_symptom[eid] = share

        # One-hop abductive resonance: finding -> disease.
        h_disease_direct = (
            self.memory.propagate_sparse_relation(REVERSE_OFFSET + 0, h_symptom)
            + self.memory.propagate_sparse_relation(REVERSE_OFFSET + 2, h_symptom)
        )

        # Two-hop pathophysiological causal resonance: finding -> mechanism -> disease.
        h_patho = self.memory.propagate_sparse_relation(REVERSE_OFFSET + 1, h_symptom)
        h_disease_causal = self.memory.propagate_sparse_relation(
            REVERSE_OFFSET + 1, h_patho
        )

        h_total = h_disease_direct + 0.6 * h_disease_causal

        disease_scores: List[Tuple[str, float]] = []
        for ent_name, eid in self.ont.entity2id.items():
            if self.ont.entity_type.get(ent_name) == "disease":
                score = h_total[eid].item()
                if score > 0.0:
                    disease_scores.append((ent_name, score))

        disease_scores.sort(key=lambda x: -x[1])
        total_score_sum = sum(s[1] for s in disease_scores) + 1e-8

        differentials = []
        for name, raw_score in disease_scores[:top_k]:
            probability = (raw_score / total_score_sum) * 100.0
            differentials.append(
                {
                    "disease": name,
                    "probability": float(probability),
                    "confidence_score": float(raw_score),
                }
            )

        primary_disease = differentials[0]["disease"] if differentials else None

        treatments: List[str] = []
        contraindications: List[str] = []
        explanations: List[str] = []
        if primary_disease:
            treatments = self.recommended_interventions(primary_disease)
            contraindications = self.check_contraindications(primary_disease)
            chain = self.trace_causal_path(primary_disease)
            if len(chain) > 1:
                explanations.append(" -> ".join(chain))

        return {
            "differentials": differentials,
            "primary_diagnosis": primary_disease,
            "pathophysiological_causal_chain": explanations,
            "recommended_interventions": treatments,
            "safety_contraindications": contraindications,
            "matched_findings": [self.ont.id2entity[i] for i in matched_ids],
            "unknown_findings": unknown_findings,
        }

    def ontology_census(self) -> Dict[str, object]:
        return self.ont.census()
