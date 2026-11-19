<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 19 — Clinical Knowledge-Graph Integration (RESEARCH DEMO)

The sparse store of Chapter 18 removed the memory wall. What it has not yet done
is reason about anything a human being would recognize as knowledge. Every graph
we have run so far was synthetic — chains and crossroads generated to isolate a
mechanism. In this chapter we point the same machinery at a *structured domain*
and watch it infer.

The domain is clinical medicine, and the warning belongs at the top rather than
buried in a footnote: **this chapter is a RESEARCH DEMO.** Nothing here is a
medical device, a clinical decision support system, or medical advice. The
ontology is a small teaching graph; the "diagnoses" the engine prints are the
outputs of a sparse matrix propagation, not the judgements of a clinician. What
we are demonstrating is narrow and honest: that the relational memory of Part V
can ingest a real-shaped knowledge graph and carry out three relational
inferences — abductive differential ranking, deductive treatment lookup, and
safety checking — with the same O(|E|) machinery, and that it agrees, bit for
bit, with the production SDK engine.

The clinical graph is a good stress test precisely because it is relational
rather than tabular. A finding points to a disease; a disease points to a
mechanism; a mechanism points to another mechanism; a drug points to a condition
it must never meet. These are edges, and edges are what a conductance lattice
stores.

Everything printed in this chapter is the output of a task that runs from the
book's root as `python3 code/gftac/ch19_tasks.py N`, and every line is pinned by
`tests/test_ch19_tasks.py`. The structural claims are asserted by
`tests/test_ch19_clinical_kg.py`.

---

## 19.1 The Shape of Clinical Knowledge

An ontology is a vocabulary plus a set of typed, directed relations among its
terms. Ours has five entity types — **disease**, **symptom**, **biomarker**,
**pathophysiology** (a mechanism), and **drug** — and six relation types. The
relations are the object of study, so we write them down as a table.

| Relation | Meaning | Direction |
|---|---|---|
| `presents_with` | disease → symptom | deductive |
| `indicated_biomarker` | disease → lab / ECG / imaging finding | deductive |
| `treated_by` | disease → drug or intervention | deductive |
| `caused_by` | disease or mechanism → mechanism | causal |
| `contraindicated_in` | drug → condition | safety |
| `complication_of` | disease → disease | association |

Each relation is a set of directed arcs, and the graph as a whole is the union
of those arcs. This is exactly the object Chapter 18 taught us to store without
inverting it into an $N \times N$ array: a knowledge graph is a set of facts, and
a fact is an edge.

The graph is small — twelve diseases across five organ systems — but it is not
toy-shaped. Its arcs form the *paths* that clinical reasoning follows. From a
finding, the reverse of `presents_with` climbs to the diseases that present with
it; from a disease, `caused_by` descends a chain of mechanisms; from a drug,
`contraindicated_in` names the condition that forbids it. The whole engine is
built by reading these paths in the right directions.

We begin by counting what we have built. The census is a pure structural fact —
no propagation, no scores — and it is the artifact against which every later
claim about the graph can be checked.

**Run this now.** Task 1 (`python3 code/gftac/ch19_tasks.py 1`):

```python
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
```

```text
Clinical Knowledge Graph Census — RESEARCH DEMO:
  • Entities:                       128
  • Relation Types:                 6
  • Directed Edges:                 118
  Entities by Type:
      symptom                  42
      drug                     26
      biomarker                24
      pathophysiology          24
      disease                  12
  Edges by Relation:
      presents_with            42
      caused_by                25
      indicated_biomarker      24
      treated_by               25
      contraindicated_in       1
      complication_of          1
```

**Interpretation.** The graph holds $128$ entities and $118$ directed edges.
The types are distributed as the domain demands: symptoms are the most numerous
($42$), because many diseases share them; drugs and mechanisms are next
($26$ and $24$); diseases are the fewest ($12$), because the whole point of a
differential is that many findings collapse onto a few candidate diagnoses. The
relation counts are equally legible: $42$ `presents_with` arcs, $25$ `caused_by`
arcs, $24$ `indicated_biomarker`, $25$ `treated_by`, and a single safety arc —
the one we will fire in §19.4. These are **MEASURED** counts of the built
artifact, and they are asserted exactly in the chapter's tests.

---

## 19.2 The Compile: Facts into Conductances

To reason on the graph we compile it into the sparse hippocampal memory of
Chapter 18. The compile is where the clinical logic lives, and it is worth
stating precisely.

Each relation $r$ contributes a set of arcs $(u, v, w)$ — a source, a target, and
a weight, here always $1$. The memory stores the *forward* conductance
$W_r$ with the Chapter 18 convention: the arc $u \to v$ becomes the coordinate
$(v, u)$ with value $w / \deg_{\text{out}}(u)$, so that $W_r h$ pushes mass from
sources to targets and each column sums to one. Deductive queries run forward on
this store.

But a differential is an *abductive* query: it starts from findings and climbs to
diseases. Running the forward `presents_with` store would push disease mass onto
symptoms — the wrong way. So the compile also registers, for every relation $r$,
a **reverse** store under the id $100 + r$, built from the same arcs read
backwards: the arc $u \to v$ becomes a reverse arc $v \to u$, normalized by the
reverse source degree. The reverse store is what makes symptom $\to$ disease,
biomarker $\to$ disease, and mechanism $\to$ disease all one-hop operations.

The first thing to prove is that the two stores carry the same facts.

> **Theorem B.19.1 (Compile support and the abductive mirror).**
> Compile the arcs $E_r = \{(u,v)\}$ of relation $r$ into a forward store $W_r$
> and a reverse store $W_{100+r}$. Then for every ordered pair $(a,b)$,
> $$
> W_r[b,a] > 0 \iff (a,b) \in E_r
> \qquad\text{and}\qquad
> W_{100+r}[a,b] > 0 \iff (a,b) \in E_r .
> $$
> Consequently the nonzero support of $W_{100+r}$ is the transpose of the
> support of $W_r$: an arc exists in the abductive store exactly when the same
> fact exists in the deductive store, read in the opposite direction. Moreover
> every column of each store sums to at most one.

**Proof.** For the forward store, the arc $(u,v) \in E_r$ is registered at
coordinate $(v,u)$ with value $w / \deg_{\text{out}}(u)$, where
$\deg_{\text{out}}(u) = \sum_{u' : (u,u') \in E_r} w(u,u') > 0$. So
$W_r[v,u] = w/\deg_{\text{out}}(u) > 0$, and it is zero whenever $(u,v) \notin
E_r$; coalescing sums duplicates but preserves the sign, so the forward
biconditional holds.

For the reverse store, the same arc $(u,v)$ is registered at coordinate $(a,b) =
(u,v)$ — the endpoints swapped — with value $w / \deg_{\text{in}}(v)$, where
$\deg_{\text{in}}(v) = \sum_{u' : (u',v) \in E_r} w(u',v) > 0$. So
$W_{100+r}[u,v] > 0$ iff $(u,v) \in E_r$, which is the reverse biconditional.
Comparing the two: $W_{100+r}[a,b] > 0 \iff (a,b) \in E_r \iff W_r[b,a] > 0$,
which is the support-transpose statement.

For the column sums, fix a source node $a$. In $W_r$, column $a$ holds the
forward entries $w(a,v)/\deg_{\text{out}}(a)$ over the out-arcs of $a$, so it sums
to $\deg_{\text{out}}(a)/\deg_{\text{out}}(a) = 1$ when $a$ has an out-arc and
$0$ otherwise. In $W_{100+r}$, column $a$ holds the reverse entries
$w(u,a)/\deg_{\text{in}}(a)$ over the in-arcs of $a$, summing to
$\deg_{\text{in}}(a)/\deg_{\text{in}}(a) = 1$ or $0$. Hence every column sums to at
most one. $\blacksquare$

The theorem is what licenses the differential to read an abductive answer off the
reverse store: the reverse support is the forward support transposed, so no fact
is invented and none is lost. The memory's own `assert_invariants` — non-negativity
and finite energy — holds on both stores, which is the numerical form of the same
statement.

**Run this now.** Task 2 (`python3 code/gftac/ch19_tasks.py 2`):

```python
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
```

```text
Compiling the Clinical Ontology into Sparse Hippocampal Memory — RESEARCH DEMO:
  • Entities:                       128
  • Forward Relations:              6
  • Reverse Relations:              6
  • Stored Nonzeros:                236
  • Sparse Physical Footprint:      4,720 bytes (0.0045 MB)
  • Dense Equivalent:               786,432 bytes (0.75 MB)
  • Compression Factor:             166.6x
  • Invariant Battery Audit:        PASSED
```

**Interpretation.** The compile produces twelve stores: six forward and six
reverse, one per relation in each direction, exactly as Theorem B.19.1
describes. The stored nonzero count is $236 = 2 \times 118$: the graph's $118$
arcs, each present once in the forward store and once in the reverse, with no
coordinate collisions to coalesce away. The sparse footprint is $236 \times 20 =
4{,}720$ bytes, the Chapter 18 law of $20$ bytes per stored COO nonzero, against
a dense equivalent of $786{,}432$ bytes — a factor of $166.6\times$.

We should be honest about the size of that factor. It is far smaller than the
$100{,}000\times$ of Chapter 18, and it must be: compression is a property of a
graph's sparsity, and this graph is *dense in its entities and tiny in its
extent*. With only $128$ nodes, the dense array is barely three quarters of a
megabyte to begin with. The number that matters is not the ratio but the fact
that the engine's per-hop cost is still $O(|E|)$, so the same code would carry a
hundred-thousand-node clinical graph at the same cost per fact. The invariant
battery passes, which is the load-bearing line: the stores the differential is
about to read are non-negative and of finite energy.

---

## 19.3 Differential Diagnosis

A differential is the answer to a single question: given a set of observed
findings, which diseases best explain them? On the compiled store this is a
short propagation. The engine activates the observed findings uniformly, then
adds two resonances:

1. **Abductive resonance** — one hop along the reverse `presents_with` store
   ($100$) plus one hop along the reverse `indicated_biomarker` store ($102$).
   This is the direct evidence: a disease accumulates mass in proportion to how
   many of the observed findings it presents with or indicates.
2. **Causal resonance** — two hops along the reverse `caused_by` store ($101$):
   finding $\to$ mechanism $\to$ disease, weighted at $0.6$. This lets a
   mechanism that explains several findings reinforce the disease it belongs to.

The total resonance is the sum of the two. The engine then keeps only entities of
type **disease**, sorts by score, and renormalizes to percentages. The next
theorem fixes the arithmetic of that renormalization, because the exact form
matters for reading the printed numbers.

> **Theorem B.19.2 (Differential normalization and ranking).**
> Let the positive disease resonance scores be $d_1 \ge d_2 \ge \cdots \ge d_m >
> 0$ and let $S = \sum_{j=1}^{m} d_j$ and $\varepsilon > 0$. The engine reports
> $p_i = 100\, d_i / (S + \varepsilon)$. Then
> $$
> 0 < p_i < 100, \qquad p_1 = \max_i p_i, \qquad
> \sum_{i=1}^{m} p_i = \frac{100\,S}{S + \varepsilon} < 100 .
> $$
> In particular the reported list is sorted in decreasing order, its first entry
> is the maximum-resonance disease, and its entries sum to just under one
> hundred percent, the deficit being exactly the regularization
> $100\,\varepsilon/(S+\varepsilon)$.

**Proof.** Since every $d_i > 0$ and $\varepsilon > 0$, the denominator
$S + \varepsilon > 0$, so $p_i > 0$. For the upper bound,
$p_i = 100\,d_i/(S+\varepsilon) < 100\,d_i/S \le 100$ because $d_i \le S$ and
$\varepsilon > 0$ makes the denominator strictly larger than $S$. The map
$d \mapsto 100\,d/(S+\varepsilon)$ is strictly increasing in $d$, so the largest
score $d_1$ yields the largest probability $p_1$. Finally
$\sum_i p_i = 100\sum_i d_i/(S+\varepsilon) = 100\,S/(S+\varepsilon) < 100$,
and the deficit is $100 - 100S/(S+\varepsilon) = 100\varepsilon/(S+\varepsilon)$.
$\blacksquare$

The theorem explains a small numerical artifact the reader will notice in the
transcript: a unanimous differential prints as $99.999999\%$, not $100\%$,
because the regularization $\varepsilon = 10^{-8}$ is added to the denominator
and Theorem B.19.2 says the entries sum to exactly
$100\varepsilon/(S+\varepsilon)$ short of one hundred. The deficit is not an
error; it is the theorem.

We run the engine on the canonical emergency that a differential exists to
separate: a patient with crushing retrosternal chest pain, diaphoresis, and ST
elevation — a textbook ST-elevation myocardial infarction — who also reports
tearing interscapular back pain, the finding that raises aortic dissection as a
competitor.

**Run this now.** Task 3 (`python3 code/gftac/ch19_tasks.py 3`):

```python
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
```

```text
Differential Diagnosis — Canonical STEMI vs Aortic Dissection (RESEARCH DEMO):
  Observed findings (4): Retrosternal_Crushing_Chest_Pain, Diaphoresis, ST_Segment_Elevation, Tearing_Interscapular_Back_Pain
  Ranked differentials:
    1. Acute_STEMI                 75.00%  (raw score 0.750000)
    2. Aortic_Dissection           25.00%  (raw score 0.250000)
  Primary diagnosis:   Acute_STEMI
  Causal chain:        Acute_STEMI -> Coronary_Atherosclerotic_Plaque_Rupture -> Acute_Occlusive_Thrombosis -> Transmural_Myocardial_Ischemia -> Myocyte_Coagulative_Necrosis
  Recommended (DEMO):  Primary_Percutaneous_Coronary_Intervention, Aspirin_Dual_Antiplatelet, Unfractionated_Heparin, High_Intensity_Atorvastatin
```

**Interpretation.** The engine ranks **Acute_STEMI** first at $75.00\%$, with
**Aortic_Dissection** second at $25.00\%$. The split is exactly what the
arithmetic predicts: three of the four findings — the crushing pain, the
diaphoresis, and the ST elevation — are STEMI findings, and one — the tearing
back pain — is a dissection finding, so the STEMI evidence outnumbers the
dissection evidence three to one, and the raw scores are $0.75$ and $0.25$. The
two scores sum to $1$, so after the $\varepsilon$-regularized normalization of
Theorem B.19.2 the reported probabilities are $75.00\%$ and $25.00\%$. The
primary diagnosis drives two deductive readouts: the causal chain descends the
`caused_by` arcs from the plaque rupture to the coagulative necrosis, and the
recommended-intervention list is the `treated_by` neighbourhood of the primary.
Both are **RESEARCH DEMO** outputs — the graph's edges read back, not a
clinician's plan. The ranking, the $75/25$ split, and the top-1 identity are
asserted by `tests/test_ch19_clinical_kg.py`.

---

## 19.4 Contraindication Safety Checking

The third inference is the one with the sharpest domain meaning. Some facts are
not diagnostic but *prohibitive*: a drug that must never be given in a condition.
The ontology encodes exactly one such fact — `Thrombolysis_tPA` is
`contraindicated_in` `Aortic_Dissection` — and it is enough to demonstrate the
mechanism, because the check is a lookup along a single relation.

The engine exposes it directly. Given a disease, `check_contraindications`
scans the `contraindicated_in` arcs for those whose target is that disease and
emits one alert per match. The next theorem says the alert is neither spurious
nor missing.

> **Theorem B.19.3 (Contraindication soundness).**
> For any disease $x$, the alert list $A(x)$ returned by
> `check_contraindications(x)` is nonempty if and only if there exists a drug
> $g$ with $(g, x) \in E_{\text{contra}}$, the contraindication arc set. When
> nonempty, each alert names exactly the drug $g$ and the disease $x$ of one such
> arc, and the alerts appear in arc order.

**Proof.** The procedure iterates once over $E_{\text{contra}}$, appending an
alert precisely when the arc's target id equals the id of $x$. Hence $A(x)$ is
the image of the set $\{g : (g,x) \in E_{\text{contra}}\}$ under the alert
formatting, and $A(x) = \varnothing$ exactly when that set is empty. Each
appended alert is a string built from the current arc's head and the queried
disease, so it names exactly that pair, and iteration in list order preserves arc
order. $\blacksquare$

The check is deliberately *narrow*: it fires on the primary diagnosis's own
contraindications, not on every disease that appeared in the differential. That
is the same rule the SDK engine uses, and §19.6 shows the two agree. The
narrowness is a feature of this demo, not a clinical recommendation — a real
system would check every therapy against every active differential, and we say so
plainly.

**Run this now.** Task 4 (`python3 code/gftac/ch19_tasks.py 4`):

```python
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
```

```text
Contraindication Safety Check — RESEARCH DEMO:
  Case A (dissection findings) -> primary Aortic_Dissection
    Alert fired:   True
    Alert:         CRITICAL WARNING: Thrombolysis_tPA is strictly CONTRAINDICATED in Aortic_Dissection!
  Case B (STEMI findings)      -> primary Acute_STEMI
    Alert fired:   False
    Alert:         none
```

**Interpretation.** Case A resolves to **Aortic_Dissection**, and the alert
fires: `Thrombolysis_tPA` is named as strictly contraindicated in that
condition. Case B resolves to **Acute_STEMI**, and the alert does not fire — the
one contraindication arc in the graph points at dissection, not at STEMI, so
Theorem B.19.3 says the alert set is empty and the transcript prints `none`. The
pair is the point: an alert that fires on everything is noise, and the test is
that it fires on exactly the disease the graph forbids and stays silent
otherwise. This is a **RESEARCH DEMO** of an edge lookup; it is not clinical
guidance, and the "CRITICAL WARNING" string is the demo's formatting, not an
instruction.

---

## 19.5 Multi-Hop Causal Path Tracing

The last inference is the descent. From a disease, the `caused_by` relation walks
a chain of pathophysiological mechanisms — the graph's model of *why* the
disease produces its findings. Tracing the chain is a walk along a relation, and
its well-definedness depends on the relation's structure.

> **Theorem B.19.4 (Causal trace termination and depth).**
> Suppose the `caused_by` relation $E_{\text{causal}}$ is a *functional acyclic
> digraph*: every head has out-degree at most one, and $E_{\text{causal}}$
> contains no directed cycle. Then for every start $x$ and budget $D \ge 0$,
> `trace_causal_path(x, D)` returns a simple path
> $x = u_0 \to u_1 \to \cdots \to u_k$ with $k \le D$, it terminates after at
> most $\min(D, L)$ hops where $L$ is the length of the maximal `caused_by` path
> from $x$, and it contains no repeated vertex.

**Proof.** Each iteration of the trace looks for the unique out-arc of the
current node $u_i$; if none exists the loop breaks, and otherwise it advances to
the unique successor $u_{i+1}$ and consumes one unit of the budget $D$. After
$D$ iterations the budget is exhausted, so the walk advances at most $D$ times,
giving $k \le D$. If the walk ever revisited a vertex, the segment between the
two visits would be a directed cycle in $E_{\text{causal}}$, contradicting
acyclicity; hence every vertex is distinct and the path is simple. If the walk
leaves $x$ along the maximal path, it stops when that path ends, after exactly
$L$ hops, so the hop count is $\min(D, L)$. $\blacksquare$

The structural hypothesis is not an assumption we take on faith: on this
ontology, every one of the $25$ `caused_by` heads has exactly one out-arc and
there is no directed cycle, so the relation is a functional acyclic digraph and
the theorem applies to every disease. The chapter's tests assert both facts
directly.

The budget matters. The engine's default is $D = 4$, and one chain saturates it
exactly: the STEMI chain from `Acute_STEMI` reaches a mechanism at depth four and
the budget stops the walk there, even though the graph continues one more arc to
the troponin biomarker. We therefore predict, before running, that the longest
chain printed will be depth four and that it will end at
`Myocyte_Coagulative_Necrosis`.

**Run this now.** Task 5 (`python3 code/gftac/ch19_tasks.py 5`):

```python
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
```

```text
Multi-Hop Pathophysiological Causal Trace — RESEARCH DEMO:
  Acute_STEMI                  depth 4  | Acute_STEMI -> Coronary_Atherosclerotic_Plaque_Rupture -> Acute_Occlusive_Thrombosis -> Transmural_Myocardial_Ischemia -> Myocyte_Coagulative_Necrosis
  Acute_Appendicitis           depth 3  | Acute_Appendicitis -> Appendiceal_Lumen_Obstruction_by_Fecalith -> Intraluminal_Bacterial_Overgrowth_and_Ischemia -> Gangrenous_Transmural_Perforation
  Diabetic_Ketoacidosis        depth 3  | Diabetic_Ketoacidosis -> Absolute_Insulin_Deficiency -> Unchecked_Lipolysis_and_Free_Fatty_Acids -> Hepatic_Ketogenesis_Acetoacetate_BetaHB
  Pulmonary_Embolism           depth 3  | Pulmonary_Embolism -> Thrombus_Embolization_to_Pulmonary_Artery -> Ventilation_Perfusion_Mismatch -> Acute_Right_Ventricular_Strain
  Acute_Ischemic_Stroke_MCA    depth 2  | Acute_Ischemic_Stroke_MCA -> Thromboembolic_Occlusion_of_Middle_Cerebral_Artery -> Ischemic_Penumbra_and_Cytotoxic_Edema
  Acute_Pancreatitis           depth 2  | Acute_Pancreatitis -> Premature_Intraacinar_Trypsinogen_Activation -> Autodigestion_and_Fat_Necrosis
  Bacterial_Meningitis         depth 2  | Bacterial_Meningitis -> Hematogenous_Seeding_to_Subarachnoid_Space -> Meningeal_Neutrophilic_Inflammation_Exudate
  Tension_Pneumothorax         depth 2  | Tension_Pneumothorax -> One_Way_Valve_Air_Entrapment -> Impaired_Venous_Return_Preload_Collapse
  Acute_Pericarditis           depth 1  | Acute_Pericarditis -> Pericardial_Serofibrinous_Inflammation
  Aortic_Dissection            depth 1  | Aortic_Dissection -> Cystic_Medial_Necrosis_and_Intimal_Tear
  Bacterial_Pneumonia          depth 1  | Bacterial_Pneumonia -> Streptococcus_Pneumoniae_Alveolar_Infiltration
  Deep_Vein_Thrombosis         depth 0  | Deep_Vein_Thrombosis
  Longest causal chain: depth 4 (Acute_STEMI)
```

**Interpretation.** The prediction holds. The longest chain is the STEMI chain
at depth four, ending at `Myocyte_Coagulative_Necrosis` because the $D=4$ budget
is exhausted exactly there — one hop short of the graph's terminal biomarker,
which is precisely what Theorem B.19.4's $\min(D, L)$ says. Three diseases share
depth three (appendicitis, ketoacidosis, pulmonary embolism); four sit at depth
two; three at depth one; and `Deep_Vein_Thrombosis` sits at depth zero, because
it appears in the graph only as the head of a `complication_of` arc and has no
`caused_by` out-arc, so the trace returns the single-node path. The sorted order
is deterministic — descending depth, then name — and the whole ledger is
**MEASURED** from the built graph. The chain structure is a model of mechanism,
not a clinical assertion; it is the graph's `caused_by` edges read back in order.

---

## 19.6 SDK Parity Audit

Every chapter of this book ends its build with a parity audit: the book's
implementation must agree with the production SDK, not merely resemble it. Here
the reference is `geoformer.clinical.engine.GeoFormerClinicalDiagnosticEngine`,
and the audit checks three things at once — that the two engines rank the same
differential in the same order, that they fire the same contraindication alert,
and that their reported probabilities agree numerically.

We predict bit-exactness, not mere closeness. Both engines compile the same arcs
into the same coalesced sparse tensors, read the same reverse stores, apply the
same $0.6$ causal weight, and renormalize by the same $\varepsilon$; there is no
randomness anywhere in the path from findings to ranking. The only difference is
the projection mode, and the projections are never touched by
`diagnose_patient`. So the maximum probability difference should be exactly
zero.

**Run this now.** Task 6 (`python3 code/gftac/ch19_tasks.py 6`):

```python
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
```

```text
SDK Parity Audit — Book Engine vs GeoFormerClinicalDiagnosticEngine (RESEARCH DEMO):
  Case 1 ranking (STEMI vs dissection): ['Acute_STEMI', 'Aortic_Dissection']
  SDK    ranking (STEMI vs dissection): ['Acute_STEMI', 'Aortic_Dissection']
  Rankings match:                       True
  Case 2 contraindication alert match:  True
  Max differential probability diff:    0.00e+00
  Full Parity Verified:                 True
```

**Interpretation.** The two engines return the identical ranking —
`['Acute_STEMI', 'Aortic_Dissection']` — and the identical contraindication
alert, and the maximum probability difference is `0.00e+00`: bit-exact, not
within tolerance. The prediction was that the projection mode is invisible to
`diagnose_patient`, and the audit confirms it. The book's engine is not a
sketch of the SDK's; on the path that matters here it *is* the SDK's arithmetic,
written down where you can read it. This is the strongest form of the parity
claim the book makes, and it is asserted by
`tests/test_ch19_clinical_kg.py::test_sdk_parity_ranking_and_alert` and its
graph-parity sibling.

---

## 19.7 Conclusion: A Domain, Not a Toy

We took the sparse relational memory of Part V and pointed it at a clinical
knowledge graph, and the machinery carried the domain without modification. The
compile turned $118$ facts into $12$ conductance stores — forward and reverse —
under Theorem B.19.1, at $4{,}720$ bytes. The differential ranked a STEMI-versus-
dissection case under Theorem B.19.2, splitting the evidence three to one. The
safety check fired exactly on the one encoded contraindication under Theorem
B.19.3 and stayed silent otherwise. The causal trace descended a functional
acyclic mechanism graph to depth four under Theorem B.19.4. And the SDK parity
audit returned bit-exact agreement.

The honesty ledger for the chapter is short and it should be read carefully.
Every printed number is the output of a graph propagation over a small hand-built
ontology. The ontology is a teaching graph. The engine is a **RESEARCH DEMO**.
None of it diagnoses anyone, and none of it should be used to. What it
demonstrates is the architectural claim the book has been building toward since
Chapter 1: that a relational memory whose cost scales with the number of facts,
rather than with the square of the number of entities, can ingest genuine domain
structure and reason over it — and that the production engine and the book's
engine agree on the answer.

Part V is now complete. We can hold a hundred-thousand-entity memory, and we can
fill it with a domain. What remains is to leave the data center entirely. The
next chapter takes the model to an Arduino Uno, exporting ternary weights into C
headers and matching the firmware's fixed-point arithmetic in simulation — the
smallest machine the architecture will fit on, and the last demonstration that
the geometry was never a luxury.

---

## Exercises

1. **Add a disease.** Extend `_CLINICAL_TRIPLES` with one new disease, three
   symptoms, and one `treated_by` arc. Before running, predict how
   `census()['num_entities']` and the `presents_with` count change; then verify.
   Does the differential of an all-new-findings case rank the new disease first?
2. **Move the budget.** Recompute Task 5 with `max_depth=5`. Which chains grow,
   and which are unchanged? State the prediction in terms of Theorem B.19.4's
   $\min(D, L)$ before measuring.
3. **Break acyclicity.** Add a `caused_by` arc that closes a cycle among
   mechanisms. The trace's `max_depth` still bounds it, but does the "simple
   path" clause of Theorem B.19.4 fail? Demonstrate the failure and explain which
   hypothesis of the theorem was violated.
4. **Widen the alert.** Modify the safety check to test every disease in the
   differential, not just the primary. On the STEMI-versus-dissection case, does
   the dissection contraindication now fire? Is that the safer behaviour, and
   what does it cost?
5. **Quantify the parity.** Replace the equality test in Task 6 with a
   comparison of raw floating-point bit patterns. Are the two probability
   vectors identical as bit strings, or only to the printed precision? What does
   your answer say about the claim "bit-exact"?

---

## Sources

- The clinical ontology is adapted from standard internal-medicine teaching
  material (disease, symptom, mechanism, drug, and contraindication structure)
  and is included here purely as a graph; it is **not** a clinical resource and
  carries no therapeutic recommendation.
- The sparse memory compiled in this chapter is `gftac.ch18_sparse_memory`,
  built in Chapter 18; the production counterpart is
  `geoformer.layers.sparse_memory.SparseHippocampalRelationalMemory`.
- The reference engine compared in §19.6 is
  `geoformer.clinical.engine.GeoFormerClinicalDiagnosticEngine`, and the
  reference ontology is `geoformer.clinical.ontology.build_clinical_knowledge_graph`.
