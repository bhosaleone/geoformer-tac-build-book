# BOOK_PLAN.md — Complete Table of Contents & Writing Ledger

**Book:** GeoFormer & TAC — The Build Book
**Rule:** A chapter is marked ✅ WRITTEN only when (a) the markdown is complete, (b) the chapter
module exists in `code/gftac/`, (c) the chapter test file passes, and (d) `verify_book.py`
records it green.

---

## House Style (binding for every chapter)

This is a book, not a dossier. The prose register is **descriptive and
rigorous argument**, chapter length unconstrained by convention:

- **Sustained explanatory prose carries the content.** Every claim is developed as an
  argument: mechanism first, consequence second, measurement third. Sections flow as
  paragraphs, with bolded lead-in sentences as signposts where needed — not bullet lists.
- **Lists and tables appear only where the object itself is a list or table** — a measured
  ledger the experiment prints, a notation table, a comparison table the surrounding
  paragraphs then argue through, exercises, sources. Never as a substitute for explanation.
- **Theorems are proved in full**, numbered B.<chapter>.<n>, cross-referenced to the Q-series.
- **Experiments are narrated, not summarized**: design rationale (what confound each choice
  removes), prediction in prose before measurement, the run, then interpretation line by line.
- **Every number is labeled MEASURED / MODELED / RESEARCH DEMO** and tied to a test.
- **Runnable tasks are interleaved with the theory.** Every chapter ships `code/gftac/chNN_tasks.py`;
  at each point of use the text shows a "Run this now" block — the verbatim task body, the command
  (`python3 code/gftac/chNN_tasks.py N`), the measured output, and an interpretation paragraph
  reading the result back. `tests/test_chNN_tasks.py` pins every printed line, so no transcript in
  the book can drift from the artifact that produces it.
- **Prose is machine-checked against its artifacts.** `verify_transcripts.py` extracts every
  `python3 code/gftac/...` command and the `text` block that follows it, runs the command, and
  asserts the transcript is byte-for-byte what the artifact prints. A run-dependent value (a timing,
  say) is written as a **volatile span** `«…»`: the checker matches any value there while everything
  outside the span must match exactly, so the volatile set is explicit, auditable, and counted.
- **The book certifies itself.** `verify_book.py` runs every chapter test, checks every transcript,
  and hashes every chapter, appendix, module, and test file into
  `book_verification_certificate.json`, which records the environment (Python/torch versions), the
  per-chapter ledger, the transcript ledger, and a SHA-256 manifest. `all_green` requires both the
  tests and the transcripts to pass. A chapter is "done" only when the certificate records it green.
- **Claims are labeled and indexed.** `code/gftac/ch22_claims.py` renders `CLAIMS.md`: one row per
  public number, each carrying a MEASURED / MODELED / RESEARCH DEMO label, the artifact that
  produces it, the test that pins it, and the command that regenerates it. The ledger audits itself —
  every cited artifact and test must exist, and every pinned value must occur in its test.
- **Voice:** direct address; "we" is earned by the reader having executed the step.

---

## Part I — Foundations: Theory You Can Touch

### Ch 0 — The Reader's Contract & Environment ✅ WRITTEN
- The five contract clauses; how to run the book; notation; the honesty ledger; the certificate.

### Ch 1 — The Euclidean Bottleneck: Geometry as Tool and Weapon ✅ WRITTEN
- **Framing:** the flat-space contract of modern transformers stated precisely (immutable ℝ^D,
  all-to-all attention, implicit metric, discrete depth, knowledge-in-weights); the four
  pathologies that follow (P1 entanglement/double-duty, P2 topological blindness, P3 discrete
  depth, P4 weight-bound memory).
- **The confrontation:** Euclidean models vs. our approach — property-by-property table
  (space, distance, routing vs. semantics, depth, knowledge write, compute, memory, verifiability).
- **Geometry as TOOL:** the four instruments — resistance matrix D → attention modulator,
  Laplacian L → dynamics generator, conductance W → memory, d_ij → complexity metric;
  constructive theorems as guarantees; hardware isomorphism (conductance is a physical quantity).
- **Geometry as WEAPON:** falsifiable superiority where relations matter (flat 0.0% vs published
  RML 84.7% ± 5.1); depth weapon (emergent τ*, no scratchpad); energy weapon (1.58-bit, O(|E|)
  sparse propagation, pJ-scale modeled / MCU-measured); memory weapon (100k entities < 10 MB);
  **armor** (machine-checkable invariants → auditability).
- **The engagement envelope (honesty):** where the weapon wins (path-separability α₀ > 1,
  relational structure, few-shot composition) and where it must not be drawn (dense expander
  graphs — the program's own Blind-PG measurement: all engines ≤ 3.3% there).
- **Build:** `TransitiveWorld` (directed chains), `OneHopMemorizer` (trained only on 1-hop facts),
  matrix-power propagation baseline — the measured collapse in miniature.
- **MEASURED:** one-hop expert at 100% train accuracy scores ~0% on unseen 2-hop and 3-hop
  queries by *echoing the direct successor* (systematically wrong, below chance); propagation
  scores 100% at every depth.
- **Tests:** world structure invariants; memorizer train accuracy == 1.0; two-hop and three-hop
  collapse ≤ 20% with echo ≥ 50%; propagation == 100%.

### Ch 2 — The Metric Lattice ✅ WRITTEN
- **Theory:** conductance lattice W ≥ 0; effective resistance d_ij = 1/(W_ij + ε);
  **Book Thm B.2.1** (Hebbian resistance contraction, proof); **B.2.2** (Laplacian PSD, proof via
  the pairwise-difference identity); **B.2.3** (Fiedler lower bound λ₂ ≥ N·Δw for complete
  min-conductance lattices — the precise statement of repo Theorem Q-2, with proof);
  b_w-bit level quantization; directed kernel P_dir and dead-end policy.
- **Build:** `MetricLattice` (observe_edge with quantization, resistance, Laplacian, Fiedler,
  directed kernel).
- **Tests:** contraction monotonicity; PSD; Fiedler bound on random dense lattices; level
  quantization invariant; row-stochasticity + dead ends.

### Ch 3 — TAC: Thermodynamic Attractor Convergence ✅ WRITTEN
- **Theory:** free energy F[s]; gradient flow ds/dτ = −κ∇F; **B.3.1** (Lyapunov dissipation
  dF/dτ = −κ‖∇F‖² ≤ 0, proof); **B.3.2** (Grönwall convergence timescale, proof);
  **B.3.3** (double-well bistability: minima ±1, barrier at 0, proof);
  dimension-independence made precise (and its honest limits); Langevin temperature & Kramers
  escape (preview of grokking).
- **Build:** `DoubleWellPotential`, `QuadraticPotential`, `TacRelaxer` (Euler–Maruyama, drift
  halting, energy history).
- **Tests:** monotone energy; basin selection ±1; Grönwall step-count prediction (±1);
  identical step counts across N ∈ {10, 100, 1000} under unit-norm initialization.

### Ch 4 — Ternary Synapses: BitNet b1.58 ✅ WRITTEN
- **Theory:** ternary quantization w_q = clip(round(w/γ), −1, 1), γ = mean|w|;
  straight-through estimation (gradient identity on effective weights); information-theoretic
  footprint log₂3 ≈ 1.585 bits and the 5-trits-per-byte packing (1.6 bits/param);
  variance-preserving initialization.
- **Build:** `ternary_quantize`, `TernaryLinear` (STE via the detach trick),
  `pack_trits`/`unpack_trits`, `packed_bytes`, `variance_preserving_init_`.
- **Tests:** trit domain; STE gradient identity; pack/unpack roundtrip incl. non-multiple-of-5;
  249,665-param foundation footprint = 49,933 B = 48.76 KiB (reproduces the published claim);
  variance preservation ratio; **parity with SDK `geoformer.quant.ternary_linear.QuantizedLinear`**.

---

## Part II — Assembling the GeoFormer Block

### Ch 5 — Geodesic Highway Attention (Q-GHA) ✅ WRITTEN
- Theory: attention scores as kernel smoothing on a metric manifold,
  S_ij = Q_iK_jᵀ/√d_k − λ_geo·d_ij; multi-head structure with ternary projections.
- Build: `BookQGHA`. Parity: SDK `geoformer.layers.attention.QuantizedGHA` (same weights → same
  attention maps, bit-exact with `act_bits=None`). Tests: softmax row-stochasticity (B.5.1);
  concentration law — d E_a[d]/dλ = −Var_a(d) ≤ 0, large-λ limit onto argmin d (B.5.2), with
  the honest boundary that entropy is *not* monotone in general (dH/dλ = Cov_a(d,S) − λVar_a(d),
  counterexample pinned); directed-Finsler asymmetry L_ij−L_ji = (S_ij−S_ji)+2B_ij and the
  ratio law A_ij/A_ji = (Z_j/Z_i)e^{2B_ij} (B.5.3).

### Ch 6 — Ternary SwiGLU & RMSNorm ✅ WRITTEN
- Build: `RMSNorm`, `TernarySwiGLU` (d_ff = 8/3·d_model convention).
- Parity vs `geoformer.layers.ffn` & `geoformer.models.block.RMSNorm` — bit-exact on both the
  weight-only path (`act_bits=None`) and the SDK default (8-bit activation quantization).
- Tests: RMSNorm scale law B.6.1 (radial map, n(cx) = ρ(c)n(x)); SiLU gate B.6.2 (bounds,
  minimum z* = −1.2784645, limits); parameter accounting B.6.3 (3·d_model·d_ff = 8·d_model²).

### Ch 7 — Q-TAC Inside the Block: Continuous Relaxation Layers ✅ WRITTEN
- Theory: layer-free depth — relaxation until Lyapunov drift < ε; multi-scale schedules
  (fixed / Fibonacci / continued-fraction convergents); why scale-hopping accelerates settling.
- Build: `BookQuantizedTAC` + `make_relaxation_schedule`. Parity vs `geoformer.layers.tac`.
- Tests: drift decreases monotonically in expectation; halting step count ≤ schedule length;
  output invariance under schedule permutation (same τ_total).

### Ch 8 — The S₅ Cayley Projector & Description-Length Bounds ✅ WRITTEN
- Theory: irreps of S₅ (dims 1,1,4,4,5,5,6; Σd² = 120); projection as symmetry pooling;
  **Kolmogorov bound K(S) ≤ Σ d_ρ log₂ d_ρ = 54.68 bits** (derivation + what it does and does
  not mean). Clearly-labeled: engineering device, not physics claim.
- Build: `BookS5Projector`. Parity vs `geoformer.layers.cayley.S5CayleyProjector`.
- Tests: projector idempotence on symmetric component; spectrum matches Σd² accounting;
  K_S bound holds for random inputs.

### Ch 9 — Assembling the Block & the Full Model ✅ WRITTEN
- Build: `BookGeoFormerBlock` (pre-RMSNorm residual Q-GHA → SwiGLU → Q-TAC) and
  `BookGeoFormer` stack; quasi-crystalline positional encodings (golden-ratio cut-and-project).
- Parity vs `geoformer.models.geoformer.GeoFormerBlock` / `GeoFormer`.
- Tests: forward/backward shapes & finiteness; parameter + packed-KB ledger formulas;
  checkpoint save/load roundtrip; collision rate of quasi encodings = 0 on N ≤ 2048.

---

## Part III — Training the Real Model

### Ch 10 — Two-Timescale Learning: Hebbian Plasticity with Invariants ✅ WRITTEN
- Theory: complementary learning systems; fast Hebbian outer-product updates on the conductance
  lattice with b_w-bit quantization; invariant battery (W ≥ 0; W_ii = 0; bounded ΔW; level
  membership) — the repo's `EndogenousConsolidator` invariant set, enforced as tests.
- Build: `PlasticConductanceEngine` (batched `update_conductance`).
- Tests: the full invariant battery after every update pattern; resistance contraction curves.

### Ch 11 — The Integrated Model & Loss Suite ✅ WRITTEN
- Build: `BookGeoFormerTACModel` (embeddings → blocks → classifier → attractor) and the loss
  suite: task CE + directed Dirichlet energy + Lyapunov drift penalty + attractor entropy +
  ternary weight regularization. Parity vs `geoformer.training.model` / `losses`.
- Tests: loss components computable & finite; gradient flows to both timescales (latents and
  lattice hooks); state_dict roundtrip.

### Ch 12 — Datasets: Chains, Crossroads, and the α₀ Control ✅ WRITTEN
- Theory: benchmark graph generators — un-truncated continuous graphs, decoy side-loops with
  conductance ratio α₀ = W_true/W_decoy, branching factor control; **no oracle sinks** (every
  node keeps outgoing edges) per the Tier-8 audit.
- Build: `TransitiveChainDataset` / `make_crossroad_graph`.
- Tests: no dead ends; decoy weights satisfy α₀ spec; chains are vertex-disjoint; query targets
  never appear as 1-hop training facts (zero-shot property).

### Ch 13 — The Training Loop & the Kramers Optimizer ✅ WRITTEN
- Theory: dual-path momentum + Langevin diffusion schedule (Kramers tunneling as exploration);
  cosine LR; two-timescale update ordering (plasticity inside the loop).
- Build: `KramersOptimizer` (book), `train()` harness with checkpointing + history JSON.
- MEASURED: full training run table (loss curves, per-epoch multi-hop eval) reproduced from the
  repo's `geoformer_tac` run protocol; determinism test (same seed ⇒ identical loss trajectory).

### Ch 14 — The Evaluation Protocol: Honest Numbers ✅ WRITTEN
- Build: Hits@1/3/10, MRR, target modal mass, entropy, emergent τ*; **exact Student-t p-values
  via Simpson quadrature** (build + verify against `scipy.stats`); 5-seed ledgers.
- Tests: Simpson integrator matches scipy to 1e-10; metric implementations verified on
  hand-computed toy cases; zero-shot discipline asserts.

---

## Part IV — Emergent Deep Reasoning

### Ch 15 — How Linear Diffusion Fails: Three Diagnostics ✅ WRITTEN
- Build & MEASURED reproduction of the three canonical failure modes: (1) residual drowning
  (η·residual dilutes the frontier), (2) symmetrization echo collapse (W+Wᵀ creates 2-cycles),
  (3) exponential dispersion (mass ~ (1/d)^k drops below chance floor).
- Tests: each failure reproduced quantitatively (drowning ≤ chance; echo degrades top-1;
  dispersion crosses the 1/N floor by a predicted hop count).

### Ch 16 — Soliton Sharpening & Theorem Q-3 (with Proof) ✅ WRITTEN
- Theory: γ-power renormalization; **Book Thm B.16.1 = repo Theorem Q-3**: under min true-path
  edge ratio α₀ > 1 and average branching d_avg, if γ > log(d_avg)/log(α₀) then distractor
  ratio contracts ρ = d_avg/α₀^γ < 1 and peak modal mass P_peak(k) ≥ 1/(1 + Cρ^k) = Ω(1).
  Full induction proof. **The falsification boundary**: when α₀ ≤ 1 or graphs are expander-like,
  sharpening concentrates arbitrarily (Blind-PG Experiment A honesty) — stated plainly.
- Build: `soliton_sharpen(z, gamma)` with entropy-adaptive γ(τ) = min(γ_max, γ₀ + β·H(z)).
- Tests: lower bound holds across random ensembles with α₀^γ > d_avg; violated (and reported)
  below the boundary; γ adaptivity raises with entropy.

### Ch 17 — Goal-Conditioned Free Energy & Autonomous 32-Hop Halting ✅ WRITTEN
- Theory: F[z; q] = λ_dir/2 Tr(zᵀL_dir z) + λ_V/2 V(z) + λ_q d_sem(z, q); oracle-free halting
  (drift + semantic capture); why this kills the Tier-8 artifacts (fixed-point saturation,
  truncated-graph sinks, metric category errors).
- Build: `GoalConditionedAttractor` (book) — parity vs `geoformer.layers.emergent_multihop`.
- MEASURED: the certified 32-hop ledger protocol (un-truncated graphs, 5 seeds, mass + argmax +
  entropy + τ*); paired t-test with exact p.
- Tests: τ* equals true depth at 4–32 hops on un-truncated graphs; no dead-end nodes present
  (oracle-freeness asserted structurally); halts for the semantic reason (d_sem below ε_q).

---

## Part V — Integration & Scale

### Ch 18 — Sparse Hippocampal Memory at 100,000 Entities ✅ WRITTEN
- Build: CSR conductance store + sparse propagation + emergent multi-hop; footprint measurement.
- MEASURED: RAM vs dense (≈ 80,000× compression claim reproduced); latency scaling.
- Tests: sampled-query correctness vs dense reference; memory ≤ budget; sparse invariants.

### Ch 19 — Clinical Knowledge-Graph Integration (RESEARCH DEMO) ✅ WRITTEN
- Build: ontology → Sparse-RML compile → differential diagnosis engine with contraindication
  alerts. Clearly labeled research demo; no clinical claims.
- Tests: known differential rankings on the built-in KG; alert firing on encoded
  contraindications; latency bounds.

### Ch 20 — MCU Deployment: GeoFormer on an Arduino Uno ✅ WRITTEN
- Build: weight export to C headers (PROGMEM), fixed-point Q8.8 simulator matching the firmware
  arithmetic; serial protocol.
- MEASURED: fixed-point vs float agreement rate; (with hardware) µs latencies from the
  firmware's own benchmark.
- Tests: Q8.8 simulator output matches float pipeline within tolerance on the exported test
  vectors; adversarial probes (noise injection, shortcut rejection) pass in simulation.

### Ch 21 — The Memristor Crossbar Energy Model (MODELED) ✅ WRITTEN
- Theory: 1T1R Ohm/KCL computing; the analytical energy model (read voltage, pulse width, HRS/LRS
  conductances, wire parasitics, 4-bit ADC/DAC peripheral factor); Monte-Carlo noise resilience.
- Clearly labeled MODELED throughout; the 443.9 pJ figure re-derived step by step.
- Tests: energy formula recomputation; noise-resilience curve reproduction (38% @ 20% noise);
  invariance of conclusions to parameter perturbation (sensitivity analysis).

---

## Part VI — Verification, Trust & Capstone

### Ch 22 — The Claims Ledger ✅ WRITTEN
- Build: generate `CLAIMS.md` programmatically from the test suite + result artifacts; the
  MEASURED / MODELED / RESEARCH DEMO taxonomy applied to every public number in the program.
- Tests: every ledger row cites an artifact that exists and a command that exits 0.

### Ch 23 — What the Z3 Certificates Do and Don't Say ✅ WRITTEN
- Run the repo's Z3 suites; read the certificates; bounded-domain disclosure; upgrade path
  (property-based randomized verification of each invariant's precondition ⇒ conclusion across
  graph ensembles, FP32/FP64).
- Tests: suites re-run green; property-based checks hold on sampled ensembles.

### Ch 24 — Capstone: The Whole Stack in One File ✅ WRITTEN
- Build: `capstone.py` — lattice → plasticity → attention block → training → soliton attractor →
  32-hop evaluation → packed export — the entire book in a single runnable artifact.
- The graduation certificate: `verify_book.py` green across all 24 chapters.

---

## Appendices ✅ WRITTEN
- **A:** Notation & symbol table (W, d_ij, P_dir, L_dir, z, γ, κ, τ, ε, α₀, d_avg, K(S)).
- **B:** Book ↔ SDK API cross-reference (which chapter module mirrors which SDK module).
- **C:** Reproducibility manifest (seeds, versions, hardware notes).
- **D:** Troubleshooting (float precision, sparse warnings, seed discipline).

---

## Status Ledger

| Chapter | Markdown | Code | Tests | Certificate |
|---|---|---|---|---|
| 00 Reader's Contract | ✅ | — | — | ✅ |
| 01 Euclidean Bottleneck (Tool & Weapon) | ✅ | ✅ | ✅ | ✅ |
| 02 Metric Lattice | ✅ | ✅ | ✅ | ✅ |
| 03 TAC Relaxation | ✅ | ✅ | ✅ | ✅ |
| 04 Ternary Quantization | ✅ | ✅ | ✅ | ✅ |
| 05 Geodesic Highway Attention | ✅ | ✅ | ✅ | ✅ |
| 06 Ternary SwiGLU & RMSNorm | ✅ | ✅ | ✅ | ✅ |
| 07 Q-TAC Continuous Relaxation | ✅ | ✅ | ✅ | ✅ |
| 08 S₅ Cayley Projector | ✅ | ✅ | ✅ | ✅ |
| 09 Assembling the Block & Full Model | ✅ | ✅ | ✅ | ✅ |
| 10 Two-Timescale Hebbian Plasticity | ✅ | ✅ | ✅ | ✅ |
| 11 Integrated Model & Loss Suite | ✅ | ✅ | ✅ | ✅ |
| 12 Datasets: Crossroads & α₀ Control | ✅ | ✅ | ✅ | ✅ |
| 13 Training Loop & Kramers Optimizer | ✅ | ✅ | ✅ | ✅ |
| 14 Evaluation Protocol | ✅ | ✅ | ✅ | ✅ |
| 15 Linear-Diffusion Failures | ✅ | ✅ | ✅ | ✅ |
| 16 Soliton Sharpening & Q-3 | ✅ | ✅ | ✅ | ✅ |
| 17 Goal-Conditioned 32-Hop Halting | ✅ | ✅ | ✅ | ✅ |
| 18 Sparse Hippocampal Memory | ✅ | ✅ | ✅ | ✅ |
| 19 Clinical Knowledge Graph (RESEARCH DEMO) | ✅ | ✅ | ✅ | ✅ |
| 20 MCU Deployment (Arduino Uno) | ✅ | ✅ | ✅ | ✅ |
| 21 Memristor Crossbar Energy (MODELED) | ✅ | ✅ | ✅ | ✅ |
| 22 The Claims Ledger | ✅ | ✅ | ✅ | ✅ |
| 23 What the Z3 Certificates Do and Don't Say | ✅ | ✅ | ✅ | ✅ |
| 24 Capstone: The Whole Stack | ✅ | ✅ | ✅ | ✅ |
| Appendices A–D | ✅ | — | — | ✅ |

**Definition of done.** A row's Certificate column is ✅ only when
`python3 verify_book.py` records it green — that is, its chapter tests pass, its
transcripts match byte-for-byte, and its files are hashed into
`book_verification_certificate.json`. The ledger is the book's own report card,
and the certificate is the evidence.
