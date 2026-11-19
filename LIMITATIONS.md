# LIMITATIONS.md — What This Book Does Not Establish

This document is the honest counterpart to the verification certificate. The certificate proves
that the book's files, tests, and printed transcripts agree with one another. It does **not**
prove that GeoFormer & TAC is better than another architecture, that its theorems are novel, or
that its demos work outside the conditions under which they were run. If you are reviewing this
work, start here.

---

## 1. What verification does and does not mean

| The certificate establishes | The certificate does not establish |
|---|---|
| 442 chapter tests pass on the stated environment. | That the architecture generalizes to data outside the book's generators. |
| 151 printed transcripts match the artifacts byte-for-byte, with 3 declared volatile spans. | That a printed number came from an independent benchmark rather than an illustrative input. |
| 219 artifact hashes match the published certificate. | That the mathematical claims are novel relative to the literature. |
| Parity imports resolve to the pinned, vendored SDK commits. | That a different machine, seed set, or graph ensemble reproduces the same results. |

`all_green: true` is a statement about **internal consistency**, not external validity. A book
can be perfectly self-consistent and still be wrong about the world.

## 2. Synthetic benchmark boundary

The striking Chapter 17 result — 100% argmax accuracy at 32 hops, target mass 0.7928, five seeds,
un-truncated graphs — is measured on **constructed graphs** with a declared true-path/conductance
ratio $\alpha_0 > 1$ and controlled branching. It is not 32-hop reasoning on arbitrary real-world
knowledge. The following are **not established** by this book:

* performance on randomly generated graphs without a planted high-conductance path;
* performance on real knowledge graphs (FB15k-237, Wikidata, clinical KGs beyond the built-in demo);
* robustness to noisy, missing, or contradictory edges;
* behaviour when several paths lead to competing targets of comparable conductance;
* behaviour when the graph changes after training;
* comparison against retrieval-augmented or graph-native strong baselines on the same data.

A result of "100% at 32 hops on constructed highways" should be read as a controlled
demonstration of a mechanism, not as a general reasoning claim.

## 3. Baseline fairness

The book's comparisons are deliberately narrow: linear diffusion, a flat transformer-style
baseline in Chapter 1, and the book's own ablations. A serious evaluation would also include
GNNs, Relational GCNs, Graph Attention Networks, knowledge-graph embedding models, a Transformer
given equivalent graph information, and strong retrieval-augmented baselines — all with equal
information, training budget, and evaluation protocol. **Those comparisons have not been run in
this book.** We consider them the single most valuable next experiment.

## 4. Statistical examples are not external evidence

* **Chapter 14, Task 4** computes Welch's $t$-test on two five-element lists *hardcoded in the
  task*. The resulting $t = 153.98$, $df = 7.93$, $p = 1.29\times10^{-14}$ demonstrates the
  calculation only. It is not five independent benchmark runs. The chapter and the task output
  now label this explicitly.
* **Chapter 17, Task 5** generates its samples from the book's declared simulator
  (`simulate_certified_32hop_protocol`) and then tests them. That is better provenance, but the
  samples still come from the author's model of the task, not from an external benchmark.
* Exact Student-t integration via Simpson quadrature (Chapter 14, Task 3) is a numerical
  correctness result, not a scientific result.

Reviewers should trace every $p$-value back to its generating artifact. Where the artifact is a
literal list, the correct reading is "worked example".

## 5. Theoretical scope

The theorems in the book are proved **within the book's stated axioms**. A correct proof of a
conditional statement does not establish that the axioms describe a useful regime, that the bound
is tight in practice, or that the construction is novel. Specifically:

* Z3 certificates are bounded-domain; Chapter 23 says exactly what they do and do not show.
* The soliton bound requires $\alpha_0^\gamma > d_{\text{avg}}$; below that boundary the book
  reports the failure itself.
* The S₅ projector and description-length bound are engineering devices with a stated
  interpretation, not a physics claim.
* "Thermodynamic" names describe a mathematical analogy (free energy, drift, Kramers escape),
  not a claim about physical hardware temperature.

We do not claim these constructions are unprecedented. Literature comparison is an open task.

## 6. Hardware and clinical boundaries

* Chapter 21's **443.897 pJ** figure is labeled **MODELED**. It is an analytical estimate with a
  sensitivity analysis, not a measurement from a fabricated crossbar.
* Chapter 20's Arduino example is a Q8.8 simulator plus exported C headers. Board-level latency
  and energy are not established here; "with hardware" numbers in the text require an actual
  board on your bench.
* Chapter 19's clinical KG is labeled **RESEARCH DEMO**. It is a toy ontology with encoded
  contraindications, not a diagnostic device, not clinically validated, and contains no patient
  data. Do not use it in a clinical setting.

## 7. Engineering boundaries

* No throughput, memory-bandwidth, kernel-efficiency, or performance-per-watt profile against
  optimized GPU implementations is included. Graph overhead at scale is unmeasured here.
* The sparse-memory result (100,000 entities) is measured in the book's CSR implementation; it
  is not a production database benchmark.
* The pinned vendored SDK subsets are trimmed copies. They pin parity for this book; they are
  not a substitute for the upstream repositories.

## 8. How to help

The most useful contribution is not praise. It is one of:

1. clone the repository and reproduce the numbers without contacting the author;
2. run the Chapter 17 protocol on a real KG or a random graph with no planted path;
3. add one strong baseline under matched budgets and report the loss;
4. find an error in a theorem statement or its precondition;
5. falsify a claim in `CLAIMS.md` and open an erratum with the artifact that shows it.

See `REPRODUCTION.md` for the exact commands and `RESEARCH_BRIEF.md` for the claims worth
attacking first.
