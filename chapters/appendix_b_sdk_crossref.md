# Appendix B — Book ↔ SDK API Cross-Reference

The book builds its own `gftac` package, chapter by chapter, so that every
mechanism can be read from scratch. The production SDK — `geoformer/src` and
`TAC/src` — contains counterparts for many of those mechanisms. Where a
counterpart exists, the chapter's test suite contains a **parity test** that runs
both implementations and asserts they agree; where the agreement is bit-exact,
the test asserts bit-exact equality, and where it is numerical, the test pins a
tolerance. This appendix is the map: for each chapter, the book module, the SDK
counterpart, and how far the parity claim goes.

The status column has three values. **Bit-exact** means the parity test asserts
exact equality of outputs (or of a scalar the two compute identically).
**Numerical** means the test pins agreement to a stated tolerance. **Firmware**
means the book's counterpart is not in `geoformer/src` at all but in the MCU
firmware tree, and the parity is between the book's simulator and the firmware's
fixed-point arithmetic. A dash in the counterpart column means the chapter is
**book-only**: it develops a mechanism the SDK does not (yet) implement, or it is
a verification chapter that has no executable counterpart.

| Ch | Book module | SDK / firmware counterpart | Parity |
|---|---|---|---|
| 01 | `ch01_compositional_collapse.py` | `geoformer/models/transitive.py` (analogue) | book-only |
| 02 | `ch02_metric_lattice.py` | `geoformer/core/laplacian.py`, `core/metric_warping.py` (analogue) | book-only |
| 03 | `ch03_tac.py` | `geoformer/layers/tac.py` (analogue) | book-only |
| 04 | `ch04_ternary.py` | `geoformer/quant/ste.py`, `quant/ternary_linear.py`, `models/geoformer.py` | Bit-exact |
| 05 | `ch05_qgha.py` | `geoformer/layers/attention.py` (`QuantizedGHA`) | Numerical |
| 06 | `ch06_swiglu.py` | `geoformer/quant/ste.py`, `models/block.py` (`RMSNorm`), `layers/ffn.py` (`TernarySwiGLU`) | Numerical |
| 07 | `ch07_qtac.py` | `geoformer/layers/cayley.py` (`S5CayleyProjector`), `layers/tac.py` (`QuantizedTAC`) | Numerical |
| 08 | `ch08_cayley.py` | `geoformer/layers/cayley.py` (`S5CayleyProjector`) | Bit-exact |
| 09 | `ch09_model.py` | `geoformer/models/geoformer.py` (`GeoFormerBlock`) | Numerical |
| 10 | `ch10_plasticity.py` | `geoformer/autonomous/endogenous_engine.py` (`EndogenousConsolidator`) | Numerical |
| 11 | `ch11_integrated_model.py` | `geoformer/training/config.py`, `training/losses.py` (`GeoFormerTACLoss`) | Numerical |
| 12 | `ch12_datasets.py` | `geoformer/training/datasets.py` (`TransitiveKnowledgeGraphDataset`) | Bit-exact |
| 13 | `ch13_trainer.py` | `geoformer/optim/kramers.py` (`DualPathKramersOptimizer`) | Numerical |
| 14 | `ch14_eval.py` | `geoformer/training/trainer.py` (analogue) | book-only |
| 15 | `ch15_diffusion_failures.py` | — | book-only |
| 16 | `ch16_soliton.py` | `geoformer/layers/emergent_multihop.py` (analogue) | book-only |
| 17 | `ch17_goal_attractor.py` | `geoformer/autonomous/`, `training/blind_attractor.py` (analogue) | book-only |
| 18 | `ch18_sparse_memory.py` | `geoformer/layers/sparse_memory.py` (`SparseHippocampalRelationalMemory`) | Bit-exact |
| 19 | `ch19_clinical_kg.py` | `geoformer/clinical/engine.py`, `clinical/ontology.py` | Bit-exact |
| 20 | `ch20_mcu.py` | `geoformer/firmware/uno_nano_geoformer/` (`main.c`, `mcu_weights.h`) | Firmware |
| 21 | `ch21_memristor.py` | `geoformer/hardware/crossbar.py`, `hardware/energy.py` | Bit-exact |
| 22 | `ch22_claims.py` | — | book-only |
| 23 | `ch23_z3_certificates.py` | `TAC/src/tac/proofs/z3_verifier.py` (`TACZ3Verifier`) | Bit-exact |
| 24 | `ch24_capstone.py` | — | book-only |

Two notes on reading the table. First, **Bit-exact does not mean the two
implementations are the same code**; it means they compute the same numbers on
the same inputs, which is the strongest claim a parity test can make without
sharing a code path. Chapter 4's quantization and Chapter 8's Cayley projector
are bit-exact because they use identical integer arithmetic; Chapter 18's sparse
memory is bit-exact because both implementations do the same degree-normalized
sparse matrix-vector product in the same order. Second, the **book-only**
chapters are not gaps to be embarrassed about. Chapter 15 (diffusion failures)
is a negative result — a *falsification* — and the SDK does not implement a
mechanism it has falsified. Chapters 16 and 17 are the book's frontier: the
soliton sharpening theorem and the autonomous halting protocol are developed
here, and their SDK analogues (`emergent_multihop.py`, the `autonomous/` package)
are the closest existing code, not a settled implementation. Chapter 22 and
Chapter 24 are verification chapters and have no mechanism to mirror.

The firmware row deserves its own sentence, because the book is careful about
it. Chapter 20 does not claim to be a 16-bit Q8.8 implementation. The firmware
uses `FP_ONE = 255`, an **0.8 fixed-point** representation, and the book says so
everywhere it appears. The parity test is between the book's fixed-point
simulator and the firmware's arithmetic at that same scale, and it passes at
that scale; the chapter's prose states the limitation plainly rather than
rounding it up to a claim it cannot support.
