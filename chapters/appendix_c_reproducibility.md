# Appendix C — Reproducibility Manifest

A measured claim is only worth what its reproduction is worth. This appendix
records exactly what the book was measured on, how to reproduce every number in
it, and where the sources of nondeterminism were and how each was closed. The
governing rule is the one stated in the reader's contract: **every printed
number is either deterministic under a seed, or it is wrapped in a volatile span
the checker matches by pattern.**

## C.1 The environment

The certificate the book emits records the interpreter and the numerical stack
it ran on. The reference run recorded:

| Component | Version |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| z3-solver | 5.1.0 |

The stack is **CPU-only**. Nothing in the book requires a GPU: every model is
tiny (the capstone's two-block GeoFormer is 29,152 parameters), every tensor
operation is small, and every measurement is deterministic on CPU. A GPU run may
change the last bits of a floating-point reduction, which is why the book's
transcripts are pinned to the CPU reference and why volatile spans exist.

## C.2 Reproducing everything, in order

From the book's root:

```
python3 verify_book.py          # run every chapter test, check every transcript, write the certificate
python3 verify_transcripts.py   # check only the transcripts (fast)
python3 -m pytest tests -q      # the raw test suite
```

`verify_book.py` runs every `tests/test_ch*.py`, then every transcript in
`chapters/`, then hashes every chapter, module, and test file into
`book_verification_certificate.json`. It exits zero if and only if the tests and
the transcripts are both green. The certificate records the environment, the
per-chapter pass/fail ledger, the transcript ledger, and a SHA-256 manifest of
every artifact, so a third party can confirm the exact files that produced the
green result.

The `Makefile` exposes the same operations: `make verify`, `make test`,
`make transcripts`, `make all`.

## C.3 Seed discipline

Determinism is a design constraint, not an afterthought. Every source of
randomness in the book is either seeded or absent. The seeds, by chapter:

| Ch | Seeds | What they fix |
|---|---|---|
| 01 | `torch.manual_seed(0)` | the transitive-world construction and the one-hop expert's init |
| 02 | `0`, `7` | the lattice construction and the stochastic-projection demonstration |
| 03 | `TacRelaxer(seed=11)` | the relaxer's tie-breaking |
| 04 | `0`, `1`, `2` | quantization inputs, STE inputs, packing inputs |
| 05 | `0`, `1` | attention inputs and the directed-kernel demonstration |
| 06 | `0`, `1`, `2`, `3` | SwiGLU / RMSNorm inputs and the gate-minimum scan |
| 07 | `seed` (per task), `1` | the Q-TAC operator and the Cayley/Chebyshev comparison |
| 08 | `0`, `1` | the projector's inputs and the collision scan |
| 09 | `0` | the assembled block and model |
| 10 | `123`, `42` | the plasticity engine (fast writes and endogenous consolidation) |
| 11 | `0`, `1`, `2`, `5`, `6` | the integrated model and the five loss terms (task 6 uses seed `6`) |
| 12 | `42` | the crossroads graph and the dataset |
| 13 | `0`, `42`, `100`, `200` | the trainer, the Kramers comparison, and the ablation runs |
| 18 | `42` | the sparse memory's readout projections |
| 24 | `0` (Stage 3), `seed=0`/`seed=42` (Stages 1–2) | the capstone's model and lattices |

Chapters 14–17, 19–23 either use no random draws or draw from closed-form
deterministic formulas (Chapter 16's soliton chain, Chapter 17's untruncated
depth protocol with fixed graph ensembles, Chapter 20's fixed-point simulator,
Chapter 21's deterministic energy model, Chapter 23's Z3 and property-based
sweeps over fixed ensembles).

**A case study in why this matters.** During the hardening pass, Chapter 11's
task 6 was found to print a value that changed from run to run: it constructed
its model without seeding the generator first, so the printed scalar depended on
whatever global RNG state the previous test had left behind. The fix was one
line — `torch.manual_seed(6)` — after which the value is a fixed
`0.060097 / 2.317219`. The lesson is the book's seed discipline in miniature: an
unseeded draw is not a small impurity, it is a number that cannot be pinned, and
a number that cannot be pinned cannot be claimed.

## C.4 Volatile spans

A few numbers genuinely depend on the machine — wall-clock timings, for example.
The book does not pretend these are reproducible; it **marks** them. Any value
that may vary between runs is wrapped in a volatile span, written `«…»` in the
markdown, and the transcript checker matches any run's value at that position
while requiring everything outside the span to match byte-for-byte. The volatile
set is therefore explicit and auditable: it is the set of `«…»` markers, and the
certificate counts them (`transcripts_volatile_spans`). A run of the book
currently carries a small, fixed number of such spans, all of them timing
figures; the count is visible in the certificate.

A second, narrower allowance covers **printed decimal tokens**. Even with Python
and torch pinned exactly, the last digits of a six-decimal eigenvalue can change
with the CPU and BLAS build. The transcript checker therefore compares every
non-numeric byte exactly, but compares each decimal token with a relative and
absolute tolerance of $10^{-4}$. That tolerance is deliberately too small to
hide a structural change, a wrong label, or a changed result; it only absorbs
last-digit formatting drift. The chapter tests use the same rule through
`tests/numeric_asserts.py`. Values whose *whole* magnitude is machine-dependent
— the Chapter 24 aggregate hash, which covers the float metrics — remain marked
with `«…»` instead.

## C.5 Numerical precision

Two precision facts recur and are worth recording here so they are not
rediscovered as surprises.

First, **FP32 versus FP64.** Several chapters report a residual under a
permutation or a symmetry, and that residual's *size* is a function of dtype:
Chapter 7's permutation residual is `2.384e-07` in FP32 and `4.441e-16` in FP64,
exactly the $2^{-24}$ and $2^{-52}$ machine epsilons of the two formats. The
tests pin both, so the dtype-dependence is asserted rather than assumed.

Second, **eigenvector sign and ordering.** The Fiedler vector and any
eigenvector-derived quantity have a sign ambiguity, and near-degenerate spectra
can reorder eigenvalues under tiny perturbations. Where the book reports an
eigen-derived number (the Fiedler value, the spectral radius), it reports an
invariant of the eigenproblem — an eigenvalue, or an absolute value — never a
raw signed eigenvector component.

## C.6 Hardware notes

No special hardware is required. The memristor chapter is explicitly **MODELED**:
its 443.9 pJ figure is computed from Ohm's law and crossbar parameters, not read
from a fabricated device, and the chapter says so on every line it appears. The
MCU chapter's footprints are computed from the firmware's actual symbol sizes and
packing, and the book is explicit that the firmware uses `FP_ONE = 255`
(0.8 fixed-point), not 16-bit Q8.8. Nothing in the book claims silicon it did not
touch.
