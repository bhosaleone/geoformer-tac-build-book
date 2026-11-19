<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 0 — The Reader's Contract

Before you build anything, you are owed a statement of terms. This book is
going to occupy you for a while — twenty-four chapters, each ending in code
you wrote and a test that passes — and you should know, before committing,
exactly what it promises and exactly how you can check that it keeps those
promises. This short chapter is that statement. Read it once now; hold the
book to it for the rest of the journey.

## 0.1 What this book is

This is a build book. Its subject is GeoFormer and TAC — a quantized,
thermodynamically relaxed reasoning engine built on propagation geometry —
and its method is construction. You will not read *about* the metric lattice
that stores knowledge as conductance; you will build it, prove its two
guarantees, and watch its resistance contract under use. You will not be
*told* that ternary synapses compress a model to 1.58 bits per weight; you
will implement the packing, count the bytes, and assert the arithmetic
against the production claim. The TAC relaxer, the geodesic attention, the
two-timescale training loop, the soliton, the goal-conditioned 32-hop
protocol, the sparse hundred-thousand-entity store, the microcontroller
firmware — each arrives in your hands as mathematics first, then as a
recipe, then as code you write into this repository, then as a test that
pins the chapter's claims to your machine.

By the final chapter you will hold three things: a complete working
implementation assembled by you across the chapters; a test suite that
re-verifies every claim in the book on demand; and a signed verification
certificate recording that the whole structure stood up, on your hardware,
on your software versions, at your timestamp. The certificate is the
book's receipt, and it is yours to keep and share.

## 0.2 The five clauses

The contract has five clauses, and every chapter is bound by all of them.

**You build everything.** Every component is constructed from first
principles in code that lives in this repository, under `code/gftac/`,
module by module, in the order the mathematics demands. The production
SDK is never imported to do your work — only to check it, under the fifth
clause below.

**Every claim is asserted by a test.** If the text states a number — an
accuracy, a contraction rate, a byte count, an energy — a test in `tests/`
pins that number, and the chapter's "Run" section shows the command that
re-derives it. This is the clause that makes the book honest by
construction rather than by intention: the text cannot quietly outrun
the artifacts, because the artifacts are the text's citation base.

**Every measured number is reproducible.** Seeds are pinned and printed
alongside every result. Reported values are means over a declared seed
set, accompanied by standard deviations. When the book quotes the
program's published measurements, it names the file they live in.

**Every claim carries its epistemic label.** Numbers in this book are of
exactly three kinds, and each says which it is, everywhere it appears. A
**MEASURED** number was produced by a machine executing the thing, and a
test re-derives it. A **MODELED** number was computed by an analytical
model — device physics, complexity arithmetic — and the model's
assumptions travel with the number; the 443.9 pJ crossbar energy of
Chapter 21 is the canonical example, and it is never quoted unlabeled. A
**RESEARCH DEMO** is a working system built to illustrate behavior, not to
claim benchmark superiority; the clinical engine of Chapter 19 is the
example. The taxonomy exists because blurring measurement with modeling
is how research programs die in review, and this one does not blur.

**Your build is parity-checked against the production SDK.** Where a
chapter rebuilds a component that exists in `geoformer/src` — the ternary
linear layer, the attention, the relaxation layer — the chapter's test
suite asserts that your implementation and the production implementation
produce identical outputs from identical weights. The check runs both
ways: your from-scratch build validates the SDK, and the SDK validates
your understanding. When you finish a parity test, you have not merely
copied an idea; you have confirmed an equivalence.

## 0.3 Environment and first verification

You need Python 3.10 or newer and PyTorch 2.0 or newer. Every experiment
in Parts I through III runs comfortably on a CPU; nothing in this book
requires a GPU. The full stack is `torch`, `scipy`, and `pytest`:

```bash
pip install torch scipy pytest
```

Verify your setup against the book itself — this is the first act of the
contract, and it takes under a minute:

```bash
cd book_geoformer_tac
python3 -m pytest tests -q     # every chapter's claims, at once
python3 verify_book.py         # the signed verification certificate
```

The reference environment for everything printed in these pages was Python
3.14 with torch 2.14.1 on Linux. Your versions will differ, and that is
fine; the certificate records what you ran, so that any number you quote
from this book carries its provenance with it.

## 0.4 How every chapter works

Each chapter follows the same six-beat rhythm, and the rhythm is the
pedagogy. **Theory** comes first, with the mathematics stated precisely
and the theorems proved in full; book theorems are numbered
B.<chapter>.<n> and cross-referenced against the research program's
Q-series where they correspond, so you can trace every result from
classroom to publication. **Recipe** translates the mathematics into an
ordered construction plan — the bridge from symbols to code, written
before any code exists, so the code that follows is never a surprise.
**Build** writes the code into `code/gftac/`, explained as you write it;
the modules are small enough to hold in your head, because they are
organized by chapter rather than by cleverness. **Verify** writes the
tests, with each assertion tied by name to the claim it guards.
**Run** executes the chapter's experiment and shows the exact output the
text quotes — the ledger, the curve, the certificate line. And
**Exercises** hand you the extensions, under the book's standing
discipline: predict in writing before you run, and treat a wrong
prediction as the finding.

Woven between those beats — not parked at the chapter's end — are the
**runnable tasks**. Whenever a claim is worth touching before the text
moves on, the chapter interrupts itself with a short snippet and the
command that executes it (`python3 code/gftac/ch02_tasks.py 4`, say),
shows the exact output the reference machine produced, and reads that
output back to you line by line. The snippet is not illustrative
pseudocode: it is the verbatim body of a numbered task in that chapter's
task module, `code/gftac/chNN_tasks.py` — whose import header supplies
the names, so typing a snippet by hand needs the chapter's imports —
and the test suite executes
every task and checks its printed output — so every transcript you read
in these pages is guaranteed to be what appears in your own terminal.
Reading this book with a keyboard within reach is therefore not
optional framing; it is the difference between watching the guarantees
and holding them.

## 0.5 Notation

The symbols used throughout the book, with the chapter in which each is
forged. Two warnings. First, $\gamma$ is deliberately reused: it is the
ternary scale in Chapter 4 and the soliton exponent in Chapter 16 — each
chapter disambiguates on first use, and the reuse is kept because both
meanings are standard in their literatures. Second, $P$ is a kernel here,
not a probability of a single event; probability mass lives in the vector
$z$ that $P$ acts on.

| Symbol | Meaning | Forged in |
|---|---|---|
| $W \ge 0$ | conductance lattice (edge weights) | Ch. 2 |
| $d_{ij}$ | effective resistance $1/(W_{ij}+\varepsilon)$ | Ch. 2 |
| $\Delta w$, $b_w$ | lattice quantization step and bit budget | Ch. 2 |
| $L$, $L_{\mathrm{dir}}$ | graph Laplacian, directed Laplacian | Ch. 2–3 |
| $P_{\mathrm{dir}}$ | row-stochastic propagation kernel | Ch. 1–2 |
| $\lambda_2$ | Fiedler value (algebraic connectivity) | Ch. 2 |
| $F[s]$ | free energy of an activation state | Ch. 3 |
| $\kappa$, $\tau$ | relaxation rate, relaxation time | Ch. 3 |
| $V(s)$ | bistable (double-well) potential | Ch. 3 |
| $\gamma$ | ternary scale (Ch. 4); soliton exponent (Ch. 16) | Ch. 4, 16 |
| $z$ | probability mass vector over entities | Ch. 1 |
| $\alpha_0$ | true-path to decoy conductance ratio | Ch. 12, 16 |
| $\tau^*$ | emergent halting depth | Ch. 17 |
| $K(S)$ | description-length bound | Ch. 8 |

## 0.6 The honesty ledger, and why it is structural

Most books label their claims somewhere in a preface and then forget the
labels in the body. This book cannot, and the reason is mechanical rather
than moral: the labels are enforced by where the numbers come from. A
MEASURED claim has a test that fails if the number drifts; a MODELED claim
is presented with its model's assumptions on the same page, and its
chapter re-derives the model rather than quoting it; a RESEARCH DEMO is
introduced as a demonstration and never appears in a comparison table
wearing benchmark clothes. Chapter 22 closes the book by generating the
claims ledger programmatically — every public number in the program,
labeled, cited to its artifact, with the command that reproduces it. By
the time you reach it, you will have built every row.

## 0.7 The verification certificate

`verify_book.py` is the book's spine. It runs every chapter's test file,
one by one, and writes `book_verification_certificate.json`: a SHA-256
signed record of which chapters verified green, with which Python and
torch versions, at what timestamp. Run it whenever you want the book's
current standing; run it before you quote any number from these pages;
run it after every change you make, because your edits are bound by the
same contract as the author's. When you finish the final chapter, the
certificate is your graduation record — and if you share the book, it is
the reader's, too.

## 0.8 The book map

The journey has six parts, ordered so that every chapter builds on
forged, tested components and nothing else. Part I finishes the
foundations: the collapse you measured in Chapter 1, the metric lattice,
the TAC relaxer, and ternary synapses. Part II assembles the GeoFormer
block — attention modulated by distance, relaxation inside the block,
the symmetry projector — and the full model. Part III makes it learn:
plasticity, datasets with honest structure control, the training loop,
and the evaluation protocol. Part IV is the frontier — the failure modes
of linear diffusion, the soliton with its theorem and proof, and the
goal-conditioned protocol that reasons to depth thirty-two and halts
itself. Part V carries the result to the places it was built for: the
sparse store at a hundred thousand entities, the clinical knowledge
graph, the microcontroller, the crossbar. Part VI closes the armor — the
claims ledger, the boundaries of the formal certificates, and the
capstone build that runs the entire stack from one file.

## 0.9 A note on voice

This book speaks to you directly, because you are the one doing the
work. "We" means you and the author, building together — and "we" is
earned: when the text says "we proved," you will have executed the
proof; when it says "we measured," the ledger will be on your screen.
The tone is deliberately that of a working laboratory notebook with the
math kept in: precise where precision is possible, plain-spoken where
it is not, and honest everywhere, including about the places where the
approach fails and the models rather than the machines did the work.

---

*Next: Chapter 1 — what the transformer's flat geometry silently assumes,
the four pathologies that follow, and the experiment in which a model
that knows every fact in its world fails every compositional question
about it.*
