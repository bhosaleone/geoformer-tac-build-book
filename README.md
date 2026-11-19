# GeoFormer & TAC — The Build Book

### From Thermodynamic First Principles to Trained, Quantized, Integrated Models
### *Every line tested. Every claim measured. Every number reproducible.*

**Author:** Shrikant Bhosale (Atmabhan Pandit) — MBBS Student, Government Medical College (GMC), Sambhajinagar, Maharashtra, India
**Organization:** SetMD │ TWIST POOL Labs │ ISL Research │ NanoCERN Unit
**Book DOI:** [10.5281/zenodo.24358478](https://doi.org/10.5281/zenodo.24358478)  
**Research Program DOI:** [10.5281/zenodo.23730305](https://doi.org/10.5281/zenodo.23730305)
**Companion codebases:** [geoformer](https://github.com/bhosaleone/geoformer) │ [TAC](https://github.com/bhosaleone/TAC)
**Drafting collaboration:** AI-assisted technical writing in the author's research program (OpenCode)

---

## The Reader's Contract

This is not a book *about* GeoFormer and TAC. It is a book in which **you build them**, piece by
piece, and in which **the book itself is verified**:

1. **You build everything.** From the metric lattice to the trained 32-hop reasoner to
   MCU-exportable weights — every component is constructed from first principles in runnable code.
2. **Every claim in these pages is asserted by a test.** If the text says a number, a test pins
   that number. Run the suite and the book re-verifies itself on your machine.
3. **Every measured number is reproducible.** Seeds are pinned. Commands are given. Means and
   standard deviations are over declared seed sets.
4. **Claims are labeled MEASURED, MODELED, or RESEARCH DEMO.** We never blur measurement with
   modeling — and we say exactly which is which.
5. **Your build is parity-checked against the production SDK.** Where a chapter rebuilds a
   component that exists in `geoformer/src`, a test asserts your implementation and the
   production implementation agree. The book verifies the SDK; the SDK verifies the book.

---

## Verification status

This tree is self-verifying on a fresh clone. The shipped certificate records the run that
produced it:

- **442** chapter tests passed, **0** failed across **48** chapters.
- **151** transcripts checked byte-for-byte, **0** mismatches, **3** explicit volatile spans.
- **222** files hashed in the public-edition certificate, including the licensing files, build
  tooling, and pinned parity SDKs (the pre-publication edition hashed 129). `python3
  verify_book.py --check` proves the certificate describes these exact bytes.
- **31** public claims labeled MEASURED (25), MODELED (4), or RESEARCH DEMO (2).

Environment: Python 3.14.4, PyTorch 2.14.1+cpu. See
[`book_verification_certificate.json`](book_verification_certificate.json) for the full manifest
and both SHA-256 hashes. Verify a checkout with:

```bash
git clone https://github.com/bhosaleone/geoformer-tac-build-book.git
cd geoformer-tac-build-book
make all                        # tests + transcripts + certificate check
python3 verify_book.py --check  # prove the committed certificate describes these bytes
```

No extra SDK install is needed: pinned parity copies ship in `third_party/`
([`PARITY_DEPS.md`](PARITY_DEPS.md)).

---

## Quickstart

```bash
# Environment: Python ≥ 3.10, torch ≥ 2.0, scipy, matplotlib (CPU is enough)
cd book_geoformer_tac

# Verify everything and emit the certificate (7 gates)
make all

# Prove the committed certificate describes these exact bytes (writes nothing)
python3 verify_book.py --check

# Or run the test suite directly
python3 -m pytest tests -q

# Or check only the transcripts: every printed block vs. its artifact, byte-for-byte
python3 verify_transcripts.py

# Run any chapter's interleaved runnable tasks (e.g., Chapter 2, task 4)
python3 code/gftac/ch02_tasks.py 4

# Read the book
chapters/ch00_reader_contract.md

# Read the ledger of every public number, with its label and provenance
CLAIMS.md
```

Output of `verify_book.py` is written to
`book_verification_certificate.json` — a signed (SHA-256) record of every chapter test that
passed, every transcript that matched, and a SHA-256 manifest of every chapter, appendix, module,
and test file, with the Python/torch versions that produced it.

---

## How the Book Verifies Itself

The book's trust rests on four machine-checked layers, each of which can fail
loudly and independently:

1. **Chapter tests** (`tests/test_chNN_*.py`) — every mechanism is built and asserted; every
   parity claim against the SDK is tested where a counterpart exists.
2. **Transcript tests** (`tests/test_chNN_tasks.py` + `verify_transcripts.py`) — every
   "Run this now" block in the prose is checked byte-for-byte against the artifact that prints it.
   Run-dependent values are marked as **volatile spans** (`«…»`), which the checker matches by
   pattern while requiring everything else to match exactly.
3. **The claims ledger** (`code/gftac/ch22_claims.py` → `CLAIMS.md`) — every public number carries
   a MEASURED / MODELED / RESEARCH DEMO label, the artifact that produces it, the test that pins
   it, and the command that regenerates it; the ledger audits its own citations.
4. **The certificate** (`book_verification_certificate.json`) — a SHA-256 manifest of the whole
   book plus the per-chapter and per-transcript ledgers. `all_green` requires layers 1 and 2 to
   both pass.


---

## Book Map

| Part | Chapters | What you will have built by the end |
|---|---|---|
| **I — Foundations** | 0–4 | The failure demo, the metric lattice, the TAC relaxer, ternary synapses |
| **II — The GeoFormer Block** | 5–9 | Q-GHA attention, SwiGLU, Q-TAC layer, S₅ projector, the assembled model |
| **III — Training** | 10–14 | Two-timescale training, Kramers optimizer, datasets, evaluation protocol |
| **IV — Emergent Deep Reasoning** | 15–17 | Failure-mode diagnostics, soliton sharpening (with proof), goal-conditioned 32-hop halting |
| **V — Integration & Scale** | 18–21 | 100k-entity sparse memory, clinical KG, MCU export, memristor energy model |
| **VI — Verification & Capstone** | 22–24 | The claims ledger, what Z3 certificates do and don't say, the one-file capstone |
| **Appendices** | A–D | Notation, SDK cross-reference, reproducibility, troubleshooting |

Full chapter-by-chapter specification and writing status: [`BOOK_PLAN.md`](BOOK_PLAN.md)

---

## Positioning and limits — read before citing

This is not a claim to have built a better Transformer. It is a reproducible architecture for
**geometry-driven relational computation**, with thermodynamic convergence, quantized learning,
and structured memory, offered with its full construction and verification framework.

Start with [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md) for the ten-page version,
[`REPRODUCTION.md`](REPRODUCTION.md) to reproduce every number without help, and
[`LIMITATIONS.md`](LIMITATIONS.md) for what the certificate does **not** prove: synthetic-graph
validity, strong baselines, statistical provenance of illustrative examples, theoretical novelty,
and hardware/clinical claims. A falsified claim is a successful contribution.

---

## Repository Layout

```
book_geoformer_tac/
├── chapters/          # The book itself (chNN_*.md) + appendices (appendix_*.md)
├── code/gftac/        # The build-along library + per-chapter runnable tasks (chNN_tasks.py)
├── tests/             # One pytest file per chapter — the book's verification suite
│                      #   (test_chNN_*.py pin the builds; test_chNN_tasks.py pin the transcripts)
├── conftest.py        # Path setup (book code + pinned vendored SDKs in third_party/)
├── third_party/       # Pinned, trimmed geoformer + TAC parity sources and license texts
├── verify_book.py     # Verification runner + --check → certificate JSON
├── verify_transcripts.py  # Byte-for-byte transcript checker (supports «…» volatile spans)
├── tools/             # CI gates: SPDX, claims, public-hygiene checks
├── .github/           # CI, release workflow, issue and PR templates
├── CLAIMS.md          # Generated ledger: every public number, labeled and sourced
├── RESEARCH_BRIEF.md  # Ten-page reviewer brief and positioning
├── REPRODUCTION.md    # Fresh-clone reproduction instructions
├── LIMITATIONS.md     # What the verification does not establish
├── Makefile           # make all | verify | check | test | transcripts | claims | license
└── BOOK_PLAN.md       # Complete table of contents + status ledger
```

## License & Citation

**Code:** AGPL-3.0-only, matching the geoformer SDK. **Book text and figures:** CC BY-NC-SA 4.0,
matching the research program. A **commercial proprietary exemption** is available for uses that
cannot meet those terms; see [`LICENSE`](LICENSE),
[`COMMERCIAL_LICENSING.md`](COMMERCIAL_LICENSING.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The names GeoFormer, TAC and the
organisation names are reserved and are not licensed by either grant.

Chapter 19 is a RESEARCH DEMO and makes no clinical claim. The software and book are provided
without warranty; the certificate is evidence of reproducibility, not fitness for a purpose.

```bibtex
@book{bhosale2026buildbook,
  title  = {GeoFormer \& TAC — The Build Book: From Thermodynamic First Principles to Trained, Quantized, Integrated Models},
  author = {Bhosale, Shrikant},
  year   = {2026},
  note   = {Every line tested; verification certificate included},
  doi    = {10.5281/zenodo.24358478},
  note   = {Every line tested; verification certificate included. Program DOI: 10.5281/zenodo.23730305}
}
```
