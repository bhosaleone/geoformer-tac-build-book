# REPRODUCTION.md — Reproduce the Book Without the Author

The Reader's Contract says every number is reproducible. This document is the independent
reproduction package: a single path from a fresh clone to the same certificate, with nothing
machine-specific required. If any step fails on your machine, that is a bug worth reporting.

---

## 1. Environment

The shipped certificate was produced with **Python 3.14.4** and **PyTorch 2.14.1+cpu** on Linux.
Use the exact patch version if you want byte-identical transcript output: several chapter tasks
print floating-point values to six or more decimals, and Python 3.14.10 changes the last digits
enough to break byte-for-byte comparison. Both GitHub workflows pin `3.14.4` for this reason.
Run-dependent wall times are confined to three declared `«volatile»` spans in the transcripts.
CPU only is enough — no GPU, no network, and no API keys are needed after the clone.

System tools: `pandoc` and Playwright's Chromium are needed **only** to rebuild the PDF. They are
not needed to verify the book.

## 2. Fresh-clone verification

```bash
git clone https://github.com/bhosaleone/geoformer-tac-build-book.git
cd geoformer-tac-build-book

python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r requirements.lock.txt

make all
```

`make all` runs seven gates in order:

| Gate | Command | Expected result |
|---|---|---|
| Chapter tests | `python3 -m pytest tests -q` | 442 passed, 0 failed |
| Transcripts | `python3 verify_transcripts.py --quiet` | 151 checked, 0 mismatches, 3 volatile spans |
| Certificate | `python3 verify_book.py` | 222 artifacts hashed, `all_green: true` |
| Certificate check | `python3 verify_book.py --check` | manifest, book hash, certificate hash and flags agree |
| Claims | `python3 tools/check_claims.py` | 31 rows audited, `CLAIMS.md` byte-identical |
| License | `python3 tools/check_spdx.py` | 105 code files and 25 chapters marked |
| Hygiene | `python3 tools/check_public.py` | 0 hits |

`make all` emits all seven gates; the table lists the direct command for each so you can run any
one of them in isolation.

## 3. Prove the certificate, not just the tests

The important command is:

```bash
python3 verify_book.py --check
```

It does not run tests and writes nothing. It recomputes every SHA-256 in the manifest, recomputes
`book_sha256`, recomputes `certificate_sha256` from the canonical payload, and refuses to validate
if `all_green`, `tests_green`, `transcripts_green`, or `parity_source` disagree. The expected
values are in `book_verification_certificate.json`; do not trust a copy pasted into prose.

## 4. Parity without ambient installs

The book ships pinned SDK sources in `third_party/`. To confirm they are truly self-contained:

```bash
env -u PYTHONPATH python3 -c "
import sys; sys.path[:0]=['third_party/geoformer/src','third_party/TAC/src']
import geoformer.quant.ste, geoformer.layers.tac, tac.proofs.z3_verifier
print('vendored parity imports OK')"
```

The pinned commits and the exact trim applied to the vendored `geoformer` copy are documented in
`PARITY_DEPS.md` and `THIRD_PARTY_NOTICES.md`. An ambient SDK installation cannot shadow them:
`conftest.py` prepends the vendored paths, and `verify_transcripts.py` sets `PYTHONPATH` for every
printed command.

## 5. Run individual chapters

```bash
python3 code/gftac/ch14_tasks.py 4     # one task
python3 code/gftac/ch17_tasks.py all   # every task in chapter 17
python3 code/gftac/ch24_tasks.py 7     # the whole-stack capstone
```

Every printed line is pinned by `tests/test_chNN_tasks.py` and re-checked against the chapter
markdown by `verify_transcripts.py`. If you change an artifact, change its test and the chapter
transcript in the same commit, then regenerate the certificate.

## 6. Rebuild the book (optional)

```bash
sudo apt-get install -y pandoc
python3 -m pip install playwright pypdf
python3 -m playwright install chromium
python3 build_pdf.py
```

This regenerates `GeoFormer_and_TAC_The_Build_Book.pdf` from the chapter sources and the shipped
certificate. The PDF is not part of the certificate manifest; the chapters are.

## 7. Independent reproduction checklist

| Step | Command | Record to report |
|---|---|---|
| Clone at a tag | `git clone --branch v1.0.0 ...` | commit SHA |
| Install | `pip install -r requirements.lock.txt` | `python3 -V`, `torch.__version__` |
| Verify | `make all` | pass/fail counts |
| Check certificate | `python3 verify_book.py --check` | `book_sha256`, `certificate_sha256` |
| Hygiene | `python3 tools/check_public.py` | hit count (expect 0) |
| Try to break a claim | see `LIMITATIONS.md` §8 | artifact and command that falsifies it |

When you report a discrepancy, include the exact command, the full output, your Python/torch
versions, and the certificate hash you were checking. Public issues are welcome; security
problems go to `research@twistpool.org` per `SECURITY.md`.
