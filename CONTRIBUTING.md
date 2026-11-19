# Contributing

Contributions are welcome, but this book has an unusual rule: **the prose and
the artifact have to change together.** A chapter that prints a number must be
backed by the code that produces it, the test that pins it, and the transcript
that checks it. A PR that changes only the markdown will be rejected.

## The correction workflow

1. Open an issue first for anything beyond a typo. State the claim ID from
   `CLAIMS.md` if there is one, and the chapter it appears in.
2. Change the artifact (`code/gftac/chNN_*.py`) and its test
   (`tests/test_chNN_*.py`) so the number is actually produced and pinned.
3. Update the chapter's `text` block to the new output. Run
   `python3 code/gftac/chNN_tasks.py N` and paste the real output; use a
   `«volatile»` span only for genuinely run-dependent values.
4. If the claim is public, update its row in `code/gftac/ch22_claims.py` and
   regenerate `CLAIMS.md` with `python3 code/gftac/ch22_claims.py`.
5. Run `make all`. Then regenerate the certificate with
   `python3 verify_book.py` and confirm with `python3 verify_book.py --check`.
6. In the PR description, quote before/after numbers and the new
   `book_sha256`/`certificate_sha256`. Tick the certificate checkbox in the PR
   template.

## House style

Chapters argue in prose; lists and tables appear only when the object itself is
a list or table. Every public number carries a MEASURED / MODELED / RESEARCH
DEMO label. Theorems are proved in full. Do not add a claim that cannot be
reproduced on a CPU-only machine with pinned seeds.

## Legal and scope

* New `.py` files need an `SPDX-License-Identifier: AGPL-3.0-only` header in
  the first ten lines; new chapters need the CC BY-NC-SA SPDX comment.
* Do not add personal data, credentials, local absolute paths, or patient
  information. Run the secret scan before pushing.
* Do not add new third-party source without a `THIRD_PARTY_NOTICES.md` entry,
  its license text, and a certificate regeneration.
* The `third_party/` SDK subsets are pinned on purpose. Do not "upgrade" them
  in a drive-by PR; see `PARITY_DEPS.md`.

## Checks you can run locally

```bash
make all                        # tests + transcripts + certificate
python3 verify_book.py --check  # certificate vs. tree, no writes
python3 tools/check_claims.py   # CLAIMS.md regenerates identically
python3 tools/check_spdx.py     # license headers and chapter notices
```

By contributing you agree that your contribution is licensed under the same
terms as the file you change, and that you have the right to grant those terms.
