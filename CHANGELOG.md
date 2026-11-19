# Changelog

All notable changes to the public edition are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-11-19

### Added

- First public edition of the build book: 24 chapters (ch00–ch24) and four
  appendices (A–D), with interleaved runnable tasks.
- Four verification layers: 48 chapter test modules, byte-for-byte transcript
  checking, the 31-row `CLAIMS.md` ledger, and a SHA-256 verification
  certificate.
- Self-contained parity: pinned, trimmed copies of the `geoformer` and `TAC`
  SDKs under `third_party/` so a fresh clone verifies without ambient installs.
- `verify_book.py --check`: validates the committed certificate against the
  current tree without writing, for CI drift detection.
- Public-repository packaging: dual-license notice, commercial licensing
  contact, third-party notices, citation metadata, security policy,
  contribution guide, issue templates, and CI workflows.
- Reviewer-facing documents: `LIMITATIONS.md`, `REPRODUCTION.md`, and
  `RESEARCH_BRIEF.md`, included as Appendices F–G of the certified PDF.
- Book citation DOI `10.5281/zenodo.24358478` wired into README, CITATION.cff,
  and the PDF title page; program DOI `10.5281/zenodo.23730305` retained as parent.

### Verified

- 442 tests passed, 0 failed across 48 chapters.
- 151 transcripts checked, 0 mismatches, 3 explicit volatile spans.
- 31 claims: 25 MEASURED, 4 MODELED, 2 RESEARCH DEMO.

### Changed

- `conftest.py` and `verify_transcripts.py` now prefer the vendored SDK copies
  over any ambient installation, so parity results are reproducible.
- Chapter 14 Task 4 is now explicitly labeled a worked Welch test on supplied
  values, not a five-seed benchmark result. Its transcript and test were updated
  with it. Previously a reader could reasonably mistake the $p$-value for
  benchmark evidence.

[1.0.0]: https://github.com/bhosaleone/geoformer-tac-build-book/releases/tag/v1.0.0
