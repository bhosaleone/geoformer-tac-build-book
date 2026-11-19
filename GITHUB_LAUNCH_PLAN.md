# GITHUB_LAUNCH_PLAN.md — Taking *The Build Book* Public

**Status:** proposal, awaiting owner sign-off and counsel review.
**Scope:** `book_geoformer_tac/` only. Companion repositories are referenced, not republished.
**Date:** 2026-11-19.

---

## 0. The decision in one paragraph

We publish this repository publicly to earn citation, reproducibility credit, and adoption for the
research program. We do **not** publish it under a permissive license, because the code and the
prose are the commercial asset: the build-along library is a teachable on-ramp to the GeoFormer
SDK, and the book is a 24-chapter, 442-test, certificate-backed exposition that a competitor
could otherwise rebrand without cost. The protection is a **dual-license poison pill** already
established in `../geoformer/LICENSE`: AGPLv3 for code (network copyleft — a hosted derivative
must publish its source), CC BY-NC-SA 4.0 for text and figures (no commercial republication,
share-alike), and a paid **commercial proprietary exemption** as the only clean off-ramp. The
repositories's rigor — tests, transcripts, certificate — becomes the launch's marketing surface,
so the CI must re-verify all three on every push and prove the published certificate was produced
from exactly the published bytes.

> **Not legal advice.** This is an engineering and release plan. License text, patent language,
> and the commercial contract must be reviewed by qualified counsel before the repository is
> made public. Nothing here should be read as an opinion on enforceability in any jurisdiction.

---

## 1. Measured starting state

Every number below was read from this directory on 2026-11-19, not estimated.

| Property | Measured value |
|---|---|
| Chapter sources (`chapters/ch*.md`) | 25 (ch00–ch24) |
| Appendices | 4 (A–D) |
| Test modules | 48 (`tests/test_ch*.py`) |
| Build-along modules | 49 (`code/gftac/*.py`, including `__init__.py`) |
| Public claims in `CLAIMS.md` | 31 (25 MEASURED / 4 MODELED / 2 RESEARCH DEMO) |
| Certificate | `all_green: true`, 442 passed / 0 failed, 48 chapters verified |
| Transcripts | 151 checked, 0 mismatches, 3 volatile spans |
| Artifact manifest (pre-publication baseline) | 129 files, `book_sha256 = 41b72dc74f6ae5b280bb23ac9a8041b93081ea2a09f3706a73c0b264567ca1d8` |
| Certificate hash (baseline) | `efd3c911b82ff609509f13b28d0236edb34fa733b031f28380f6b76516f8f699` |
| Artifact manifest (public edition) | 222 files; current `book_sha256` and `certificate_sha256` are in `book_verification_certificate.json` — the hash is deliberately not duplicated in prose, so this document can never drift from the certificate |
| Environment that produced it | Python 3.14.4, torch 2.14.1+cpu |
| PDF | 18.2 MB (`GeoFormer_and_TAC_The_Build_Book.pdf`) |
| Tracked-size candidate | ~35 MB total after vendoring; figures 7.7 MB, build 4.3 MB, vendored SDKs 0.7 MB |
| Git state at planning time | **not a git repository** — `git init` was performed during implementation |
| Secrets / local paths in tree | 0 matches for `/home/shri`, emails, tokens, or private keys (scanned excluding `build/` and the PDF) |

Two facts shape the plan. First, the book is already a verified product; the launch should not
rewrite it, only wrap it. Second, the certificate is environment-bound (Python 3.14.4, torch
2.14.1+cpu) and the parity tests import sibling SDKs directly, so a fresh clone is **not** yet
self-verifying. Fixing that is the main engineering work before we go public.

---

## 2. Repository identity

| Field | Value |
|---|---|
| Name | `geoformer-tac-build-book` (proposed; kebab-case, no `book_` prefix on GitHub) |
| Owner | `bhosaleone` |
| Description | *A verified, build-along exposition of GeoFormer & TAC — every line tested, every claim measured, every number reproducible.* |
| Homepage | Book record DOI `10.5281/zenodo.24358478`; parent program DOI `10.5281/zenodo.23730305` |
| Topics | `geoformer`, `tac`, `thermodynamic-computing`, `ternary-quantization`, `reproducible-research`, `verified-book`, `scientific-ml` |
| Default branch | `main` (protected; required checks = `verify`, `claims`, `license`, `secrets`) |
| Release tags | `v1.0.0`, `v1.0.1`, … semantic versioning; tag only a green certificate |
| License classifier | AGPL-3.0 (GitHub) with a pinned `LICENSE` notice file, not an auto-detected single license |

The DOI is a program-level DOI already printed in the README and the PDF title page. The repo
should link to it, and each release should archive to Zenodo so the book's citation acquires a
versioned, immutable target. Do not mint a second DOI for the repository.

---

## 3. Licensing architecture — the poison pill

### 3.1 What the poison pill actually is

A poison pill is not a scary clause pasted into a README; it is a set of linked obligations that
makes the cheap path (take the code, hide the improvements, sell the result) more expensive than
the honest path (buy a license). Three mechanisms do the work:

**Copyleft on the code.** AGPLv3 §13 treats remote network interaction as distribution: if a
company hosts a modified GeoFormer-derived service, it must offer those users the Corresponding
Source. §5(c) requires modification notices, §10 forbids adding further restrictions, and §11
conditions the patent grant on the same terms. A closed SaaS cannot swallow the library quietly.

**Non-commercial share-alike on the exposition.** `chapters/`, `figures/`, `README.md`,
`BOOK_PLAN.md`, and `CLAIMS.md` are CC BY-NC-SA 4.0. Attribution is required, commercial use is
reserved to the author, and adaptations must carry the same license. A competitor cannot turn
442 pinned tests and 24 chapters into a paid course or a proprietary manual by re-hosting them.

**A priced off-ramp.** The commercial proprietary exemption is the deliberate escape hatch: for
a fee, a company gets the right to embed, host, and sublicense without AGPL §13 disclosure and to
use the exposition commercially. This is the commercial interest the public release protects —
the public version is the demonstration, not the giveaway.

A fourth, softer layer is **reservation**: "GeoFormer", "TAC", "TWIST POOL Labs", "SetMD", and the
book's title are trademarks/identifiers of the author and are not licensed by the copyright
grants. Patent rights beyond the AGPL §11 / CC scope are likewise reserved and addressed only in
the commercial contract. Counsel decides the precise wording; do not draft an enforceability
claim here.

### 3.2 Chosen licenses, file by file

| Path | License | Rationale |
|---|---|---|
| `code/gftac/**/*.py`, `conftest.py`, `verify_*.py`, `build_pdf.py`, `render_book_figures.py`, `tests/**` | AGPL-3.0-only | Matches `../geoformer/LICENSE` §1; network copyleft is the pill. |
| `chapters/**/*.md`, `figures/**`, `README.md`, `BOOK_PLAN.md`, `CLAIMS.md`, `CITATION.cff` | CC BY-NC-SA 4.0 | Matches the book's current README and `../geoformer/LICENSE` §3. |
| `LICENSE` | Dual-licensing notice (not a single SPDX identifier) | Mirrors the geoformer notice format: §1 code, §2 commercial exemption, §3 text, §4 third-party, §5 trademark/patent reservation. |
| `COMMERCIAL_LICENSING.md` | Prose + contact, **not** a contract | States what an exemption covers, how to inquire (`research@twistpool.org`), and that terms are agreed in writing. |
| `build/tex-svg.js`, `build/input/tex/extensions/boldsymbol.js` | MathJax 3, Apache-2.0 | Vendored; must be listed in `THIRD_PARTY_NOTICES.md` with the upstream URL and license text. |
| `.github/**`, `Makefile`, `pyproject.toml`, lockfiles | AGPL-3.0-only (build tooling) | Explicit is better than implicit. |

**Why not SSPL or BSL.** SSPL would force a SaaS operator to release its entire management stack,
and BSL 1.1 would restrict production use for a fixed term before converting to Apache-2.0. Both
are stronger pills in the abstract, but both are non-OSI, legible mainly to lawyers, and they
contradict this book's central promise — "you build everything, from first principles, in runnable
code" — by making that code awkward to use in production. AGPL plus a paid exemption is already
the program's posture, is recognized by GitHub and by users, and keeps the research contract
intact. If counsel later recommends SSPL for a hosted service, that belongs in that service's
repository, not in a teaching book.

### 3.3 Cross-repository consistency check — an open decision

| Repository | Code | Docs | Commercial terms |
|---|---|---|---|
| `geoformer` (public) | AGPL-3.0 | CC BY-NC-SA 4.0 | Exemption offered |
| `book_geoformer_tac` (proposed) | AGPL-3.0 | CC BY-NC-SA 4.0 | Exemption offered |
| `TAC` (public) | MIT | CC BY 4.0 | None stated |
| `geotac_browser_extension` (local) | AGPL-3.0 | mixed | None stated |

The TAC repository is materially more permissive than the SDK and the book. That is a real gap in
the commercial perimeter: a competitor could take TAC's core under MIT while the book that teaches
it is NC-SA. Decide before launch whether to (a) align TAC to the same dual-license notice, (b)
leave it as a deliberately permissive reference and document why, or (c) publish the book without
emphasizing the TAC dependency. Do not silently assume the book's license reaches TAC's code — it
does not.

---

## 4. Self-containment — implemented

The original `conftest.py` looked for `../geoformer/src` and `../TAC/src`, and 20-plus parity
tests imported them directly with no `importorskip` guard. On a clean clone, `make all` would
have failed with `ModuleNotFoundError` — fatal for a book whose contract is self-verification.

**Implemented solution: pinned vendored parity sources.** `third_party/geoformer/` (upstream
commit `aa95ad307d479100a4a1d4a06c9262d706a47dfa`, AGPL-3.0-only) and `third_party/TAC/`
(commit `4ca7355798f6369df97d7b92e14b9c6807db6d68`, MIT) contain only the modules the book
needs. The unrelated `geoformer/security/` package was removed and `geoformer/__init__.py`
trimmed accordingly: it carried machine-specific paths and unrelated security tooling that must
not ship in a public book. All 20 required modules were verified to import with no ambient
`PYTHONPATH`. `conftest.py` prepends the vendored paths; `verify_transcripts.py` sets
`PYTHONPATH` to them for every printed command. The certificate records both commits under
`parity_commits` and `--check` refuses to validate if that field disappears. Provenance, the
exact trim, and a refresh procedure are in `PARITY_DEPS.md` and `THIRD_PARTY_NOTICES.md`.

This is deliberately stronger than the original "pinned install script" proposal: no network,
no ambient install, and no silent reduced mode. Parity is always on, everywhere.

---

## 5. Rigor gates for CI

### 5.1 Certificate verification mode (new)

`verify_book.py` originally only *writes* the certificate, and its `certificate_sha256` covers a
timestamp, so regenerating it always produces a diff. The implemented `--check` mode:

* recompute `_collect_artifacts()` and compare every `path:sha256` to the committed certificate;
* recompute `book_sha256` from the canonical manifest and compare;
* recompute `certificate_sha256` as SHA-256 of the canonical payload with that key removed;
* assert `all_green`, `tests_green`, `transcripts_green`, zero recorded failures, and
  `parity_source == "vendored"`;
* report the recorded Python/torch versions and compare them to `requirements.lock.txt` manually
  at release time (environment strings are informational, not a CI gate);
* never write a file; exit 1 on any mismatch.

The timestamped write mode is kept for releases. `SOURCE_DATE_EPOCH` support remains optional.
This flag turns the certificate from a file into a verifiable claim, which is the whole point of
publishing it.

### 5.2 Required checks on `main`

| Check | Command | Gate |
|---|---|---|
| `certificate` | `python3 verify_book.py --check` | manifest, book hash, certificate hash, green flags, parity source |
| `tests` | `python3 -m pytest tests -q` | 442 passed, 0 failed |
| `transcripts` | `python3 verify_transcripts.py --quiet` | 151 checked, 0 mismatches |
| `claims` | `python3 tools/check_claims.py` | regenerated `CLAIMS.md` byte-identical; 31 rows audited |
| `license` | `python3 tools/check_spdx.py` | every `.py` has an AGPL SPDX header; every chapter carries the CC notice |
| `hygiene` | `python3 tools/check_public.py` | no local paths, private emails, or credential-shaped strings |
| `pdf` (release only) | `python3 build_pdf.py` | PDF builds; hashes recorded in the release notes |

These are implemented in `.github/workflows/verify.yml` and `release.yml`. A
`.pre-commit-config.yaml` covers whitespace, YAML, large files, SPDX, claims,
and the public-hygiene scan. CI runs Python 3.14, the certified environment;
compatibility with 3.10–3.12 is a follow-up, not a release gate.

### 5.3 Release provenance

On a green tag, `.github/workflows/release.yml` attaches the PDF, certificate, `CLAIMS.md`,
`requirements.lock.txt`, and a pip-freeze SBOM manifest. `cosign` signing and Zenodo archiving
through `.zenodo.json` are manual follow-ups. The release notes must quote the hashes from the
committed certificate, not from this planning document.

**Zenodo record `24358478` (DOI `10.5281/zenodo.24358478`) — sync required.** The record already
exists but its metadata is stale relative to the repository: it reports 129 hashed artifacts
(the repository now hashes 222), it carries GPL-3.0-or-later as the book license (wrong — the book
text is CC BY-NC-SA 4.0 and the code is AGPL-3.0-only), its PDF predates the reviewer appendices,
and it has no GitHub repository link. Before the public push, edit the record to: set the license
to CC BY-NC-SA 4.0, update the verification numbers to 442/151/222, upload the freshly built PDF,
add the repository URL as a related identifier, and keep `10.5281/zenodo.23730305` as the parent
program DOI. The repository now cites the book DOI in `README.md`, `CITATION.cff`, and the PDF
title page.

---

## 6. Polishing checklist

**README** — keep the Reader's Contract and the measured badge block, but add, above the book map:
a one-paragraph "what this is", a 30-second quickstart, a "Verification status" badge row, a
"Licensing at a glance" box linking `LICENSE` and `COMMERCIAL_LICENSING.md`, and a "Citation" block
with both BibTeX and the Zenodo DOI. Move the author's institutional affiliation into a short
"About the author" section; the current header is dense for a first-time visitor.

**Docs site** — publish `build/book.html` through GitHub Pages at `/docs/` or a `gh-pages` branch,
with a banner stating that the PDF is the certified edition and the HTML is a convenience view.
Do not let the Pages build mutate the certificate.

**Community files** — `CONTRIBUTING.md` (how to propose a correction: change the artifact *and* the
chapter, then regenerate the certificate), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
`SECURITY.md` (private reporting address, no public issues), `CHANGELOG.md` (Keep a Changelog),
`CITATION.cff` + `.zenodo.json`, `ISSUE_TEMPLATE/{bug,claim-correction,licensing}.yml`,
`PULL_REQUEST_TEMPLATE.md` with a mandatory "certificate regenerated" checkbox, and
`DISCUSSIONS` categories for questions, errata, and commercial licensing (the last pointed at the
email, not answered in public).

**Repository hygiene** — `.gitignore` for `__pycache__/`, `.pytest_cache/`, `.venv/`, `.parity/`,
`build/book_body.md`, `build/book.html`, `build/titlepage.html`; keep `build/template.html` and the
vendored MathJax files because the PDF build needs them. `.gitattributes` to mark `*.pdf` and
`*.png` as binary and to normalise line endings. Consider Git LFS only if the PDF grows past ~50 MB.

**One-time license sweep — done** — 105 code files carry the AGPL SPDX header and all 25 chapter
sources carry the CC SPDX comment. The sweep changed the manifest from 129 to 222 hashed files;
it was followed by a clean `make all` and a fresh certificate. Do not scatter further sweeps
across commits; the certificate must describe a coherent tree.

**Reviewer-facing documents — done.** Three files were added after an external critique of the
book's likely reception: `LIMITATIONS.md` (what internal verification does not prove, synthetic
benchmarks, baseline gaps, statistical provenance, theoretical scope, hardware/clinical
boundaries), `REPRODUCTION.md` (fresh-clone checklist with exact commands), and
`RESEARCH_BRIEF.md` (the ten-page positioning: complementary relational substrate, not a better
Transformer). All three are hashed into the certificate and appended as Appendices F–G of the
PDF. Chapter 14's Welch task was relabeled as a worked example on supplied values, because a
$p$-value computed from hardcoded lists is not benchmark evidence; its transcript and test were
updated in the same commit.

---

## 7. Launch runbook (ordered, each step exits clean before the next)

1. **Counsel review.** Sign off on `LICENSE`, `COMMERCIAL_LICENSING.md`, the trademark/patent
   reservation, and the cross-repo inconsistency in §3.3. Record the decision date.
2. **License sweep + hygiene — done.** All public files carry the right notice; `make all` passed
   on Python 3.14.4 / torch 2.14.1+cpu and the certificate was regenerated.
3. **Parity fix — done.** Pinned SDK subsets are vendored (§4); a fresh clone needs no ambient
   installation. Confirm once more after cloning to a scratch directory.
4. **Scrub — done, repeat before push.** `python3 tools/check_public.py` reports zero hits; review
   `git log -p` before the first push. Anything that should not be public must be removed from
   history *before* pushing, not after.
5. **Init and commit — done locally.** `main` branch and first commit prepared; signing needs a
   verified key on the pushing machine.
6. **Create remote.** `git remote add origin https://github.com/bhosaleone/geoformer-tac-build-book.git`,
   push, enable branch protection with the `verify` workflow required.
7. **CI dry run.** Open a test PR, confirm the gates run and that `--check` fails when a chapter
   hash is deliberately perturbed (then revert). A gate that has never failed is not a gate.
8. **Tag and release.** `git tag -s v1.0.0`; the release workflow attaches the PDF, certificate,
   claims, lockfile, and SBOM. Quote the current hashes from `book_verification_certificate.json`
   in the notes, never from a stale copy of this plan.
9. **Zenodo + announcement.** Archive the tagged release, add the DOI badge and the versioned
   citation to the README, then announce with the verification numbers (442/151/222) and the
   licensing line — not with adjectives.

**Acceptance criteria for "public":** fresh clone + `make all` green; `verify_book.py --check`
green against the committed certificate; `CLAIMS.md` regenerates identically; zero hygiene hits;
`verify` workflow green on `main`; `LICENSE`, `COMMERCIAL_LICENSING.md`,
`THIRD_PARTY_NOTICES.md`, `SECURITY.md`, `CITATION.cff`, and `PARITY_DEPS.md` present; release
assets attached; counsel sign-off recorded.

---

## 8. Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| AGPL text/chapter mismatch with `TAC`'s MIT/CC BY 4.0 | Resolve §3.3 before launch; add a root-level `LICENSE-MATRIX.md` even if repos stay separate | Owner + counsel |
| Certificate is environment-bound and timestamped | `--check` implemented (§5.1), pinned lockfile committed, Python/torch stated in README | Engineering |
| Clean clone fails parity tests | Vendored pinned SDK subsets (§4), parity always on, certificate records `parity_commits` | Engineering |
| Vendored MathJax shipped without notice | `THIRD_PARTY_NOTICES.md` with Apache-2.0 text and upstream URL | Engineering |
| Clinical KG chapter read as medical advice | Keep the existing RESEARCH DEMO label; add a visible disclaimer in README and chapter 19 | Owner |
| 18 MB PDF bloats history | Acceptable now; move to release assets + LFS if it exceeds 50 MB | Engineering |
| AI-assisted drafting disclosure buried in README | Keep it in the preface and in `CITATION.cff` note field | Owner |
| Commercial exemption wording over-promises | `COMMERCIAL_LICENSING.md` says "terms agreed in writing"; no prices or rights promised in-repo | Counsel |
| Someone removes SPDX headers or license footer | `license` required check + SPDX sweep commit | Engineering |

---

## 9. Open decisions requiring the owner

1. Final repository name (`geoformer-tac-build-book` proposed) and whether the `book_geoformer_tac`
   directory is renamed to match.
2. TAC licensing alignment from §3.3 — align, exempt explicitly, or leave silent.
3. Commercial license tiers and price, and whether the book text is included in the exemption.
4. Patent strategy: research non-assertion pledge or full reservation until the commercial contract.
5. Whether to publish the HTML book on GitHub Pages now or only after the v1.0.0 tag.
6. Which email receives security and licensing inquiries; avoid putting a personal address in a
   public footer if an organizational one exists.

---

## 10. Definition of done for this plan

This document is a plan, not an implementation. It is done when the six open decisions in §9 are
answered, counsel has signed §3, and steps 1–9 of §7 have been executed in order with the
acceptance criteria met. Until then the repository stays local.
