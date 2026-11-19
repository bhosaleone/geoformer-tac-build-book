# THIRD_PARTY_NOTICES.md

This file records third-party material redistributed with the book. It is not
a grant of rights; each item keeps its upstream license.

---

## 1. MathJax 3 (Apache-2.0)

* Files: `build/tex-svg.js`, `build/input/tex/extensions/boldsymbol.js`
* Upstream: <https://www.mathjax.org/>, <https://github.com/mathjax/MathJax>
* License: Apache License, Version 2.0
* Full text: [`third_party/licenses/Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt)

These files are vendored so the PDF can be typeset without a network
connection. The build script (`build_pdf.py`) fetches them only if missing.

## 2. `geoformer` SDK parity subset (AGPL-3.0-only)

* Path: `third_party/geoformer/`
* Upstream: <https://github.com/bhosaleone/geoformer>
* Upstream commit: `aa95ad307d479100a4a1d4a06c9262d706a47dfa`
* License: AGPL-3.0-only; upstream notice at `third_party/geoformer/LICENSE`,
  canonical text at `third_party/licenses/AGPL-3.0.txt`

Modifications made to the vendored copy:

1. Only source files needed for the book's parity tests were retained.
2. The unrelated `geoformer/security/` package was removed. It is not used by
   this book, and removing it keeps local-machine paths and unrelated security
   tooling out of the public repository.
3. `geoformer/__init__.py` was trimmed to drop the removed security imports and
   their names in `__all__`. Every other exported name is unchanged.

The parity tests import this copy in preference to any ambient installation.

## 3. `TAC` SDK parity subset (MIT)

* Path: `third_party/TAC/`
* Upstream: <https://github.com/bhosaleone/TAC>
* Upstream commit: `4ca7355798f6369df97d7b92e14b9c6807db6d68`
* License: MIT; upstream notice at `third_party/TAC/LICENSE`

Modifications made to the vendored copy: only source files needed for parity
were retained; cache directories were removed. The TAC repository's
documentation is CC BY 4.0 upstream, but this vendored subset contains source
files only, which are MIT-licensed per that repository's `LICENSE`.

## 4. License texts included verbatim

* `third_party/licenses/AGPL-3.0.txt` — GNU Affero General Public License v3
  (<https://www.gnu.org/licenses/agpl-3.0.txt>)
* `third_party/licenses/Apache-2.0.txt` — Apache License 2.0
  (<https://www.apache.org/licenses/LICENSE-2.0.txt>)

## 5. Tools used but not redistributed

pandoc, Playwright/Chromium, pypdf, Python 3.14.4, PyTorch 2.14.1+cpu, SciPy,
pytest, and Z3 are required to build or verify the book. They are not copied
into this repository and remain under their own licenses. No third-party model
weights, datasets, patient records, or customer data are distributed here.

## 6. Provenance statement

The vendored sources were copied from the pinned commits above on 2026-11-19
and hashed into `book_verification_certificate.json`. If you need to confirm
that the vendored copy matches upstream, clone the two repositories at those
commits and compare the `.py` files; only the three documented modifications
in §2 should differ.
