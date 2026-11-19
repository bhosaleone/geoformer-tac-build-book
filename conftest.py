# SPDX-License-Identifier: AGPL-3.0-only
"""
PyTest configuration for the GeoFormer & TAC Build Book.

Makes the book's build-along package (``gftac``) importable from the test suite,
and supplies the pinned, vendored companion SDKs (``geoformer`` and ``tac``)
used by the parity tests. The vendored copies come first so a public clone
verifies against exactly the SDK revisions recorded in THIRD_PARTY_NOTICES.md,
regardless of what is installed in the ambient environment.
"""

import os
import sys

BOOK_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BOOK_ROOT)

VENDORED = os.path.join(BOOK_ROOT, "third_party")

_PATHS = [
    os.path.join(BOOK_ROOT, "code"),                      # -> import gftac
    os.path.join(VENDORED, "geoformer", "src"),           # -> import geoformer (pinned)
    os.path.join(VENDORED, "TAC", "src"),                 # -> import tac (pinned)
]

# The parent-directory SDKs are kept only as a development fallback. They are
# appended, never prepended, so they cannot shadow the vendored copies.
_FALLBACK = [
    os.path.join(REPO_ROOT, "geoformer", "src"),
    os.path.join(REPO_ROOT, "TAC", "src"),
]

for _p in _PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

for _p in reversed(_FALLBACK):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
