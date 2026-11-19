#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check that the committed CLAIMS.md is exactly what the ledger generates.

Also runs the ledger's own citation audit: every artifact and test must exist,
every pinned value must occur in its test, and every label must be one of the
three permitted labels. Runs in CI via `make claims`.
"""

from __future__ import annotations

import os
import sys

BOOK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BOOK_ROOT, "code"))

from gftac import ch22_claims as ledger  # noqa: E402


def main() -> int:
    errors = []
    for cid, ok, message in ledger.audit_claims():
        if not ok:
            errors.append(f"{cid}: {message}")

    path = os.path.join(BOOK_ROOT, "CLAIMS.md")
    if not os.path.isfile(path):
        errors.append("CLAIMS.md: missing")
    else:
        with open(path, encoding="utf-8") as fh:
            committed = fh.read()
        generated = ledger.render_claims_md()
        if committed != generated:
            errors.append("CLAIMS.md: regenerate with "
                          "`python3 code/gftac/ch22_claims.py`")

    counts = ledger.count_by_label()
    print(f"Claims check: {len(ledger.CLAIMS)} rows, "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for err in errors:
        print("  ✘ " + err)
    if errors:
        print(f"  {len(errors)} problem(s)")
        return 1
    print("  ✔ ledger audits clean and CLAIMS.md is byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
