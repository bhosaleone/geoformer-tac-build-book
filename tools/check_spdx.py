#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check that every public source file carries its license marker.

Code must declare AGPL-3.0-only; chapter markdown must declare CC-BY-NC-SA-4.0
in an HTML comment near the top (pandoc drops the comment from the PDF).
This gate runs in CI and locally via `make license`.
"""

from __future__ import annotations

import glob
import os
import sys

BOOK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_SPDX = "SPDX-License-Identifier: AGPL-3.0-only"
TEXT_SPDX = "SPDX-License-Identifier: CC-BY-NC-SA-4.0"


def _first_lines(path: str, n: int) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readline() for _ in range(n))


def main() -> int:
    errors = []
    code_files = (
        sorted(glob.glob(os.path.join(BOOK_ROOT, "code", "gftac", "*.py")))
        + sorted(glob.glob(os.path.join(BOOK_ROOT, "tests", "*.py")))
        + sorted(glob.glob(os.path.join(BOOK_ROOT, "tools", "*.py")))
        + [
            os.path.join(BOOK_ROOT, name)
            for name in ("conftest.py", "verify_book.py", "verify_transcripts.py",
                         "build_pdf.py", "render_book_figures.py")
        ]
    )
    for path in code_files:
        rel = os.path.relpath(path, BOOK_ROOT)
        if not os.path.isfile(path):
            errors.append(f"{rel}: missing file")
            continue
        if CODE_SPDX not in _first_lines(path, 12):
            errors.append(f"{rel}: missing {CODE_SPDX} in first 12 lines")

    for path in sorted(glob.glob(os.path.join(BOOK_ROOT, "chapters", "ch*.md"))):
        rel = os.path.relpath(path, BOOK_ROOT)
        if TEXT_SPDX not in _first_lines(path, 6):
            errors.append(f"{rel}: missing {TEXT_SPDX} in first 6 lines")

    for name in ("LICENSE", "COMMERCIAL_LICENSING.md", "THIRD_PARTY_NOTICES.md",
                 "SECURITY.md", "CITATION.cff", "PARITY_DEPS.md", "LIMITATIONS.md",
                 "REPRODUCTION.md", "RESEARCH_BRIEF.md"):
        if not os.path.isfile(os.path.join(BOOK_ROOT, name)):
            errors.append(f"{name}: required public file missing")

    print(f"SPDX check: {len(code_files)} code files, "
          f"{len(glob.glob(os.path.join(BOOK_ROOT, 'chapters', 'ch*.md')))} chapters")
    for err in errors:
        print("  ✘ " + err)
    if errors:
        print(f"  {len(errors)} problem(s)")
        return 1
    print("  ✔ all license markers and public files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
