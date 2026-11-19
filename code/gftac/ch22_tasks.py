# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 22 — runnable tasks.

Each task is a self-contained experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch22_tasks.py 1        # a single task
    python3 code/gftac/ch22_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch22_tasks.py.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch22_claims import (
    BOOK_ROOT,
    CLAIMS,
    VALID_LABELS,
    audit_claims,
    claims_sha256,
    count_by_label,
    write_claims_md,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§22.2 — Ledger census: how many claims, under which labels."""
    counts = count_by_label()
    print("Claims Ledger Census:")
    print(f"  • Total claims registered:        {len(CLAIMS)}")
    for label in VALID_LABELS:
        print(f"  • {label:<16} claims:        {counts[label]}")
    print(f"  • Every claim carries a valid label: {sum(counts.values()) == len(CLAIMS)}")


def task_2() -> None:
    """§22.3 — Artifact existence audit: every row cites a real artifact."""
    missing = []
    for claim in CLAIMS:
        if not os.path.isfile(os.path.join(BOOK_ROOT, claim.artifact)):
            missing.append((claim.cid, claim.artifact))
    print("Artifact Existence Audit:")
    print(f"  • Artifacts cited:                {len(CLAIMS)}")
    print(f"  • Artifacts present on disk:      {len(CLAIMS) - len(missing)}")
    print(f"  • Missing artifacts:              {len(missing)}")
    print(f"  • All cited artifacts exist:      {len(missing) == 0}")


def task_3() -> None:
    """§22.4 — Test-linkage audit: every row cites a test that exists."""
    missing = []
    for claim in CLAIMS:
        if not os.path.isfile(os.path.join(BOOK_ROOT, claim.test)):
            missing.append((claim.cid, claim.test))
    print("Test-Linkage Audit:")
    print(f"  • Distinct tests cited:           {len({c.test for c in CLAIMS})}")
    print(f"  • Rows with an existing test:     {len(CLAIMS) - len(missing)}")
    print(f"  • Rows with a missing test:       {len(missing)}")
    print(f"  • Every row is test-backed:       {len(missing) == 0}")


def task_4() -> None:
    """§22.5 — Pinned-value cross-check: pinned numbers appear in their tests."""
    pinned = [c for c in CLAIMS if c.pinned is not None]
    failures = []
    for claim in pinned:
        text = open(os.path.join(BOOK_ROOT, claim.test), encoding="utf-8").read()
        if claim.pinned not in text:
            failures.append((claim.cid, claim.pinned))
    print("Pinned-Value Cross-Check:")
    print(f"  • Rows with a pinned value:       {len(pinned)}")
    print(f"  • Pinned values found in test:    {len(pinned) - len(failures)}")
    print(f"  • Pinned values not found:        {len(failures)}")
    print(f"  • All pinned values are asserted: {len(failures) == 0}")


def task_5() -> None:
    """§22.6 — Generate CLAIMS.md from the registry."""
    path = write_claims_md()
    text = open(path, encoding="utf-8").read()
    rows = sum(1 for line in text.splitlines() if line.startswith("| CL-"))
    print("CLAIMS.md Generation:")
    print(f"  • Output file:                    {os.path.relpath(path, BOOK_ROOT)}")
    print(f"  • Ledger rows written:            {rows}")
    print(f"  • File size (bytes):              {len(text.encode('utf-8'))}")
    print(f"  • Ledger SHA-256 (first 16):      {claims_sha256()[:16]}")
    print(f"  • Rows match registry:            {rows == len(CLAIMS)}")


def task_6() -> None:
    """§22.7 — Full audit, plus a live 'command exits 0' probe."""
    results = audit_claims()
    failed = [r for r in results if not r[1]]
    probe = subprocess.run(
        [sys.executable, "code/gftac/ch18_tasks.py", "1"],
        capture_output=True, text=True, cwd=BOOK_ROOT,
    )
    print("Full Ledger Audit:")
    print(f"  • Rows audited:                   {len(results)}")
    print(f"  • Rows passing:                   {len(results) - len(failed)}")
    print(f"  • Rows failing:                   {len(failed)}")
    print(f"  • Live command probe exit code:   {probe.returncode}")
    print(f"  • Ledger is green:                {len(failed) == 0 and probe.returncode == 0}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        tasks = [task_1, task_2, task_3, task_4, task_5, task_6]
    else:
        num = int(sys.argv[1])
        tasks = [{1: task_1, 2: task_2, 3: task_3, 4: task_4, 5: task_5, 6: task_6}[num]]

    for i, t in enumerate(tasks, start=1 if len(tasks) > 1 else int(sys.argv[1])):
        print(f"\n--- Task {i} ---")
        t()


if __name__ == "__main__":
    main()
