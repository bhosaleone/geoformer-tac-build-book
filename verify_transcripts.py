#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
verify_transcripts.py — the book's transcript checker.

The chapter tests pin each task's *printed output*. This checker closes the
remaining gap: it extracts every ``python3 code/gftac/...`` command and the
``text`` block that immediately follows it from the chapter markdown, runs the
command, and asserts the transcript in the book is byte-for-byte what the
artifact prints.

A ``text`` block is only checked when a command appears between it and the
previous ``text`` block; inline fragments of a longer ledger that the prose
quotes mid-chapter are therefore skipped rather than mis-paired.

That makes the prose self-checking: a chapter transcript cannot drift from the
code that produces it without this check failing.

Usage:
    python3 verify_transcripts.py            # check every chapter, print a ledger
    python3 verify_transcripts.py --quiet    # only print failures and the summary
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import math
import os
import re
import subprocess
import sys

BOOK_ROOT = os.path.dirname(os.path.abspath(__file__))
CHAPTERS_DIR = os.path.join(BOOK_ROOT, "chapters")

# A command mention anywhere in the prose: python3 code/gftac/<module>[ <arg>]
_CMD_RE = re.compile(
    r"python3\s+code/gftac/(?P<module>[A-Za-z0-9_]+\.py)(?:\s+(?P<arg>[0-9]+|all))?"
)


def _extract_transcripts(md_path: str):
    """Yield (command, expected_lines) for every text block that follows a command."""
    with open(md_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    transcripts = []
    pending_cmd = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _CMD_RE.search(line)
        if m:
            arg = f" {m.group('arg')}" if m.group("arg") else ""
            pending_cmd = f"code/gftac/{m.group('module')}{arg}"
        if line.strip() == "```text":
            block = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                block.append(lines[j])
                j += 1
            if pending_cmd is not None:
                transcripts.append((pending_cmd, block))
                pending_cmd = None          # a command labels one block only
            i = j
        i += 1
    return transcripts


def _sdk_pythonpath() -> str:
    """Vendored SDKs first, so transcripts do not depend on a developer's install."""
    paths = [
        os.path.join(BOOK_ROOT, "third_party", "geoformer", "src"),
        os.path.join(BOOK_ROOT, "third_party", "TAC", "src"),
    ]
    ambient = os.environ.get("PYTHONPATH", "")
    if ambient:
        paths.append(ambient)
    return os.pathsep.join(paths)


def _run_command(command: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = _sdk_pythonpath()
    proc = subprocess.run(
        [sys.executable] + command.split(),
        capture_output=True, text=True, cwd=BOOK_ROOT, env=env,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _normalise(text: str):
    """Rstrip each line and drop leading/trailing blank lines.

    Transcripts are compared modulo surrounding whitespace only; interior
    blank lines and all non-blank content must match exactly.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


# A volatile span: the book shows a representative measured value, and the
# checker matches any run's value there. Everything outside «...» must still
# be byte-for-byte identical, so the volatile set stays explicit and auditable.
_VOLATILE_RE = re.compile(r"«.*?»")

# Decimal tokens printed by artifacts. Last digits move with CPU/BLAS builds
# even at a fixed Python/torch pin, so decimal tokens are compared with an
# explicit tolerance. All non-numeric bytes remain byte-for-byte identical.
_NUM_RE = re.compile(r"(?<![\d.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\d.])")
_REL_TOL = 1e-4
_ABS_TOL = 1e-4


def _numeric_line_matches(expected: str, actual: str) -> bool:
    e_parts = _NUM_RE.split(expected)
    a_parts = _NUM_RE.split(actual)
    if len(e_parts) != len(a_parts):
        return False
    if any(e != a for e, a in zip(e_parts[::2], a_parts[::2])):
        return False
    e_nums = [float(x) for x in _NUM_RE.findall(expected)]
    a_nums = [float(x) for x in _NUM_RE.findall(actual)]
    return len(e_nums) == len(a_nums) and all(
        math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
        for a, b in zip(e_nums, a_nums)
    )


def _line_matches(expected: str, actual: str) -> bool:
    if "«" in expected:
        pattern = ".*".join(re.escape(part) for part in _VOLATILE_RE.split(expected))
        if re.fullmatch(pattern, actual) is None:
            return False
        # Inside a volatile span, any run-dependent value is acceptable; the
        # fixed text around it is still exact. Do not apply numeric tolerance
        # here as well, so the volatile set remains the explicit escape hatch.
        return True
    return _numeric_line_matches(expected, actual)


def _matches(expected_lines, actual_lines) -> bool:
    return len(expected_lines) == len(actual_lines) and all(
        _line_matches(e, a) for e, a in zip(expected_lines, actual_lines)
    )


def check_chapter(md_path: str, quiet: bool) -> dict:
    name = os.path.basename(md_path)
    transcripts = _extract_transcripts(md_path)

    # Run distinct commands once, concurrently.
    commands = sorted({c for c, _ in transcripts})
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        outputs = dict(zip(commands, pool.map(_run_command, commands)))

    failures = []
    volatile_spans = 0
    for command, expected in transcripts:
        actual = _normalise(outputs[command])
        expected_n = _normalise("\n".join(expected))
        volatile_spans += sum(len(_VOLATILE_RE.findall(ln)) for ln in expected_n)
        if not _matches(expected_n, actual):
            failures.append((command, expected_n, actual))

    if failures and not quiet:
        print(f"\n  [x] {name}: {len(failures)} transcript mismatch(es)")
        for command, expected, actual in failures:
            print(f"      command: python3 {command}")
            n = max(len(expected), len(actual))
            for k in range(n):
                e = expected[k] if k < len(expected) else "<missing>"
                a = actual[k] if k < len(actual) else "<missing>"
                if not _line_matches(e, a):
                    print(f"        line {k + 1}:")
                    print(f"          book:   {e}")
                    print(f"          actual: {a}")
                    break

    return {
        "chapter_md": name,
        "transcripts": len(transcripts),
        "volatile_spans": volatile_spans,
        "failures": len(failures),
        "status": "PASS" if not failures else "FAIL",
    }


def check_all(quiet: bool = False):
    """Check every chapter; return (entries, total_transcripts, total_failures, volatile)."""
    md_files = sorted(glob.glob(os.path.join(CHAPTERS_DIR, "ch*.md")))
    entries = [check_chapter(md, quiet) for md in md_files]
    total = sum(e["transcripts"] for e in entries)
    failed = sum(e["failures"] for e in entries)
    volatile = sum(e["volatile_spans"] for e in entries)
    return entries, total, failed, volatile


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify chapter transcripts against artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    md_files = sorted(glob.glob(os.path.join(CHAPTERS_DIR, "ch*.md")))
    if not md_files:
        print("[!] No chapter markdown found under chapters/")
        return 1

    print("=" * 72)
    print("VERIFYING TRANSCRIPTS: GeoFormer & TAC — The Build Book")
    print("=" * 72)

    entries, total, failed, volatile = check_all(args.quiet)
    for entry in entries:
        mark = "✔" if entry["status"] == "PASS" else "✘"
        note = f"  ({entry['volatile_spans']} volatile)" if entry["volatile_spans"] else ""
        print(f"  {mark} {entry['chapter_md']:<44} {entry['transcripts']:>3} transcripts{note}")

    print("-" * 72)
    print(f"  {total} transcripts checked, {failed} mismatch(es), "
          f"{volatile} volatile span(s) -> "
          f"{'ALL MATCH ✔' if failed == 0 else 'DRIFT DETECTED ✘'}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
