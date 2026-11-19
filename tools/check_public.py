#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pre-publish hygiene scan: no local paths, credentials, or private data.

Text files only. Generated `build/` intermediates are skipped (they are
gitignored and rebuilt from the chapters). The upstream geoformer LICENSE is
allowed to retain the author's historical contact line because it is a verbatim
third-party notice that is already public in that repository. This scanner and
the launch plan mention the patterns they detect, so they are allowlisted too.
"""

from __future__ import annotations

import os
import re
import sys

BOOK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LITERAL_PATTERNS = (
    "/home/shri",
    "/Users/",
    "neural Networks",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "shrikantbhosale786@gmail.com",
)
# Credential shapes are matched with a length requirement: the word "task-"
# occurs throughout the book and must not be treated as an API key.
REGEX_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)

ALLOWLIST = {
    os.path.join("third_party", "geoformer", "LICENSE"),
    os.path.join("tools", "check_public.py"),
    "GITHUB_LAUNCH_PLAN.md",
}

SKIP_DIRS = (".git", "__pycache__", ".pytest_cache", ".venv", "build")
TEXT_EXT = (".py", ".md", ".txt", ".json", ".cff", ".yml", ".yaml", ".toml",
            ".html", ".js", ".ino", ".sh", ".cfg", ".ini", "")


def _iter_files():
    for root, dirs, files in os.walk(BOOK_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), BOOK_ROOT)
            if name.endswith(TEXT_EXT) and rel not in ALLOWLIST:
                yield os.path.join(root, name), rel


def main() -> int:
    hits = []
    scanned = 0
    for path, rel in _iter_files():
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in LITERAL_PATTERNS:
                if pattern in line:
                    hits.append(f"{rel}:{lineno}: {pattern}")
            for regex in REGEX_PATTERNS:
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {regex.pattern}")
    print(f"Public-hygiene scan: {scanned} text files, {len(hits)} hit(s)")
    for hit in hits:
        print("  ✘ " + hit)
    if hits:
        return 1
    print("  ✔ no local paths, private emails, or credential-shaped strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
