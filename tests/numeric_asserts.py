# SPDX-License-Identifier: AGPL-3.0-only
"""Numeric-tolerant assertions for printed chapter artifacts.

Floating-point results printed to six or more decimals differ in the last
digits across CPUs and BLAS builds even with identical Python and torch pins.
These helpers compare the *non-numeric* bytes exactly and compare each decimal
token with a declared tolerance. They are deliberately narrow: structural
labels, booleans, integer counts, and all surrounding text must still match.
"""

from __future__ import annotations

import math
import re

REL_TOL = 1e-4
ABS_TOL = 1e-4

# A decimal token with optional exponent, not part of a dotted identifier.
_NUM_RE = re.compile(r"(?<![\d.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\d.])")


def line_matches(expected: str, actual: str) -> bool:
    """True when non-number bytes are byte-identical and numbers are close."""
    e_parts = _NUM_RE.split(expected)
    a_parts = _NUM_RE.split(actual)
    if len(e_parts) != len(a_parts):
        return False
    if any(e != a for e, a in zip(e_parts[::2], a_parts[::2])):
        return False
    e_nums = [float(x) for x in _NUM_RE.findall(expected)]
    a_nums = [float(x) for x in _NUM_RE.findall(actual)]
    return len(e_nums) == len(a_nums) and all(
        math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        for a, b in zip(e_nums, a_nums)
    )


def assert_close_in(expected: str, out: str) -> None:
    """Assert an expected line appears in stdout, modulo decimal-token drift.

    The original ``assert expected in out`` allowed a match as a substring of
    a printed line. Preserve that behaviour: build a pattern whose fixed text
    is escaped and whose decimal tokens are captured, then require a captured
    number close to the expected one at each position.
    """
    parts = _NUM_RE.split(expected)
    want = [float(x) for x in _NUM_RE.findall(expected)]
    pieces = []
    for i, part in enumerate(parts):
        pieces.append(re.escape(part))
        if i < len(want):
            pieces.append("(" + _NUM_RE.pattern + ")")
    pattern = re.compile("".join(pieces))
    for match in pattern.finditer(out):
        got = [float(x) for x in match.groups()]
        if len(got) == len(want) and all(
            math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
            for a, b in zip(got, want)
        ):
            return
    raise AssertionError(f"no numerically matching line for {expected!r} in:\n{out}")
