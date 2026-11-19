#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
verify_book.py — the book's verification runner.

Runs every chapter test file in tests/ and emits a signed verification
certificate (SHA-256) recording exactly which chapters verified green, on
which Python/torch versions, at what time. This is the book keeping its own
contract: "every claim in these pages is asserted by a test."

Usage:
    python verify_book.py            # run all chapter tests, write certificate
    python verify_book.py --quiet    # suppress per-chapter output
    python verify_book.py --check    # verify the committed certificate, write nothing
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

BOOK_ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BOOK_ROOT, "tests")
CERT_PATH = os.path.join(BOOK_ROOT, "book_verification_certificate.json")

_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+) passed(?:\s+(?P<failed>\d+) failed)?"
)


def _sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_artifacts() -> list:
    """Hash every file the published certificate claims to cover.

    The manifest now includes the licensing and provenance files, the build
    tools, and the pinned vendored SDK sources: a reader must be able to tie
    the published certificate to the exact legal terms and parity deps they
    received, not just to the chapter markdown.
    """
    patterns = [
        ("chapter", "chapters/ch*.md"),
        ("appendix", "chapters/appendix_*.md"),
        ("module", "code/gftac/*.py"),
        ("test", "tests/test_ch*.py"),
        ("doc", "README.md"),
        ("doc", "BOOK_PLAN.md"),
        ("doc", "CLAIMS.md"),
        ("doc", "LICENSE"),
        ("doc", "COMMERCIAL_LICENSING.md"),
        ("doc", "THIRD_PARTY_NOTICES.md"),
        ("doc", "PARITY_DEPS.md"),
        ("doc", "LIMITATIONS.md"),
        ("doc", "REPRODUCTION.md"),
        ("doc", "RESEARCH_BRIEF.md"),
        ("doc", "GITHUB_LAUNCH_PLAN.md"),
        ("doc", "CITATION.cff"),
        ("doc", "CHANGELOG.md"),
        ("doc", ".zenodo.json"),
        ("config", ".gitignore"),
        ("config", ".gitattributes"),
        ("config", ".pre-commit-config.yaml"),
        ("config", "pyproject.toml"),
        ("config", "requirements.lock.txt"),
        ("config", ".github/**/*.yml"),
        ("config", ".github/**/*.md"),
        ("tool", "conftest.py"),
        ("tool", "verify_book.py"),
        ("tool", "verify_transcripts.py"),
        ("tool", "build_pdf.py"),
        ("tool", "render_book_figures.py"),
        ("tool", "tools/*.py"),
        ("license", "third_party/licenses/*.txt"),
        ("vendor", "third_party/geoformer/LICENSE"),
        ("vendor", "third_party/geoformer/src/**/*.py"),
        ("vendor", "third_party/TAC/LICENSE"),
        ("vendor", "third_party/TAC/src/**/*.py"),
    ]
    artifacts = []
    for kind, pattern in patterns:
        for path in sorted(glob.glob(os.path.join(BOOK_ROOT, pattern), recursive=True)):
            if os.path.isfile(path):
                artifacts.append({
                    "kind": kind,
                    "path": os.path.relpath(path, BOOK_ROOT),
                    "sha256": _sha256_file(path),
                })
    return artifacts


def _run_chapter_tests(test_file: str, quiet: bool) -> dict:
    """Run one chapter's test file; return its ledger entry."""
    chapter = os.path.basename(test_file).replace("test_", "").replace(".py", "")

    cmd = [
        sys.executable, "-m", "pytest", test_file,
        "-q", "--tb=short", "-p", "no:cacheprovider",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BOOK_ROOT)
    output = (proc.stdout or "") + (proc.stderr or "")

    m = _SUMMARY_RE.search(output)
    passed = int(m.group("passed")) if m else 0
    failed = int(m.group("failed") or 0) if m else (0 if proc.returncode == 0 else 1)
    status = "PASS" if proc.returncode == 0 and failed == 0 else "FAIL"

    if not quiet and status == "FAIL":
        print(output)

    return {
        "chapter": chapter,
        "test_file": os.path.relpath(test_file, BOOK_ROOT),
        "passed": passed,
        "failed": failed,
        "status": status,
        "exit_code": proc.returncode,
    }


def check_certificate() -> int:
    """Validate the committed certificate against the current tree; never write.

    Returns 0 only when the file manifest, the book hash, the certificate hash,
    and every green flag agree. This is the CI gate: it proves the published
    certificate describes exactly the bytes being shipped.
    """
    if not os.path.isfile(CERT_PATH):
        print("[!] No certificate found — run `python3 verify_book.py` first.")
        return 1

    with open(CERT_PATH, encoding="utf-8") as fh:
        recorded = json.load(fh)

    errors = []
    artifacts = _collect_artifacts()
    current = {a["path"]: a["sha256"] for a in artifacts}
    committed = {a["path"]: a["sha256"] for a in recorded.get("artifacts", [])}

    added = sorted(set(current) - set(committed))
    removed = sorted(set(committed) - set(current))
    changed = sorted(p for p in set(current) & set(committed)
                     if current[p] != committed[p])
    for path in added:
        errors.append(f"unhashed new file: {path}")
    for path in removed:
        errors.append("missing hashed file: " + path)
    for path in changed:
        errors.append("hash drift: " + path)

    manifest = "\n".join(f"{a['path']}:{a['sha256']}" for a in artifacts)
    book_sha = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    if book_sha != recorded.get("book_sha256"):
        errors.append(f"book_sha256 mismatch: {book_sha} != {recorded.get('book_sha256')}")

    canonical = dict(recorded)
    cert_sha = canonical.pop("certificate_sha256", None)
    recomputed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()
    if cert_sha != recomputed:
        errors.append(f"certificate_sha256 mismatch: {recomputed} != {cert_sha}")

    for flag in ("all_green", "tests_green", "transcripts_green"):
        if recorded.get(flag) is not True:
            errors.append(f"{flag} is not true")
    if recorded.get("total_failed", 0) or recorded.get("transcripts_failed", 0):
        errors.append("certificate records failures")

    if recorded.get("parity_source") != "vendored":
        errors.append("parity_source is not 'vendored' (parity deps changed?)")

    print("=" * 72)
    print("CHECKING CERTIFICATE: no files will be written")
    print("=" * 72)
    print(f"  {len(artifacts)} artifacts hashed, book {book_sha[:16]}…, "
          f"certificate {str(cert_sha)[:16]}…")
    if errors:
        for err in errors:
            print("  ✘ " + err)
        print(f"  {len(errors)} problem(s) -> certificate does not describe this tree")
        print("=" * 72)
        return 1
    print("  ✔ manifest, book hash, certificate hash and all green flags agree")
    print(f"  ✔ parity from {recorded.get('parity_source')} at "
          f"{recorded.get('parity_commits', {})}")
    print("=" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the GeoFormer & TAC Build Book")
    parser.add_argument("--quiet", action="store_true", help="suppress failure output")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed certificate against this tree; write nothing")
    args = parser.parse_args()

    if args.check:
        return check_certificate()

    test_files = sorted(glob.glob(os.path.join(TESTS_DIR, "test_ch*.py")))
    if not test_files:
        print("[!] No chapter tests found under tests/ — nothing to verify.")
        return 1

    print("=" * 72)
    print("VERIFYING: GeoFormer & TAC — The Build Book")
    print("=" * 72)

    chapters = []
    for tf in test_files:
        entry = _run_chapter_tests(tf, args.quiet)
        chapters.append(entry)
        mark = "✔" if entry["status"] == "PASS" else "✘"
        print(f"  {mark} {entry['chapter']:<42} {entry['passed']} passed"
              + (f", {entry['failed']} FAILED" if entry["failed"] else ""))

    total_passed = sum(c["passed"] for c in chapters)
    total_failed = sum(c["failed"] for c in chapters)
    tests_green = total_failed == 0 and all(c["status"] == "PASS" for c in chapters)

    # --- Transcript check: the book's prose cannot drift from its artifacts. ---
    print("-" * 72)
    print("TRANSCRIPTS: checking every 'Run this now' block against its artifact")
    import verify_transcripts as vt
    transcript_entries, transcript_total, transcript_failed, transcript_volatile = vt.check_all(args.quiet)
    for entry in transcript_entries:
        if entry["status"] != "PASS" and not args.quiet:
            print(f"  ✘ {entry['chapter_md']}: {entry['failures']} mismatch(es)")
    transcripts_green = transcript_failed == 0
    print(f"  {transcript_total} transcripts checked, "
          f"{transcript_failed} mismatch(es), {transcript_volatile} volatile span(s) -> "
          f"{'ALL MATCH ✔' if transcripts_green else 'DRIFT DETECTED ✘'}")

    # --- Artifact hashes: every file the book is built from, tamper-evident. ---
    artifacts = _collect_artifacts()
    book_manifest = "\n".join(f"{a['path']}:{a['sha256']}" for a in artifacts)
    book_sha256 = hashlib.sha256(book_manifest.encode("utf-8")).hexdigest()

    all_green = tests_green and transcripts_green

    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unavailable"

    payload = {
        "book": "GeoFormer & TAC — The Build Book",
        "verification_contract": (
            "Every measured claim in the book is asserted by a chapter test, "
            "and every printed transcript is the byte-for-byte output of the "
            "artifact that produces it; this certificate records both, plus a "
            "SHA-256 manifest of every chapter, module and test file."
        ),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch_version,
        "chapters_verified": len(chapters),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "tests_green": tests_green,
        "transcripts_checked": transcript_total,
        "transcripts_failed": transcript_failed,
        "transcripts_volatile_spans": transcript_volatile,
        "transcripts_green": transcripts_green,
        "artifacts_hashed": len(artifacts),
        "parity_source": "vendored",
        "parity_commits": {
            "geoformer": "aa95ad307d479100a4a1d4a06c9262d706a47dfa",
            "TAC": "4ca7355798f6369df97d7b92e14b9c6807db6d68",
        },
        "book_sha256": book_sha256,
        "all_green": all_green,
        "chapters": chapters,
        "artifacts": artifacts,
    }

    # Signed certificate: SHA-256 over the canonical payload (hash excluded).
    canonical = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    payload["certificate_sha256"] = hashlib.sha256(canonical).hexdigest()

    with open(CERT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("-" * 72)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed "
          f"across {len(chapters)} chapters -> "
          f"{'ALL GREEN ✔' if all_green else 'FAILURES PRESENT ✘'}")
    print(f"  Artifacts: {len(artifacts)} files hashed "
          f"(book_sha256: {book_sha256[:16]}…)")
    print(f"  Certificate: {os.path.relpath(CERT_PATH, BOOK_ROOT)} "
          f"(sha256: {payload['certificate_sha256'][:16]}…)")
    print("=" * 72)

    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
