#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
build_pdf.py — assemble the integrated PDF for
"GeoFormer & TAC — The Build Book".

Pipeline
--------
1. Concatenate, in book order, the chapters (ch00..ch24), the appendices
   (A–D), and the generated claims ledger (CLAIMS.md).
2. Convert the combined Markdown to a single HTML document with pandoc
   (GitHub-flavoured Markdown + TeX math + implicit figures + a table of
   contents), preserving math as \\(..\\) / \\[..\\] for MathJax.
3. Render the HTML to PDF with headless Chromium (Playwright), typesetting
   the math with a locally vendored, self-contained MathJax.

Output: GeoFormer_and_TAC_The_Build_Book.pdf in the book root.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CHAPTERS = ROOT / "chapters"
MATHJAX = BUILD / "tex-svg.js"
TEMPLATE = BUILD / "template.html"
TITLE = BUILD / "titlepage.html"
BODY_MD = BUILD / "book_body.md"
OUT_HTML = BUILD / "book.html"
OUT_PDF = ROOT / "GeoFormer_and_TAC_The_Build_Book.pdf"
CERT = ROOT / "book_verification_certificate.json"

MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
BOLDSYMBOL_URL = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/input/tex/extensions/boldsymbol.js"
)
EXT_DIR = BUILD / "input" / "tex" / "extensions"

# A4 (210mm) minus 15mm side margins, in CSS pixels at 96 dpi.
PRINT_CONTENT_PX = 680

# Shrink only the rare display equation that is wider than the print column.
# MathJax pins each container's width with an inline `min-width` in `ex`, and
# `ex` is relative to `font-size`; scaling the font-size therefore scales the
# box exactly, so the equation fits without distortion.
FIT_MATH_JS = """
() => {
  const W = document.documentElement.clientWidth;
  const fixed = [];
  document.querySelectorAll('mjx-container[display="true"]').forEach(el => {
    el.style.fontSize = '';
    const w = el.getBoundingClientRect().width;
    if (w > W) {
      const k = W / w;
      el.style.fontSize = k.toFixed(4) + 'em';
      fixed.push([Math.round(w), Number(k.toFixed(3))]);
    }
  });
  return fixed;
}
"""

FOOTER = (
    '<div style="width:100%; font-size:8px; color:#8a97a4;'
    " font-family:'Noto Sans',sans-serif; padding:0 15mm;"
    ' -webkit-print-color-adjust:exact;">'
    '<span style="float:left;">GeoFormer &amp; TAC &#8212; The Build Book</span>'
    '<span style="float:right;"><span class="pageNumber"></span>'
    ' &#8201;/&#8201; <span class="totalPages"></span></span>'
    "</div>"
)


def ordered_sources() -> list[Path]:
    srcs = sorted(CHAPTERS.glob("ch[0-9]*.md"))
    srcs += sorted(CHAPTERS.glob("appendix_*.md"))
    ledger = ROOT / "CLAIMS.md"
    if ledger.exists():
        srcs.append(ledger)
    return srcs


def readme_preface() -> str:
    """README, retitled as a preface and stripped of the title-page lines."""
    meta = (
        "**Author:**", "**Organization:**", "**Research Program DOI:**",
        "**Companion codebases:**", "**Drafting collaboration:**",
    )
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    out = []
    for i, ln in enumerate(lines):
        if i == 0 and ln.startswith("# "):
            out.append("# Preface — About This Edition")
        elif ln.startswith("### From Thermodynamic First Principles"):
            continue
        elif ln.startswith("### *Every line tested"):
            continue
        elif ln.startswith(meta):
            out.append("- " + ln)
        else:
            out.append(ln)
    return "\n".join(out)


def book_plan_appendix() -> str:
    """BOOK_PLAN.md, retitled as the final appendix (the writing ledger)."""
    lines = (ROOT / "BOOK_PLAN.md").read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = "# Appendix E — Book Plan & Writing Ledger"
    return "\n".join(lines)


def review_appendices() -> list[str]:
    """Append the reviewer-facing honesty documents to the certified PDF.

    A reader of the PDF should see the same boundaries as a reader of the
    repository, not discover them only after following a relative link.
    """
    out = []
    for name, title in (
        ("LIMITATIONS.md", "# Appendix F — Limitations & Boundaries"),
        ("RESEARCH_BRIEF.md", "# Appendix G — Research Brief for Reviewers"),
    ):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("# "):
            lines[0] = title
        out.append("\n".join(lines))
    return out


def combine_markdown() -> None:
    parts = [readme_preface()]
    for path in ordered_sources():
        parts.append(path.read_text(encoding="utf-8").strip())
    parts.append(book_plan_appendix())
    parts.extend(review_appendices())
    BODY_MD.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def load_cert() -> dict:
    if CERT.exists():
        try:
            return json.loads(CERT.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_titlepage() -> None:
    cert = load_cert()
    passed = cert.get("total_passed", "—")
    transcripts = cert.get("transcripts_checked", "—")
    artifacts = cert.get("artifacts_hashed", "—")
    n_chapters = len(list(CHAPTERS.glob("ch[0-9]*.md")))
    n_appendices = len(list(CHAPTERS.glob("appendix_*.md")))
    book_sha = str(cert.get("book_sha256", ""))[:16]
    cert_sha = str(cert.get("certificate_sha256", ""))[:16]
    py = cert.get("python", "—")
    torch = cert.get("torch", "—")

    html = f"""<section class="title-page">
  <div class="band"></div>
  <div class="kicker">GeoFormer Research Program</div>
  <h1 class="title">GeoFormer &amp; TAC<br/>The Build Book</h1>
  <div class="subtitle">From Thermodynamic First Principles to Trained,
    Quantized, Integrated Models</div>
  <div class="tagline">Every line tested. Every claim measured.
    Every number reproducible.</div>
  <div class="rule"></div>
  <div class="author">Shrikant Bhosale (Atmabhan Pandit)</div>
  <div class="org">SetMD &#9474; TWIST POOL Labs &#9474; ISL Research &#9474; NanoCERN Unit</div>
  <div class="meta">
    Book DOI: 10.5281/zenodo.24358478 &#183; Research Program DOI: 10.5281/zenodo.23730305<br/>
    Companion codebases: geoformer &#9474; TAC<br/>
    Verified edition &#183; Python {py} &#183; PyTorch {torch}
  </div>
  <div class="cert">
    certificate {cert_sha} &#183; book {book_sha}<br/>
    {n_chapters} chapters &#183; {n_appendices} appendices &#183; {passed} tests passed<br/>
    {transcripts} transcripts checked &#183; {artifacts} artifacts hashed
  </div>
</section>
"""
    TITLE.write_text(html, encoding="utf-8")


def run_pandoc() -> None:
    cmd = [
        "pandoc", str(BODY_MD),
        "-f", "gfm+tex_math_dollars+implicit_figures",
        "-t", "html5",
        "--standalone",
        "--toc",
        "--toc-depth=1",
        f"--mathjax={MATHJAX.as_uri()}",
        "--highlight-style=tango",
        f"--template={TEMPLATE}",
        f"--include-before-body={TITLE}",
        "--metadata", "title=GeoFormer & TAC — The Build Book",
        "-o", str(OUT_HTML),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def render_pdf() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox", "--font-render-hinting=none"]
        )
        page = browser.new_page(
            viewport={"width": PRINT_CONTENT_PX, "height": 1000}
        )
        page.emulate_media(media="print")
        page.goto(OUT_HTML.as_uri(), wait_until="load")
        # Wait for MathJax to finish its initial typeset.
        page.wait_for_function(
            "() => !!(window.MathJax && window.MathJax.startup && window.MathJax.startup.promise)",
            timeout=120_000,
        )
        page.evaluate("window.MathJax.startup.promise")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(400)
        fixed = page.evaluate(FIT_MATH_JS)
        if fixed:
            print(f"  scaled {len(fixed)} over-wide equation(s): {fixed}")
        page.wait_for_timeout(200)
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=FOOTER,
            margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
        )
        browser.close()


def _key(text: str) -> str:
    """Case/format-insensitive key for matching a heading to page text."""
    text = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def postprocess_pdf() -> None:
    """Add document metadata and a chapter/bookmark outline (Chromium omits one)."""
    try:
        import pypdf
    except Exception as exc:  # pragma: no cover
        print(f"  (pypdf unavailable, skipping metadata/bookmarks: {exc})")
        return

    html = OUT_HTML.read_text(encoding="utf-8")
    body = html.split("<main>", 1)[1].split("</main>", 1)[0] if "<main>" in html else html
    heads = re.findall(r"<h([12])\b[^>]*>(.*?)</h\1>", body, flags=re.S)
    titles: list[tuple[int, str]] = []
    for lvl, raw in heads:
        text = html_mod.unescape(re.sub(r"<[^>]+>", "", raw))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            titles.append((int(lvl), text))

    reader = pypdf.PdfReader(str(OUT_PDF))
    n = len(reader.pages)
    page_keys = [_key(reader.pages[i].extract_text() or "") for i in range(n)]

    writer = pypdf.PdfWriter(clone_from=str(OUT_PDF))
    writer.add_metadata({
        "/Title": "GeoFormer & TAC — The Build Book",
        "/Author": "Shrikant Bhosale (Atmabhan Pandit)",
        "/Subject": "A build book: thermodynamic, quantized reasoning from first "
                    "principles to an integrated, verified stack.",
        "/Keywords": "GeoFormer, TAC, metric lattice, ternary quantization, "
                     "thermodynamic reasoning, build book",
        "/Creator": "book_geoformer_tac / build_pdf.py",
    })

    cursor, parent, found = 0, None, 0
    for lvl, text in titles:
        k = _key(text)
        page = None
        for i in range(cursor, n):
            if k in page_keys[i]:
                page = i
                break
        if page is None or not k:
            continue
        if lvl == 1:
            parent = writer.add_outline_item(text, page)
        else:
            writer.add_outline_item(text, page, parent=parent)
        cursor, found = page, found + 1

    tmp = OUT_PDF.with_suffix(".pdf.tmp")
    with open(tmp, "wb") as fh:
        writer.write(fh)
    os.replace(tmp, OUT_PDF)
    print(f"  + metadata and {found} bookmarks written")


def ensure_mathjax() -> None:
    """Vendor MathJax (self-contained SVG build + the one autoloaded extension)."""
    import urllib.request

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    for url, dest in ((MATHJAX_URL, MATHJAX), (BOLDSYMBOL_URL, EXT_DIR / "boldsymbol.js")):
        if not dest.exists():
            print(f"  fetching {dest.name} …")
            urllib.request.urlretrieve(url, dest)


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    ensure_mathjax()
    print("[1/5] combining markdown …")
    combine_markdown()
    print("[2/5] building title page …")
    build_titlepage()
    print("[3/5] converting markdown → HTML with pandoc …")
    run_pandoc()
    print("[4/5] rendering HTML → PDF with Chromium …")
    render_pdf()
    print("[5/5] adding metadata and bookmarks …")
    postprocess_pdf()
    size = OUT_PDF.stat().st_size
    print(f"\n✓ {OUT_PDF.name}  ({size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
