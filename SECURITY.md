# Security Policy

This repository is a scientific book plus build-along code. It is not a hosted
service, so there is no live production system to attack. The realistic report
surface is: the verification tooling, the exported MCU/firmware examples in the
chapters, the pinned vendored SDK subsets, and the PDF build pipeline.

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Write to
**research@twistpool.org** with the subject line `SECURITY: <short title>` and
include:

* the file and line or command involved;
* the impact you believe it has (what an attacker could do);
* a minimal reproduction, if you have one;
* whether you have already shared it anywhere else.

We will acknowledge as soon as we can and keep the discussion private until a
fix or a documented decision is available. Please give us a reasonable window
before public disclosure. Licensing questions belong in
`COMMERCIAL_LICENSING.md`, not here.

## Scope notes

* Issues in the upstream `geoformer` or `TAC` projects should be reported to
  those projects as well; our vendored copies are pinned, so a fix here does
  not automatically fix upstream.
* Findings in third-party build tools (pandoc, MathJax, Chromium, pypdf, Z3)
  should go to their maintainers. We will update vendored MathJax if a
  published advisory affects the pinned build.
* Chapter 20's firmware and Chapter 19's clinical KG are **illustrative** and
  carry no safety or clinical assurance. Do not deploy them in a medical or
  safety-critical context.

## Verification integrity reports

If you can show that a published certificate does not describe the published
files, treat that as a security-relevant integrity issue and use the same
private address. Include the certificate hash, the file you believe drifted,
and the output of `python3 verify_book.py --check`.
