# PARITY_DEPS.md — Pinned parity dependencies

The book's Reader's Contract says your build is parity-checked against the
production SDKs. To make that promise true on a machine that has never heard of
this project, the public repository ships pinned SDK sources in `third_party/`.

| Dependency | Pinned commit | Path | License |
|---|---|---|---|
| `geoformer` | `aa95ad307d479100a4a1d4a06c9262d706a47dfa` | `third_party/geoformer/` | AGPL-3.0-only |
| `TAC` | `4ca7355798f6369df97d7b92e14b9c6807db6d68` | `third_party/TAC/` | MIT (source subset) |

`conftest.py` puts these paths ahead of everything else during pytest, and
`verify_transcripts.py` sets `PYTHONPATH` to them when it runs the printed
commands. An ambient SDK installation can therefore never change a parity
result. The certificate records both commits under `parity_commits` and refuses
to validate (`--check`) if that field is missing.

## What was trimmed and why

The vendored `geoformer` copy is a **parity subset**, not a full mirror. Its
unrelated `security/` package was removed and `geoformer/__init__.py` was
trimmed accordingly. The removal keeps local-machine paths and unrelated
security tooling out of a public repository. All 20 modules actually imported
by the book were retained and import cleanly with no ambient paths:

```bash
env -u PYTHONPATH python3 -c "
import sys; sys.path[:0]=['third_party/geoformer/src','third_party/TAC/src']
import geoformer.quant.ste, geoformer.layers.tac, tac.proofs.z3_verifier
print('vendored parity imports OK')"
```

## Refreshing the pin

Do this deliberately, in one commit, and regenerate the certificate:

1. Copy the updated SDK source files into `third_party/<name>/src/`.
2. Apply the same trim (§ "What was trimmed and why") and re-run the import
   smoke test above.
3. Update the commit SHA in this file, in `THIRD_PARTY_NOTICES.md`, and in
   `verify_book.py`'s `parity_commits` payload.
4. Run `make all`, then `python3 verify_book.py`, then
   `python3 verify_book.py --check`.

No network access is needed to verify the book. Network access is needed only
to rebuild the PDF from scratch if the vendored MathJax files are missing.
