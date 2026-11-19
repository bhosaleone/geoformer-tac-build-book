.PHONY: verify check test transcripts claims license all

# Full verification: chapter tests + transcript check + signed certificate.
verify:
	python3 verify_book.py

# Prove the committed certificate describes the current tree (no writes).
check:
	python3 verify_book.py --check

# Chapter tests only.
test:
	python3 -m pytest tests -q

# Transcript check only: every printed block vs. its artifact, byte-for-byte.
transcripts:
	python3 verify_transcripts.py

# Generated ledger is identical to the committed ledger.
claims:
	python3 tools/check_claims.py

# SPDX headers on code and CC notice comments on chapters.
license:
	python3 tools/check_spdx.py

# Everything the book promises, in one shot.
all: test transcripts verify check claims license
