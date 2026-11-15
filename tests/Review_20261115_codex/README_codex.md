# Isolated review, 15 November 2026

Read REPORT_codex.md first. Findings are backed by CPU reproductions and saved-result arithmetic, with limitations explicitly recorded. No active source edits or replacement model weights are included.

From the IdeaForge repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../venv-geoloc/bin/python tests/Review_20261115_codex/reproduce_findings_codex.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../venv-geoloc/bin/python tests/Review_20261115_codex/audit_snapshot_codex.py
```

The first command asserts that the current defects reproduce; it deliberately does not patch production code. After future fixes, these assertions should be replaced by regression tests of corrected behavior. The second command re-scores20 delivered T10/T11 per-photo CSVs and saves source hashes. Review logs preserve the completed aggregate suite and five separately executed test scripts omitted from that aggregate.

The new review is entirely under this suffixed directory. Removing only this directory reverts this review's additions; retain the user's original Iteration8–11 work and previous reviews.
