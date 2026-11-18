# AGENTS.md: instructions for any AI working in this repo

1. Read `REPO_MAP.md` (what is where), then `docs/HANDOFF_2026-11-17.md` (state, results, open tasks) and `CHANGELOG.md` (newest work). The workspace folder above has `index.md` and `MAIN_PIPELINE.md` for the submission. Do not open every file; the READMEs say what each folder holds.
2. Code style: `../HUMAN_CODE_STYLE.md` (plain loops, full names, short casual comments). Design: `DESIGN.md` (notebook is the face, atomic swappable blocks).
3. Never commit, push or overwrite unasked. Keep one copy of `src/`; changed files only in an experiment's own `changed_src/`. Experiments are executed notebooks, not `run_*.py`.
4. A gain counts only if it holds on the old official set, NUDT and the new high-altitude set, with a logical reason. Write the kill rule before the run and report wrong answers (beyond 8 sigma) next to every score. Never tune on the locked test split.
5. Heavy jobs: laptop via `tools/run_capped.sh`; the shared GPU server rules are in `REPO_MAP.md` section 6. Do not touch other people's files or processes there.
6. After an iteration: add a line to `CHANGELOG.md`, an addendum to the handoff, a row in the relevant folder README, and update the Notion results page.
7. Keep personal details (names, emails) out of anything sent to outside services.
