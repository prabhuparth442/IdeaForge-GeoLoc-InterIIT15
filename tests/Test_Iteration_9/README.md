# Test_Iteration_9: red team — is the decision independent of the input photos? (IN PROGRESS)

**Answer so far: no.** Full status and numbers: [../HANDOFF.md](../HANDOFF.md), section 4 (T9).

- `It1_Offline_Red_Team`: NUDT vs official separable (AUC 0.988); P not calibrated across sets; thin cut margin on NUDT 707; no wrong answers by
  height band (100-351 m); one-feature +-1 SD shifts that create wrong answers.
- `It2_Hardening`: feature hardening does not help; uniform inlier inflation breaks the judge mainly through the 12-inlier override; without
  the override and at cut 0.98 it stays at 0 wrong up to x2 inflation (today's bars: 14-18 wrong at x2 on NUDT 707).
- `Rich_Dump_Perturbed` + `run_perturbed_dumps.sh` (block `train/checks/perturb.py`): real pipeline runs on altered photos — official 5 of 7
  alterations dumped, rest pending. `It3_Altered_Photos` written, not run.
- Still to do: finish the dumps, run It3, REPORT, Notion.
