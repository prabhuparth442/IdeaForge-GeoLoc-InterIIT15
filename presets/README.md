# Presets
`pv_x.py` is the frozen PV-X configuration (photo view + pose fit + XoFTR for every photo). Off by default; `settings.py` stays the submitted behaviour.
Frozen 2026-11-19. Results and caveats: research repo `tests/Test_Iteration_12/results/photo_view/FREEZE_PV-X.md`.
Third-party: XoFTR code is Apache-2.0 (`third_party/XoFTR/LICENSE`); the MINIMA XoFTR weights file `minima_xoftr.ckpt` must be declared with the other model files.
Not measured yet: Jetson time and memory.
