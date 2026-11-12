XoFTR (Tuzcuoglu et al., CVPRW 2024), inference-only copy of `src/xoftr` from https://github.com/OnderT/XoFTR (commit e0fbea4).
Removed: the pretraining model and `utils/supervision.py` (training only, needed loguru). Nothing else is edited. Licence: `LICENSE`.
Used by `src/fine_matching/xoftr_matcher.py` with the MINIMA weights `minima_xoftr.ckpt`.
