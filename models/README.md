# models/

Put the two weight files here (they are git-ignored; in the evaluation they are mounted at `/data/models`).

| file | what | licence | where from |
|---|---|---|---|
| `minima_loftr.ckpt` | MINIMA weights for the LoFTR matcher (LoFTR code comes from the `kornia` package) | Apache-2.0 (MINIMA repo and HF card) | MINIMA release |
| `dinov2_vits14_timm.safetensors` | DINOv2 ViT-S/14 weights in `timm` format (code from the `timm` package) | Apache-2.0; the organisers allowed DINOv2 and DINOv3 | `timm/vit_small_patch14_dinov2.lvd142m` |

Nothing else is loaded. In particular there is no SuperPoint, LightGlue, SuperGlue or RoMa here: SuperPoint's licence is research-only and we do not ship it.
Full list for the declaration: `../Submission_Files/DECLARATIONS.md`.
