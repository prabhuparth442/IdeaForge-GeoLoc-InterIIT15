# DINOv2 descriptors as an optional hint

Code: `src/coarse_search/dino_hint.py` (class `DinoHint`). Off by default (`settings.USE_DINO = False`); switch on with `python run.py --dino`.

---

## 1. Idea: compare *what is in* two pictures, not *where* things are

The edge correlation compares layout cell by cell. A different approach: ask a pretrained neural network to describe a picture with a vector of numbers (a **descriptor**), and compare pictures through their descriptors. Two pictures of similar content, even if shifted or from different sensors, get close descriptors.

We use **DINOv2** (a Vision Transformer, ViT-S/14), trained by Meta without labels on millions of ordinary photos. It is used *frozen*: never trained on our data.

### What a Vision Transformer does (the minimum you need)

1. The 224 × 224 picture is cut into 14 × 14 pixel patches → 16 × 16 = 256 patches.
2. Each patch is flattened and linearly projected to a 384-number **token**; a position code is added.
3. A stack of transformer blocks refines the tokens. In each block every token *attends* to all others: with queries `Q`, keys `K`, values `V` (linear maps of the tokens)

$$\text{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt d}\Big)V,\qquad d=\text{key size}.$$

   Each token becomes a weighted average of all value vectors, with weights from how well its query matches each key. After a block, a small feed-forward network refines each token.
4. The output is 256 tokens of 384 numbers: one description per patch.

DINO's training trick ("self-distillation with no labels") makes patch tokens meaningful: similar parts of objects get similar tokens.

## 2. Turning tokens into one descriptor per picture

```python
tokens = self.encoder.get_intermediate_layers(x, n=[8], norm=True)[0]
```

Take the tokens after the 9th block (index 8 of 12; mid-depth features are more about local structure, last-layer ones more semantic), layer-normalised. Not every patch is real: the canvas has empty corners. Weight each patch by whether at least half its pixels are real:

```python
real = pixel_valid.reshape(batch, 16, 14, 16, 14).mean(dim=(2, 4)).reshape(batch, -1)
weights = (real >= 0.5).float().unsqueeze(-1)
vectors = (tokens.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
```

The `reshape(batch, 16, 14, 16, 14)` splits each image axis into (patch index, pixel within patch); averaging over the pixel axes gives the real-pixel fraction per patch. The weighted mean of tokens is the picture's 384-number descriptor. Finally `l2_normalise` scales it to length 1, so a dot product is a cosine similarity in `[−1, 1]`:

$$s=\hat d_{\text{query}}\cdot\hat d_{\text{tile}}.$$

Grey pictures are repeated into 3 identical channels and standardised with the ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, because the network was trained on such inputs.

## 3. The map side: a gallery of tiles

For each zoom level the level picture is cut into 224 × 224 tiles with stride `TILE_STEP = 112` (50 % overlap). Tiles that are less than half real map are skipped. All tiles are described **once, offline** (batches of 32) into `gallery[level]` of shape `(n_tiles, 384)`. `tiles[level]` keeps their top-left pixel corners and `tile_centres[level]` their centres in metres.

At query time: `scores = gallery[level][near] @ query_vector` is one matrix–vector product giving the similarity of the photo to every tile in the 5 km disc (`near` selects tiles whose centre is within `R + tile/2`).

## 4. Putting the scores on a grid and interpolating

A tile with top-left pixel `(top, left)` goes to raster cell `(top/112, left/112)`. Tiles outside the disc get the mean score (neutral). Heat-map positions are query centres in metres. A tile whose raster index is `k` has its *centre* at `(k + 1)·112` pixels (top-left `k·112` plus half a tile = one step). So a position `y_m` in steps `y_m/(g·112)` corresponds to raster index `y_m/(g·112) − 1`:

```python
row_pos = (geometry["y_m"] / gsd / TILE_STEP - 1).clamp(0, raster.shape[0] - 1.001)
```

Then **bilinear interpolation** between the four surrounding raster cells (`r0, c0` floors; `fr, fc` fractional parts):

$$h=(1-f_r)\big[(1-f_c)R_{r_0c_0}+f_cR_{r_0c_0+1}\big]+f_r\big[(1-f_c)R_{r_0+1,c_0}+f_cR_{r_0+1,c_0+1}\big].$$

`clamp(..., shape−1.001)` keeps `r0 + 1` inside the array. The result `prior_map` has the same shape as the heat map.

## 5. How it is used: a hint, not a decision

The first approach of the project (v0.4.1) used DINO to retrieve the top 10–50 tiles and checked those. It lost photos of repetitive scenes. Reason: a descriptor is a *summary of content*: two city tiles with the same mix of roofs and roads are close, whichever tile is the right one. Now:

$$\text{ranking}=\text{ZNCC}+\beta\cdot s,\qquad\beta=0.5.$$

A place that fits the *layout* (ZNCC) and also *looks like* the query gets lifted; a place that matches by chance in layout alone gets no help. The reported dense score, ratio and sharpness still use the plain ZNCC.

## 6. Inputs, outputs, constraints

* `DinoHint(level_images)` builds the whole gallery (minutes on a GPU; slow on CPU). Needs `timm` and the ViT-S weights (`timm` downloads them from the internet the first time; **there is no network at evaluation, so ship the weights**).
* `tile_score_raster(level, canvas, valid, prior_m)` → `(raster float32 (rows, cols), cutoff)`.
* `prior_on_grid(raster, geometry)` → torch tensor, same shape as the heat map.
* Memory: the gallery for the development map is small (tens of thousands of 384-vectors).
* On GPU the encoder runs in half precision (`.half()`).
* Status: ported from the notebook, **not yet executed in this repo** (needs `timm`).
