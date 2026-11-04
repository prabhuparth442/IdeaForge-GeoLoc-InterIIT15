# DINOv2 descriptors as an optional hint

**Code:** [`../dino_hint.py`](../dino_hint.py) (class `DinoHint`). Off by default (`settings.USE_DINO = False`); switch on with `python run.py --dino`.
**Status:** ported from the notebook but **not yet executed in this repo**, because it needs the `timm` package and the downloaded weights, which are not installed where the code was tested. Read this note as a description of what the code is meant to do.

---

## 1. The idea: compare *what is in* two pictures, not *where* things are

The edge correlation of the previous notes is strict: it compares the layout cell by cell. A completely different way to compare pictures is to ask a pretrained neural network to describe each one by a **descriptor**, a vector of a few hundred numbers, and then compare the vectors. Pictures with similar content (a river bend with fields, an industrial yard) get similar vectors, even when they are shifted, slightly rotated, or taken by different sensors.

The network is **DINOv2 ViT-S/14**, trained by Meta on millions of ordinary photos *without labels*. We use it **frozen**: it is never trained on our data and we only read its outputs.

This gives a cheap, rough sense of "does this tile of the map look like my photo?" that is independent of the edge-direction method, which is why it can complement it.

## 2. What a Vision Transformer does (only what you need)

You do not need the full theory. Follow one 224 × 224 picture through the network:

1. **Cut into patches.** The picture is cut into 14 × 14-pixel patches: 224 / 14 = 16 per side, so 16 × 16 = **256 patches**.
2. **Each patch becomes a token.** A patch (196 pixels) is multiplied by a learned matrix into a list of **384 numbers**, the *token*. A code for the patch's position is added so the network knows where each patch was.
3. **Blocks of attention.** There are 12 blocks. In each, every token looks at all others and updates itself as a weighted average of them. In symbols, with three linear views of the tokens called queries `Q`, keys `K` and values `V`:

$$\text{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt d}\Big)V.$$

   In words: for each token, compare its query with every token's key (a dot product, big if they "match"); turn the scores into weights that add to 1 (softmax; dividing by `√d` keeps the numbers in a sensible range); output the weighted average of the values. After the attention a small network refines each token on its own.
4. **Output.** 256 tokens of 384 numbers: one description per patch, now informed by the whole picture.

DINO's training ("self-distillation without labels") makes these patch tokens meaningful: patches of similar things end up with similar tokens.

## 3. From 256 tokens to one descriptor per picture

```python
tokens = self.encoder.get_intermediate_layers(x, n=[8], norm=True)[0]
```

We do not take the last block's output; we take the tokens after the **9th block** (index 8 of 0 to 11). Earlier blocks describe local structure (textures, shapes), later ones are more "semantic" (what kind of object). For matching terrain from above, structure matters more. `norm=True` applies the network's final normalisation to them.

### Ignoring the empty corners

The query canvas has empty corners (the tilted photo does not cover them), and map tiles near the map border can too. We only average tokens whose patch is mostly real:

```python
real = pixel_valid.reshape(batch, 16, 14, 16, 14).mean(dim=(2, 4)).reshape(batch, -1)
weights = (real >= 0.5).float().unsqueeze(-1)
vectors = (tokens.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
```

Read the first line carefully, because the reshape is the trick: a `224`-long image axis is split into `(16 patches) × (14 pixels within a patch)`; the same for the other axis. Averaging over the two "pixels within patch" axes (`dim=(2, 4)`) leaves the **fraction of real pixels in each patch**, a 16 × 16 table; flatten it to 256 numbers. A patch counts if at least half its pixels are real (`>= 0.5`). The last line is a weighted mean: add the weighted tokens, divide by how many patches counted (`clamp(min=1)` avoids dividing by zero).

The result: one 384-number vector per picture. Finally `l2_normalise` rescales it to length 1:

$$\hat d=\frac{d}{\|d\|+10^{-8}},\qquad s=\hat d_{\text{query}}\cdot\hat d_{\text{tile}}\in[-1,1].$$

With unit-length vectors the dot product *is* the cosine similarity, so `s` near 1 means "very alike".

### Input preparation

The network was trained on colour photos, so a grey picture is repeated into 3 identical channels and standardised with the ImageNet averages `(0.485, 0.456, 0.406)` and spreads `(0.229, 0.224, 0.225)`. On a GPU the network runs in half precision to save memory and time.

## 4. The map side: a gallery of tiles, described once

For each zoom level the level picture is cut into 224 × 224 tiles, one every `TILE_STEP = 112` pixels (so tiles overlap by half). Tiles that are less than half real map are skipped. Every tile is described **once, in advance** (batches of 32) and stored:

* `gallery[level]`: array `(n_tiles, 384)` of unit vectors;
* `tiles[level]`: each tile's top-left pixel corner;
* `tile_centres[level]`: each tile's centre in map metres (x east, y south).

At search time the query is described once, then `scores = gallery[level][near] @ query_vector` is **one matrix times vector product**: the similarity of the photo to every tile near the prior (`near` = tiles whose centre is within `R + tile/2` of the prior). That is why the hint is cheap per photo; the cost is the gallery build (minutes on a GPU, slow on a CPU) at the start.

## 5. Putting tile scores on a grid, then interpolating to the heat map

Each tile has a score; the heat map wants a value at every query centre, so two conversions are needed.

**Tile to raster.** A tile with top-left pixel `(top, left)` goes into raster cell `(top/112, left/112)`. Raster cells outside the disc get the mean score, which is neutral (they neither help nor hurt).

**Raster to heat-map positions, with bilinear interpolation.** A tile with raster index `k` has its *centre* at `(k+1)·112` pixels (top-left `k·112` plus half a tile, which is one step). A heat-map position `y_m` is therefore `y_m / (g·112)` steps from the origin, and the matching raster coordinate is that minus 1:

```python
row_pos = (geometry["y_m"] / gsd / TILE_STEP - 1).clamp(0, raster.shape[0] - 1.001)
```

(`clamp(..., shape−1.001)` keeps the next index `r0+1` inside the array.) Between four raster cells we blend by distance, **bilinear interpolation**. With `r0, c0` the integer parts and `fr, fc` the fractional parts:

$$h=(1-f_r)\big[(1-f_c)R_{r_0c_0}+f_cR_{r_0c_0+1}\big]+f_r\big[(1-f_c)R_{r_0+1,c_0}+f_cR_{r_0+1,c_0+1}\big].$$

A tiny example. Raster `[[0.2, 0.6], [0.4, 0.8]]`, a point a quarter of the way down (`fr = 0.25`) and half-way across (`fc = 0.5`):
top edge: 0.5·0.2 + 0.5·0.6 = 0.4; bottom edge: 0.5·0.4 + 0.5·0.8 = 0.6; blend: 0.75·0.4 + 0.25·0.6 = **0.45**.

The code does this for every row and column at once with indexing: `raster[r0][:, c0]` picks the top-left corner values for all positions, and so on. The output `prior_map` has the same shape as the heat map.

## 6. How it is used: as a hint, not as a decision

An earlier version of the project used DINO as the **retrieval** step: take the 10 to 50 most similar tiles and only check those. It lost photos of repetitive scenes. The reason is built into what a descriptor is: a *summary of content*. Two city tiles with the same mixture of roofs and roads get nearly the same descriptor, whichever one is the right one, and a correct tile can fall outside the top 50.

Now the hint only nudges the ranking of the full-search candidates:

$$\text{ranking}=\text{ZNCC}+\beta\cdot s,\qquad\beta=0.5\ (\texttt{settings.DINO\_BETA}).$$

A place that fits the layout (high ZNCC) **and** looks like the photo (high `s`) is lifted; a place that matches the layout by chance and looks nothing like it gets no help. The stored score, ratio and sharpness still use plain ZNCC, so they stay interpretable.

## 7. Inputs, outputs, constraints

* `DinoHint(level_images)`: builds the whole gallery. Needs `timm` and the ViT-S weights. `timm` downloads weights from the internet on first use, so **ship the weights** if the final run has no network.
* `describe(images, pixel_valid)`: `(batch, 224, 224)` float pictures and bool masks to `(batch, 384)` vectors (asserted).
* `query_vector(canvas, valid)`: one unit vector.
* `tile_score_raster(level, canvas, valid, prior_m)` returns `(raster, cutoff)`; `cutoff` is the 30 % quantile of the scores.
* `prior_on_grid(raster, geometry)` returns a torch tensor shaped like the heat map.

## 8. Common mistakes and honest limits

* Using the descriptor similarity as the final answer. It is only a hint.
* Forgetting that the thermal canvas and the visible-light map differ in sensor: the network was not trained on thermal data, so the hint may be weak. Whether it helps is an **empirical question** that this repo has not answered yet.
* Building the gallery at test time without a GPU (it can take a long time).

## 9. Check yourself

1. Why are the descriptors normalised to length 1?
2. A query canvas has a quarter of its patches empty. How does that affect the descriptor?
3. Why can two different tiles have an almost identical descriptor, and why is that a problem for retrieval but not for a small hint?
