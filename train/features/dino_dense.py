"""F3 of patch v0.5.5: dense DINO tokens as one more feature family.

Same idea as the edge cells, but the cell content is "what is here" (DINO), not "which way do the edges run":
  map    DINO runs over every level picture in overlapping windows; the token grid is reduced to 8 channels by a PCA and
         resampled to the 4 x 4 pixel cell grid. Done once per map (cached on disk, it is the slow part).
  query  DINO on the 224 px canvas (16 x 16 tokens), the same PCA, resampled to 56 x 56 cells, masked like the other families.
Then the usual masked ZNCC (coarse_search.correlation) gives a second score map over the whole disc.

KNOWN LIMIT, found while building this (see tests/Test_Iteration_2/README.md): DINO tokens sit on a 14 px grid, so this family is not
shift-invariant. A query identical to a map window is found exactly (score 1.0 at the truth). The same query shifted by a few pixels
is not: on the official map its score at the truth is about -0.1 while places elsewhere reach about 0.85, because the correlation
latches onto grid-phase patterns. Windows with margins, and averaging over shifted copies (PHASES), did not cure it. A phase-exact map
(a separate map per token-grid phase, 49 of them) would, at a large cost. Treat the dense map as an experiment to be judged on real
photos (patch test X6), not as a finished feature.

The patch asks for 16 channels. The pipeline here is built for N_BINS = 8 channels (src/coarse_search/search.py checks it), so
n_components is 8; that is also what the patch recommends for small GPUs. Tokens are smooth (one per 14 px), so this map alone
localises broadly; it is meant to be fused with the sharper edge maps.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from header.settings import CELL, DEVICE, N_BINS, TILE
from image_processing.edge_features import no_data_cells
from train.features.base import Family
from train.features.dino_tokens import PATCH, token_grids
from train.features.learned_net import cell_mask_of


def fit_pca(samples, n_components=N_BINS):
    """samples (N, 384) token vectors. Returns dict(mean, components, explained) with components (384, n_components)."""
    samples = samples.float().to(DEVICE)
    mean = samples.mean(dim=0)
    centred = samples - mean
    _, singular, vectors = torch.linalg.svd(centred, full_matrices=False)
    components = vectors[:n_components].T.contiguous()
    explained = (singular[:n_components] ** 2).sum() / (singular ** 2).sum()
    return dict(mean=mean, components=components, explained=float(explained))


PHASES = ((0, 0), (7, 7), (0, 7), (7, 0))      # shifts in pixels (down, right): half a token


def tokens_to_cells(grids, pca, cells_per_side, shift=(0, 0)):
    """grids (batch, 384, h, w) -> (batch, n_components, cells, cells): PCA, then bilinear resampling to the cell grid.
    shift = how far (down, right) the picture was moved before DINO saw it; the cell grid still refers to the unshifted picture."""
    centred = grids - pca["mean"].view(1, -1, 1, 1)
    reduced = torch.einsum("bchw,ck->bkhw", centred, pca["components"])
    tokens_per_side = reduced.shape[-1]
    centres = CELL * torch.arange(cells_per_side, device=reduced.device, dtype=reduced.dtype) + CELL / 2
    rows = (centres + shift[0]) / (PATCH / 2 * tokens_per_side) - 1
    columns = (centres + shift[1]) / (PATCH / 2 * tokens_per_side) - 1
    grid_x, grid_y = torch.meshgrid(columns, rows, indexing="xy")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(reduced.shape[0], 1, 1, 1)
    return F.grid_sample(reduced, grid, mode="bilinear", padding_mode="border", align_corners=False)


def sample_tokens(level_images, encoder, window=TILE, windows_per_level=150, seed=0):
    """Token vectors from random windows of the map (only windows that are at least half real map), for fitting the PCA."""
    rng = np.random.default_rng(seed)
    pieces = []
    for level, image in level_images.items():
        found = 0
        tries = 0
        while found < windows_per_level and tries < windows_per_level * 20:
            tries += 1
            top = int(rng.integers(0, image.shape[0] - window))
            left = int(rng.integers(0, image.shape[1] - window))
            crop = image[top:top + window, left:left + window]
            if (crop > 0).mean() < 0.5:
                continue
            grids = token_grids(encoder, (crop[None] / 255.0).astype(np.float32))
            pieces.append(grids[0].reshape(grids.shape[1], -1).T.cpu())
            found += 1
    vectors = torch.cat(pieces)
    keep = torch.from_numpy(rng.choice(len(vectors), size=min(len(vectors), 200_000), replace=False))
    return vectors[keep]


def map_token_features(level_image, encoder, pca, window=TILE, margin=0, batch=32, phases=((0, 0),)):
    """(n_components, rows, cols) cells for a whole level picture, built from overlapping windows.
    margin = 0 (default): the windows tile the map and each one is a stand-alone 224 px picture, exactly what the query is; this scored best.
    margin > 0: each window contributes only its middle part, so the windows step by window - 2 * margin.
    phases: every window is shown to DINO once per shift (down, right) and the cells are averaged; ((0, 0),) is the plain version."""
    assert level_image.ndim == 2 and level_image.dtype == np.uint8, "level image must be 2-D uint8"
    assert margin % CELL == 0 and (window - 2 * margin) % CELL == 0, "margin and step must be whole cells"
    step = window - 2 * margin
    cells_window = window // CELL
    cells_margin = margin // CELL
    cells_step = step // CELL
    rows, cols = level_image.shape[0] // CELL, level_image.shape[1] // CELL
    tops = list(range(0, level_image.shape[0], step))
    lefts = list(range(0, level_image.shape[1], step))
    pad = margin + PATCH                                   # room for the window margin and for the shifted copies
    padded = np.zeros((tops[-1] + step + 2 * pad, lefts[-1] + step + 2 * pad), np.float32)
    padded[pad:pad + level_image.shape[0], pad:pad + level_image.shape[1]] = level_image / 255.0
    features = np.zeros((pca["components"].shape[1], (tops[-1] + step) // CELL, (lefts[-1] + step) // CELL), np.float32)

    corners = [(top, left) for top in tops for left in lefts]
    for first in range(0, len(corners), batch):
        chunk = corners[first:first + batch]
        total = 0
        for shift_down, shift_right in phases:
            pictures = []
            for top, left in chunk:
                start_row = top - margin + pad - shift_down
                start_column = left - margin + pad - shift_right
                pictures.append(padded[start_row:start_row + window, start_column:start_column + window])
            cells = tokens_to_cells(token_grids(encoder, np.stack(pictures)), pca, cells_window, (shift_down, shift_right))
            total = total + cells
        middle = (total / len(phases))[:, :, cells_margin:cells_margin + cells_step, cells_margin:cells_margin + cells_step].cpu().numpy()
        for (top, left), block in zip(chunk, middle):
            row, column = top // CELL, left // CELL
            features[:, row:row + cells_step, column:column + cells_step] = block
    features = features[:, :rows, :cols]
    features[:, no_data_cells(level_image)[:rows, :cols]] = 0
    return features


def query_token_features(canvas, valid, encoder, pca, phases=((0, 0),)):
    """(n_components, 56, 56) cells of the query canvas and the (56, 56) mask of cells that are real photo."""
    assert canvas.shape == (TILE, TILE), "canvas must be %d x %d" % (TILE, TILE)
    padded = np.pad(canvas.astype(np.float32), PATCH, mode="edge")
    total = 0
    for shift_down, shift_right in phases:
        picture = padded[PATCH - shift_down:PATCH - shift_down + TILE, PATCH - shift_right:PATCH - shift_right + TILE]
        total = total + tokens_to_cells(token_grids(encoder, picture[None]), pca, TILE // CELL, (shift_down, shift_right))
    cells = (total / len(phases))[0].cpu().numpy()
    cell_valid = cell_mask_of(valid)
    return (cells * cell_valid).astype(np.float32), cell_valid


def cache_name(level, level_image, n_components, window, phases, margin):
    fingerprint = hashlib.md5(level_image[::97, ::97].tobytes()).hexdigest()[:8]
    return "dino_dense_L%d_C%d_w%d_m%d_p%d_%s.npy" % (level, n_components, window, margin, len(phases), fingerprint)


class DenseDinoFamily(Family):
    name = "dino"

    def __init__(self, level_images, levels, encoder, pca, window=TILE, cache_dir=None, log=print, phases=((0, 0),), margin=0):
        super().__init__()
        assert pca["components"].shape[1] == N_BINS, "the pipeline expects %d channels, got %d" % (N_BINS, pca["components"].shape[1])
        self.encoder = encoder
        self.pca = pca
        self.phases = phases
        self.features = {}
        for level in levels:
            path = None
            if cache_dir is not None:
                path = Path(cache_dir) / cache_name(level, level_images[level], N_BINS, window, phases, margin)
            if path is not None and path.exists():
                self.features[level] = np.load(path)
                log("dense DINO level %d: loaded from %s" % (level, path.name))
                continue
            self.features[level] = map_token_features(level_images[level], encoder, pca, window, margin=margin, phases=phases)
            log("dense DINO level %d: built, %s" % (level, self.features[level].shape))
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.features[level])

    def map_features(self, level):
        return self.features[level]

    def query(self, canvas, valid):
        return query_token_features(canvas, valid, self.encoder, self.pca, self.phases)
