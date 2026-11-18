"""Small read-only analysis blocks. Paths and plotting stay in the notebook."""
import math

import cv2
import numpy as np
import pandas as pd


HEIGHT_EDGES = [0, 300, 500, 800, 1000, 1300, 1600, 2100]
HEIGHT_NAMES = ['100-300', '300-500', '500-800', '800-1000', '1000-1300', '1300-1600', '1600-2100']


def load_development(metadata_path, split_path):
    metadata = pd.read_csv(metadata_path)
    metadata['source_row'] = np.arange(len(metadata))
    split = pd.read_csv(split_path)
    table = metadata.merge(split[['filename', 'split', 'band']], on='filename', validate='one_to_one')
    development = table[table['split'] == 'dev'].sort_values('source_row').reset_index(drop=True)
    assert len(development) == 332
    return development, metadata, split


def add_height_band(table, height_column='height'):
    result = table.copy()
    result['height_band'] = pd.cut(result[height_column], HEIGHT_EDGES, labels=HEIGHT_NAMES, right=False)
    return result


def summarise_stages(table):
    rows = []
    for band, group in add_height_band(table).groupby('height_band', observed=True):
        answered = group['answered']
        rows.append(dict(band=str(band), photos=len(group), score60=6 * group['points'].mean(),
                         wrong=int((group['points'] < 0).sum()), answered=int(answered.sum()),
                         retrieval_share=float(group['in_K'].mean()), kept_share=float(group['in_M'].mean()),
                         accepted_within1=int((answered & (group['err_sigma'] <= 1)).sum()),
                         median_inliers=float(group['inliers'].median())))
    return pd.DataFrame(rows)


def select_visual_samples(development):
    rows = []
    for band in HEIGHT_NAMES:
        members = development[development['band'] == band].sort_values('alt')
        if len(members) == 0:
            continue
        for fraction in [.25, .75]:
            position = int(round((len(members) - 1) * fraction))
            rows.append(members.iloc[position])
    return pd.DataFrame(rows).reset_index(drop=True)


def read_view(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    if image.ndim == 3:
        image = image.mean(axis=2)
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    return image, np.clip((image - low) / (high - low + 1e-6), 0, 1)


def sample_physics(height, focal, level_cap=11):
    photo_gsd = height / focal
    footprint = 320 * photo_gsd
    requested_level = math.ceil(2 * math.log2(footprint / (224 * .3)))
    level = min(max(requested_level, 3), level_cap)
    canvas_gsd = .3 * 2 ** (level / 2)
    sigma = math.sqrt(64 + (3 * max(height, 100) / focal) ** 2)
    return dict(height_m=height, photo_gsd_m=photo_gsd, footprint_width_m=footprint,
                requested_level=requested_level, capped_level=level, canvas_gsd_m=canvas_gsd,
                coarse_grid_m=4 * canvas_gsd, scoring_sigma_m=sigma,
                coarse_grid_sigma=4 * canvas_gsd / sigma)


def summarise_coarse(table, error_column):
    error = table[error_column].fillna(table['err_c'])
    points = np.select([error <= 1, error <= 2, error <= 4, error <= 8], [10, 6, 3, 0], default=-20)
    return dict(photos=len(table), mean_points10=float(points.mean()), score60=float(6 * points.mean()),
                wrong=int((error > 8).sum()), within1_share=float((error <= 1).mean()))
