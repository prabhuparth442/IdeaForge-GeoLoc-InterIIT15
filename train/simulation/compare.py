"""Numbers for comparing a real photo with a simulated one (or either with the map). No plotting here."""
import cv2
import numpy as np


def gradient_magnitude(picture, sigma=1.0):
    picture = cv2.GaussianBlur(picture.astype(np.float32), (0, 0), sigma)
    return np.hypot(cv2.Sobel(picture, cv2.CV_32F, 1, 0), cv2.Sobel(picture, cv2.CV_32F, 0, 1))


def image_statistics(picture, valid=None):
    """Plain properties of an image: spread of grey values, how much is edge, how much is fine detail (above a blur of 2 px)."""
    picture = picture.astype(np.float32)
    if valid is None:
        valid = np.ones(picture.shape, bool)
    values = picture[valid]
    edges = gradient_magnitude(picture)[valid]
    fine = (picture - cv2.GaussianBlur(picture, (0, 0), 2.0))[valid]
    scale = values.std() + 1e-6
    return dict(std=float(values.std()), edge_strength=float(edges.mean() / scale), fine_detail=float(fine.std() / scale))


def edge_correlation(first, second, valid=None, search=0):
    """Correlation of the gradient-magnitude pictures of two aligned images (-1..1).

    With search > 0 the second one may be shifted up to `search` pixels each way and the best value is returned,
    so a small alignment error does not hide a good match. Returns (best correlation, shift_x, shift_y).
    """
    edges_first = gradient_magnitude(first)
    edges_second = gradient_magnitude(second)
    if valid is None:
        valid = np.ones(first.shape, bool)
    best = (-1.0, 0, 0)
    for shift_y in range(-search, search + 1):
        for shift_x in range(-search, search + 1):
            moved = np.roll(np.roll(edges_second, shift_y, axis=0), shift_x, axis=1)
            moved_valid = np.roll(np.roll(valid, shift_y, axis=0), shift_x, axis=1) & valid
            if moved_valid.sum() < 100:
                continue
            a = edges_first[moved_valid]
            b = moved[moved_valid]
            value = float(np.corrcoef(a, b)[0, 1])
            if value > best[0]:
                best = (value, shift_x, shift_y)
    return best


# ---------------------------------------------------------------- realism: statistics that do not need the same content
# Real and simulated photos never show the same scene at the same pixel, so these compare *sets* of canvases (north-up, lens
# distortion and tilt already removed by the pipeline's projection) on their central, valid part.

def central_mask(valid, keep_fraction=0.7):
    """valid pixels inside the central ellipse that holds keep_fraction of the canvas (drops the corners, where the warp is strongest)."""
    height, width = valid.shape
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    inside = ((rows - (height - 1) / 2) / (height / 2)) ** 2 + ((cols - (width - 1) / 2) / (width / 2)) ** 2 <= keep_fraction
    return valid & inside


def radial_power(picture, mask, bands=8):
    """Mean power of the picture in `bands` rings of spatial frequency (low to high), from the masked-and-windowed picture."""
    picture = picture.astype(np.float32)
    fill = float(picture[mask].mean()) if mask.any() else 0.0
    flat = np.where(mask, picture, fill) - fill
    window = np.outer(np.hanning(picture.shape[0]), np.hanning(picture.shape[1])).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(flat * window))) ** 2
    rows, cols = np.mgrid[0:picture.shape[0], 0:picture.shape[1]]
    radius = np.hypot(rows - picture.shape[0] // 2, cols - picture.shape[1] // 2)
    edges = np.geomspace(2.0, radius.max() / np.sqrt(2), bands + 1)
    values = []
    for index in range(bands):
        ring = (radius >= edges[index]) & (radius < edges[index + 1])
        values.append(float(power[ring].mean()) + 1e-9)
    return np.array(values)


def orientation_histogram(picture, mask, bins=8):
    """Share of gradient energy per direction (0..180 degrees, direction-less). Normalised to sum 1."""
    blurred = cv2.GaussianBlur(picture.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    weight = np.hypot(gx, gy)
    histogram, _ = np.histogram(angle[mask], bins=bins, range=(0, np.pi), weights=weight[mask])
    return histogram / (histogram.sum() + 1e-9)


SUMMARY_NAMES = ["q05", "q25", "q75", "q95", "skew", "edge_strength", "fine_detail", "power_slope_low", "power_slope_high",
                 "orientation_entropy", "grad_kurtosis"]


def summary_vector(canvas, valid, keep_fraction=0.7):
    """11 numbers per canvas (names in SUMMARY_NAMES): grey-value shape, edge and detail level, spectrum slopes, direction spread."""
    mask = central_mask(valid, keep_fraction)
    values = canvas[mask].astype(np.float64)
    median = np.median(values)
    spread = values.std() + 1e-6
    quantiles = np.percentile(values, [5, 25, 75, 95])
    gradient = gradient_magnitude(canvas)[mask]
    power = np.log(radial_power(canvas, mask))
    positions = np.arange(len(power))
    low = np.polyfit(positions[:4], power[:4], 1)[0]
    high = np.polyfit(positions[4:], power[4:], 1)[0]
    histogram = orientation_histogram(canvas, mask)
    entropy = -float((histogram * np.log(histogram + 1e-9)).sum())
    kurtosis = float(((gradient - gradient.mean()) ** 4).mean() / (gradient.var() ** 2 + 1e-9))
    stats = image_statistics(canvas, mask)
    return np.array([(quantiles[0] - median) / spread, (quantiles[1] - median) / spread, (quantiles[2] - median) / spread,
                     (quantiles[3] - median) / spread, float(((values - values.mean()) ** 3).mean() / spread ** 3),
                     stats["edge_strength"], stats["fine_detail"], low, high, entropy, kurtosis])


def frechet_distance(features_a, features_b, dims=24):
    """Frechet distance between two sets of feature vectors (rows), after a shared PCA to `dims` numbers (stable for a few hundred samples).

    0 for the same distribution. Only comparable between runs that use the same features, dims and sample counts.
    """
    from scipy import linalg
    both = np.vstack([features_a, features_b]).astype(np.float64)
    centre = both.mean(axis=0)
    spread = both.std(axis=0) + 1e-9
    _, _, components = np.linalg.svd((both - centre) / spread, full_matrices=False)
    basis = components[:dims].T
    a = ((features_a - centre) / spread) @ basis
    b = ((features_b - centre) / spread) @ basis
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(cov_a @ cov_b)
    root = root.real
    return float(((mean_a - mean_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def classifier_auc(features_a, features_b, folds=5, seed=0):
    """Can a simple classifier tell set A from set B? Cross-validated AUC: 0.5 = cannot tell, 1.0 = always. Returns (mean, std over folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = np.vstack([features_a, features_b])
    labels = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])
    scores = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_ids, test_ids in splitter.split(data, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        model.fit(data[train_ids], labels[train_ids])
        scores.append(roc_auc_score(labels[test_ids], model.predict_proba(data[test_ids])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))
