"""uint16-aware photo perturbations (the first versions assumed 8-bit and changed nothing)."""
import os, sys, cv2, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
meta = pd.read_csv(os.path.join(src, "metadata.csv")).iloc[::2].reset_index(drop=True)
def write(name, fn):
    d = os.path.join(out, name); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    meta.to_csv(os.path.join(d, "metadata.csv"), index=False); meta.to_csv(os.path.join(d, "metadata_orig.csv"), index=False)
    for f in meta.filename:
        im = cv2.imread(os.path.join(src, "images", f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        cv2.imwrite(os.path.join(d, "images", f), np.clip(fn(im, f), 0, 65535).astype(np.uint16))
    print(name, len(meta))
write("invert2", lambda im, f: im.max() + im.min() - im)
write("lowcontrast2", lambda im, f: im.mean() + 0.4 * (im - im.mean()))
write("noise2", lambda im, f: im + np.random.default_rng(int(f[:6], 16)).normal(0, 0.5 * im.std(), im.shape))
write("blur2b", lambda im, f: cv2.GaussianBlur(im, (0, 0), 1.0))
