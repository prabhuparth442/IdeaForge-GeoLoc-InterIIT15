"""Red-team dataset variants. usage: make_variants.py SRC_DATASET OUT_ROOT  (every 2nd photo of SRC; images/ + metadata.csv per variant; originals kept in metadata_orig.csv for scoring)."""
import os, sys, cv2, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
meta = pd.read_csv(os.path.join(src, "metadata.csv")).iloc[::2].reset_index(drop=True)
rng = np.random.default_rng(11)
N = len(meta)
# extra metadata noise (1x = 1 deg tilt, 2 deg yaw, 2 % height; the earlier nominal prior), independent of the dataset's own noise
dp, dr, dy, dh = rng.normal(size=(4, N))
def write(name, m, img_fn=None):
    d = os.path.join(out, name); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    m.to_csv(os.path.join(d, "metadata.csv"), index=False); meta.to_csv(os.path.join(d, "metadata_orig.csv"), index=False)
    for f in m.filename:
        p = os.path.join(d, "images", f)
        if img_fn is None:
            if not os.path.exists(p): os.link(os.path.join(src, "images", f), p)
        else:
            im = cv2.imread(os.path.join(src, "images", f), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(p, img_fn(im, f))
    print(name, len(m))
def tweak(**kw):
    m = meta.copy()
    for k, fn in kw.items(): m[k] = fn(m)
    return m
write("control", meta.copy())
for k in (1, 2, 3):
    write("noise%dx" % k, tweak(pitch=lambda m: m.pitch + k*dp, roll=lambda m: m.roll + k*dr, yaw=lambda m: m.yaw + 2*k*dy, alt=lambda m: m.alt * (1 + 0.02*k*dh)))
write("scale_p10", tweak(alt=lambda m: m.alt * 1.10)); write("scale_m10", tweak(alt=lambda m: m.alt * 0.90))
write("yaw_bias3", tweak(yaw=lambda m: m.yaw + 3.0)); write("tilt_bias2", tweak(pitch=lambda m: m.pitch + 2.0, roll=lambda m: m.roll + 2.0))
def blur(im, f): return cv2.GaussianBlur(im, (0, 0), 2.0)
def contrast(im, f): return np.clip(128 + 0.4*(im.astype(np.float32) - 128), 0, 255).astype(im.dtype) if im.dtype == np.uint8 else im
def noisy(im, f):
    r = np.random.default_rng(abs(hash(f)) % (2**32)); return np.clip(im.astype(np.float32) + r.normal(0, 12, im.shape), 0, 255).astype(np.uint8) if im.dtype == np.uint8 else im
def invert(im, f): return (255 - im) if im.dtype == np.uint8 else im
def crop(im, f):
    h, w = im.shape[:2]; o = im.copy(); o[:, :w//4] = 0; o[:, 3*w//4:] = 0; return o
for n, fn in (("blur2", blur), ("lowcontrast", contrast), ("noise12", noisy), ("invert", invert), ("sidecrop", crop)):
    write(n, meta.copy(), fn)
# altitude emulation: shrink the photo by 1/k into the centre of the 320x256 frame (black border) and multiply the height by k. Ground pixel x k, footprint of the content unchanged. Pessimistic: a real photo at k x h fills the whole frame.
hi = meta[meta.alt >= 1000].reset_index(drop=True)
for k in (1.25, 1.5, 2.0):
    def shrink(im, f, k=k):
        h, w = im.shape[:2]; s = cv2.resize(im, (int(round(w/k)), int(round(h/k))), interpolation=cv2.INTER_AREA)
        o = np.zeros_like(im); y0, x0 = (h - s.shape[0])//2, (w - s.shape[1])//2; o[y0:y0+s.shape[0], x0:x0+s.shape[1]] = s; return o
    m = hi.copy(); m["alt"] = m.alt * k
    d = os.path.join(out, "alt%.2f" % k); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    m.to_csv(os.path.join(d, "metadata.csv"), index=False); hi.to_csv(os.path.join(d, "metadata_orig.csv"), index=False)
    for f in m.filename: cv2.imwrite(os.path.join(d, "images", f), shrink(cv2.imread(os.path.join(src, "images", f), cv2.IMREAD_UNCHANGED), f))
    print("alt%.2f" % k, len(m))
