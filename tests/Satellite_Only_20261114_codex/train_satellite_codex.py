"""Training entry point: intentionally imports no drone/evaluation modules."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch
from torch.nn import functional as F

from encoder_codex import StructuralEncoder, augment, encode, structural

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'results'
MAP = REPO / 'dataset/Dataset_Official_2/268234931/BAND.tif'
SEED = 141126


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patches():
    rng = np.random.default_rng(SEED)
    manifest, arrays = [], {}
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024), rasterio.open(MAP) as ds:
        source = dict(path=str(MAP), size_bytes=MAP.stat().st_size, mtime_ns=MAP.stat().st_mtime_ns,
                      shape=[ds.height, ds.width], crs=str(ds.crs), transform=list(ds.transform))
        for name, count, lo, hi in [('train', 512, 0, .55), ('validation', 128, .60, .75), ('calibration', 256, .80, 1)]:
            pictures = []
            while len(pictures) < count:
                side = int(rng.integers(700, 2401))
                x = int(rng.integers(int(lo*ds.width)+256, int(hi*ds.width)-side-256))
                y = int(rng.integers(256, ds.height-side-256))
                a = ds.read(1, window=Window(x, y, side, side), out_shape=(224, 224), resampling=Resampling.average).astype(np.float32)
                low, high = np.percentile(a, [1, 99])
                if (a > 0).mean() < .99 or high-low < 1:
                    continue
                a = np.clip((a-low)/(high-low), 0, 1)
                pictures.append(a)
                manifest.append(dict(split=name, x=x, y=y, side=side, sha256=hashlib.sha256(a.tobytes()).hexdigest()))
            arrays[name] = np.stack(pictures)
            print('satellite patches', name, len(pictures), flush=True)
    (OUT/'map_windows_codex.json').write_text(json.dumps(dict(source=source, seed=SEED, windows=manifest), indent=2))
    return arrays


def loss_on(model, data, seed, optimise=None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    losses = []
    model.train(optimise is not None)
    for start in range(0, len(order), 32):
        ids = order[start:start+32]
        # Map side and query share the synthetic coverage mask, as at inference.
        pairs = [augment(data[i], rng) for i in ids]
        a = torch.from_numpy(np.stack([p[0] for p in pairs]))
        b = torch.from_numpy(np.stack([structural(data[i], p[1]) for i,p in zip(ids,pairs)]))
        with torch.set_grad_enabled(optimise is not None):
            u, v = model(a), model(b)
            logits = u @ v.T / .1
            target = torch.arange(len(ids))
            loss = (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2
            if optimise is not None:
                optimise.zero_grad()
                loss.backward()
                optimise.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def calibrate(model, data):
    rng = np.random.default_rng(SEED+40)
    records = []
    with torch.no_grad():
        for i, a in enumerate(data):
            query, mask = augment(a, rng)
            q = model(torch.from_numpy(query[None])).numpy()[0]
            # Mine the nearest among every patch in this calibration stripe,
            # then keep 49 negatives; disallow overlapping geographic footprints.
            windows = json.loads((OUT/'map_windows_codex.json').read_text())['windows'][-len(data):]
            w = windows[i]
            allowed = [j for j,v in enumerate(windows) if j != i and
                       (w['x']+w['side'] <= v['x'] or v['x']+v['side'] <= w['x'] or
                        w['y']+w['side'] <= v['y'] or v['y']+v['side'] <= w['y'])]
            emb = encode(model, data, [mask]*len(data))
            similarities = emb @ q
            negatives = sorted(allowed, key=lambda j: -similarities[j])[:50]
            for present in (True, False):
                ids = ([i]+negatives[:49]) if present else negatives
                order = sorted(ids, key=lambda j: -similarities[j])
                records.append(dict(query=i, present=present, correct=bool(present and order[0]==i),
                                    cosine=float(similarities[order[0]]), margin=float(similarities[order[0]]-similarities[order[1]])))
    # Finite fixed grid; conservative all-abstain default if no usable calibration gate.
    best = dict(cosine=1.01, margin=2.01, retained_correct=0)
    for c in np.linspace(0, 1, 101):
        for m in np.linspace(0, .5, 101):
            chosen = [r for r in records if r['cosine'] >= c and r['margin'] >= m]
            if chosen and all(r['correct'] for r in chosen) and len(chosen) > best['retained_correct']:
                best = dict(cosine=float(c), margin=float(m), retained_correct=len(chosen))
    best.update(calibration_positive_sets=len(data), calibration_null_sets=len(data), empirical_false_accepts=0,
                caveat='Finite satellite calibration only; no thermal error-rate guarantee.')
    (OUT/'calibration_tasks_codex.json').write_text(json.dumps(records, indent=2))
    (OUT/'calibration_codex.json').write_text(json.dumps(best, indent=2))
    return best


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    data = patches()
    model = StructuralEncoder()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best = [], float('inf')
    initial = loss_on(model, data['validation'], SEED+1)
    for epoch in range(12):
        t = time.time()
        train = loss_on(model, data['train'], SEED+10+epoch, optimiser)
        val = loss_on(model, data['validation'], SEED+1)
        history.append(dict(epoch=epoch+1, train_loss=train, validation_loss=val, seconds=time.time()-t))
        print(history[-1], flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), OUT/'encoder_satellite_codex.pt')
    model.load_state_dict(torch.load(OUT/'encoder_satellite_codex.pt', weights_only=True))
    model.eval()
    calibration = calibrate(model, data['calibration'])
    manifest = dict(seed=SEED, initial_validation_loss=initial, epochs=history,
                    parameters=sum(p.numel() for p in model.parameters()), calibration=calibration,
                    provenance='Random initialisation; only raw satellite raster pixels and metadata; no drone inputs.',
                    files={p.name:digest(p) for p in [HERE/'PROTOCOL_codex.md', HERE/'encoder_codex.py', HERE/'train_satellite_codex.py',
                                                     OUT/'encoder_satellite_codex.pt', OUT/'calibration_codex.json', OUT/'map_windows_codex.json']},
                    frozen_at_unix=time.time())
    (OUT/'FROZEN_TRAINING_MANIFEST_codex.json').write_text(json.dumps(manifest, indent=2))
    print('FROZEN', json.dumps(calibration), flush=True)


if __name__ == '__main__':
    main()
