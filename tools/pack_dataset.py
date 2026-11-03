"""Pack the photos + metadata.csv into ONE file, and read it back.

    python tools/pack_dataset.py pack   dataset/d1 dataset/d1.npz
    python tools/pack_dataset.py check  dataset/d1 dataset/d1.npz

The file is a plain numpy .npz (no extra libraries, opens on any OS / Colab / Jetson):
    images   uint16 (N, 256, 320)   raw thermal pixels, lossless
    columns  every column of metadata.csv as its own array (text columns as str)
Row i of every array is the same photo.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pack(folder, out_file):
    folder = Path(folder)
    table = pd.read_csv(folder / "images" / "metadata.csv")
    images = np.stack([cv2.imread(str(folder / "images" / name), cv2.IMREAD_UNCHANGED) for name in table["filename"]])
    arrays = {column: table[column].to_numpy() for column in table.columns}
    arrays = {k: (v.astype(str) if v.dtype == object else v) for k, v in arrays.items()}
    np.savez_compressed(out_file, images=images, **arrays)
    print("wrote", out_file, images.shape, images.dtype)


def load(npz_file):
    """Returns (images uint16 (N,256,320), table pandas DataFrame in the original column order)."""
    data = np.load(npz_file, allow_pickle=False)
    names = [k for k in data.files if k != "images"]
    return data["images"], pd.DataFrame({k: data[k] for k in names})


def check(folder, npz_file):
    images, table = load(npz_file)
    original = pd.read_csv(Path(folder) / "images" / "metadata.csv")
    assert list(table.columns) == list(original.columns), "columns differ"
    assert table.equals(original), "metadata differs"
    for i in range(len(table)):
        assert np.array_equal(images[i], cv2.imread(str(Path(folder) / "images" / table["filename"][i]), cv2.IMREAD_UNCHANGED)), table["filename"][i]
    print("ok: %d photos and the metadata are identical to the folder" % len(table))


if __name__ == "__main__":
    {"pack": pack, "check": check}[sys.argv[1]](sys.argv[2], sys.argv[3])
