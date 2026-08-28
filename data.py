"""Image/mask loading for the building segmentation project."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = Path("data/massachusetts-buildings")
IGNORE_INDEX = 255


class ClassMap:
    """Named classes with their RGB codes, from label_class_dict.csv."""

    def __init__(self, names, palette):
        self.names = tuple(names)
        self.palette = np.asarray(palette, dtype=np.uint8)

    @classmethod
    def from_csv(cls, path):
        """Read the name,r,g,b palette file that ships with the dataset."""
        df = pd.read_csv(path)
        return cls(df["name"].str.lower(), df[["r", "g", "b"]].to_numpy())

    @property
    def n_classes(self):
        """How many classes the palette declares."""
        return len(self.names)

    def index_of(self, name):
        """Class number of a named class, e.g. `index_of("building")`."""
        return self.names.index(name.lower())

    def to_index(self, mask_rgb):
        """(H, W, 3) RGB mask -> (H, W) class indices; unknown colours get 255."""
        rgb = np.asarray(mask_rgb, dtype=np.int32)
        keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        out = np.full(keys.shape, IGNORE_INDEX, dtype=np.uint8)
        for i, (r, g, b) in enumerate(self.palette):
            out[keys == ((int(r) << 16) | (int(g) << 8) | int(b))] = i
        return out

    def to_rgb(self, index):
        """(H, W) class indices -> (H, W, 3) RGB, for drawing predictions."""
        out = np.zeros((*np.asarray(index).shape, 3), dtype=np.uint8)
        valid = np.asarray(index) != IGNORE_INDEX
        out[valid] = self.palette[np.asarray(index)[valid]]
        return out


def read_rgb(path):
    """Any image file as an (H, W, 3) uint8 array."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def decode_mask(path, class_map):
    """Read a mask file as an (H, W) class-index map."""
    with Image.open(path) as handle:
        if handle.mode == "P":
            handle = handle.convert("RGB")
        array = np.asarray(handle)

    if array.ndim == 3 and array.shape[-1] >= 3:
        return class_map.to_index(array[..., :3])
    band = array[..., 0] if array.ndim == 3 else array
    values = np.unique(band)
    if values.max() < class_map.n_classes:
        return band.astype(np.uint8)
    foreground = class_map.n_classes - 1
    return np.where(band > 0, foreground, 0).astype(np.uint8)


def find_pairs(root=DATA_ROOT):
    """Pair images with masks.

    Supports two layouts: a metadata.csv of relative paths, or a pre-split
    train/train_labels directory tree.
    """
    root = Path(root)
    metadata = root / "metadata.csv"
    if metadata.exists():
        table = pd.read_csv(metadata)
        pairs = {
            split: [
                (root / str(row.png_image_path), root / str(row.png_label_path))
                for _, row in part.iterrows()
            ]
            for split, part in table.groupby("split")
        }
        return pairs.get("train", []), pairs.get("val", []), pairs.get("test", [])

    splits = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mask_dir = Path(str(split_dir) + "_labels")
        if not mask_dir.is_dir():
            continue
        masks = {p.stem: p for p in mask_dir.iterdir()}
        pairs = [
            (img, masks[img.stem])
            for img in sorted(split_dir.iterdir())
            if img.suffix.lower() in {".png", ".jpg", ".jpeg"} and img.stem in masks
        ]
        name = split_dir.name
        name = "train" if "train" in name else ("val" if "val" in name else "test")
        splits[name] = pairs

    train = splits.get("train", [])
    val = splits.get("val", [])
    test = splits.get("test", [])
    if not val or not test:
        all_pairs = train + val + test
        rng = np.random.default_rng(0)
        rng.shuffle(all_pairs)
        n_train, n_val = round(0.75 * len(all_pairs)), round(0.12 * len(all_pairs))
        train, val, test = all_pairs[:n_train], all_pairs[n_train:n_train + n_val], all_pairs[n_train + n_val:]
    return train, val, test
