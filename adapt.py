"""Label-free domain adaptation between two aerial datasets.

A model trained on Massachusetts loses building IoU on Polish orthophotos: the
cameras, the light and the roofs are all different. Two cheap corrections need
no Polish labels at all:

* **colour alignment** - rescale the target imagery so its radiometry matches
  the statistics of the *training* set. The reference statistics come from the
  training split, never from the tiles being scored, so nothing here peeks.
* **test-time augmentation** - average the prediction over the eight symmetries
  of the square. An aerial tile is valid in any orientation, so disagreement
  between the eight views is model noise worth averaging away.

Run this file to measure a dataset's statistics:

    python adapt.py data/massachusetts-buildings --split train -o stats/massachusetts.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = 256
COLOUR_METHODS = ("none", "moments", "histogram")


class DomainStats:
    """Radiometry of a dataset: per-channel mean, spread and 256-level histogram."""

    def __init__(self, mean, std, histogram, n_pixels, name=""):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.histogram = np.asarray(histogram, dtype=np.float64)
        self.n_pixels = int(n_pixels)
        self.name = name

    @property
    def cdf(self):
        """Per-channel cumulative distribution over the 256 levels."""
        total = self.histogram.sum(axis=1, keepdims=True)
        return np.cumsum(self.histogram / np.where(total > 0, total, 1.0), axis=1)

    @classmethod
    def from_images(cls, images, name="", exclude=()):
        """Accumulate over (H, W, C) uint8 arrays in one pass.

        `exclude` drops whole colours before counting. It exists because 8.5% of
        the Massachusetts training pixels are no-data padding: leaving that black
        in the reference costs a third of what histogram matching is worth.
        """
        total = squares = histogram = None
        count = 0
        excluded = [np.asarray(c, dtype=np.float64) for c in exclude]

        for image in images:
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"expected (H, W, C) images, got shape {array.shape}")
            flat = array.reshape(-1, array.shape[-1]).astype(np.float64)
            for colour in excluded:
                flat = flat[~np.all(flat == colour, axis=1)]
            if not len(flat):
                continue
            if total is None:
                channels = flat.shape[1]
                total, squares = np.zeros(channels), np.zeros(channels)
                histogram = np.zeros((channels, LEVELS))
            total += flat.sum(axis=0)
            squares += (flat**2).sum(axis=0)
            count += len(flat)
            levels = np.clip(np.rint(flat), 0, LEVELS - 1).astype(np.int64)
            for c in range(levels.shape[1]):
                histogram[c] += np.bincount(levels[:, c], minlength=LEVELS)

        if not count:
            raise ValueError("no images to accumulate statistics over")
        mean = total / count
        variance = np.clip(squares / count - mean**2, 0.0, None)
        return cls(mean, np.sqrt(variance), histogram, count, name)

    def to_json(self, path):
        """Write these statistics where predict.py can read them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "name": self.name,
            "n_pixels": self.n_pixels,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "histogram": self.histogram.astype(np.int64).tolist(),
        }))
        return path

    @classmethod
    def from_json(cls, path):
        """Read back what to_json wrote."""
        raw = json.loads(Path(path).read_text())
        return cls(raw["mean"], raw["std"], raw["histogram"],
                   raw["n_pixels"], raw.get("name", ""))


def _to_uint8(array):
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def match_moments(image, reference, source=None):
    """Shift and scale each channel onto the reference mean and spread.

    With `source` statistics the same correction is applied to every tile, which
    is what you want when the target set really is one camera. Without them each
    tile is normalised on its own, which also flattens genuine scene-to-scene
    variation.
    """
    array = np.asarray(image, dtype=np.float64)
    flat = array.reshape(-1, array.shape[-1])
    source_mean = source.mean if source is not None else flat.mean(axis=0)
    source_std = source.std if source is not None else flat.std(axis=0)
    usable = source_std > 1e-6                      # a flat channel cannot be rescaled
    scale = np.where(usable, reference.std / np.where(usable, source_std, 1.0), 1.0)
    return _to_uint8((array - source_mean) * scale + reference.mean)


def match_histogram(image, reference):
    """Remap each channel so its cumulative distribution follows the reference's.

    Non-parametric, so it corrects a whole tone curve rather than just a mean and
    a width - at the cost of following any oddity in the reference exactly.
    """
    array = np.asarray(image)
    levels = np.clip(np.rint(array), 0, LEVELS - 1).astype(np.int64)
    reference_cdf = reference.cdf
    out = np.empty_like(levels)
    for c in range(array.shape[-1]):
        counts = np.bincount(levels[..., c].ravel(), minlength=LEVELS).astype(np.float64)
        source_cdf = np.cumsum(counts / max(counts.sum(), 1.0))
        lookup = np.clip(np.searchsorted(reference_cdf[c], source_cdf), 0, LEVELS - 1)
        out[..., c] = lookup[levels[..., c]]
    return out.astype(np.uint8)


def build_colour_adapter(method, reference=None, source=None):
    """Resolve a method name to a function on uint8 (H, W, 3) images, or None."""
    if method in (None, "", "none"):
        return None
    if method not in COLOUR_METHODS:
        raise ValueError(f"unknown colour method {method!r}; try one of {list(COLOUR_METHODS)}")
    if reference is None:
        raise ValueError(f"colour method {method!r} needs reference statistics")
    return {
        "moments": lambda image: match_moments(image, reference, source),
        "histogram": lambda image: match_histogram(image, reference),
    }[method]


#: The eight symmetries of the square: (quarter turns, mirror first).
DIHEDRAL = tuple((turns, flip) for flip in (False, True) for turns in range(4))


def tta_forward(forward, batch):
    """Average `forward` over the eight dihedral views, each mapped back."""
    import torch

    total = None
    for turns, flip in DIHEDRAL:
        view = torch.flip(batch, dims=[-1]) if flip else batch
        scores = forward(torch.rot90(view, turns, dims=(-2, -1)))
        scores = torch.rot90(scores, -turns, dims=(-2, -1))
        if flip:
            scores = torch.flip(scores, dims=[-1])
        total = scores if total is None else total + scores
    return total / len(DIHEDRAL)


def main():
    """Measure one split's radiometry and write it as JSON for predict.py."""
    from data import DATA_ROOT, find_pairs, read_rgb

    parser = argparse.ArgumentParser(description="Dataset radiometry statistics.")
    parser.add_argument("data_root", type=Path, nargs="?", default=DATA_ROOT)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N tiles")
    parser.add_argument("--exclude-colour", nargs=3, type=int, action="append", default=None,
                        metavar=("R", "G", "B"), help="drop this colour, e.g. no-data padding")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    pairs = dict(zip(("train", "val", "test"), find_pairs(args.data_root)))[args.split]
    if not pairs:
        raise SystemExit(f"no {args.split} tiles under {args.data_root}")
    if args.limit:
        pairs = pairs[:args.limit]

    stats = DomainStats.from_images(
        (read_rgb(image) for image, _ in pairs),
        name=f"{Path(args.data_root).name}:{args.split}",
        exclude=args.exclude_colour or (),
    )
    stats.to_json(args.out)
    print(f"{len(pairs)} tiles, {stats.n_pixels:,} pixels counted")
    print(f"mean {np.round(stats.mean, 1).tolist()}  std {np.round(stats.std, 1).tolist()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
