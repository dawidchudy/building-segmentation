"""Score a trained model on a split, optionally adapted to a new dataset.

Without flags this is plain inference. With `--colour` and `--tta` it is the
label-free adaptation arm: see `adapt.py` for what each correction does, and the
README for what each one bought on Polish orthophotos.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adapt import DomainStats, build_colour_adapter, tta_forward
from data import DATA_ROOT, ClassMap, decode_mask, find_pairs, read_rgb
from train import BuildingsDataset


def predict_tile(model, dataset, index, device, tta=False):
    """Class scores for one tile, as an (H, W, classes) array."""
    image, _ = dataset[index]
    batch = torch.from_numpy(image[None]).to(device)
    with torch.no_grad():
        scores = tta_forward(model, batch) if tta else model(batch)
    return scores.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def save_panel(path, raw_image, truth_rgb, pred_rgb):
    """Input, ground truth and prediction side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, panel, title in zip(
        axes, (raw_image, truth_rgb, pred_rgb), ("input", "ground truth", "prediction")
    ):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def colour_adapter(args):
    """The image correction asked for on the command line, or None."""
    if args.colour == "none":
        return None
    if not args.reference_stats:
        raise SystemExit(
            f"--colour {args.colour} needs --reference-stats: the radiometry of the "
            f"training set, from `python adapt.py <training-data> --split train -o ...`"
        )
    return build_colour_adapter(
        args.colour,
        reference=DomainStats.from_json(args.reference_stats),
        source=DomainStats.from_json(args.dataset_stats) if args.dataset_stats else None,
    )


def main():
    """Score one split and write panels, with whatever adaptation was requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/default/best_model.pth"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N tiles")
    parser.add_argument("--colour", default="none",
                        choices=("none", "moments", "histogram"),
                        help="align the imagery to --reference-stats before predicting")
    parser.add_argument("--reference-stats", type=Path, default=None,
                        help="statistics of the training set (the target of the alignment)")
    parser.add_argument("--dataset-stats", type=Path, default=None,
                        help="statistics of the data being scored; without it, per tile")
    parser.add_argument("--tta", action="store_true",
                        help="average over the eight symmetries of the tile")
    parser.add_argument("--no-panels", action="store_true", help="metrics only, no figures")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()

    class_map = ClassMap.from_csv(args.data_root / "label_class_dict.csv")
    pairs = dict(zip(("train", "val", "test"), find_pairs(args.data_root)))[args.split]
    if not pairs:
        raise SystemExit(f"no {args.split} tiles found under {args.data_root}")
    if args.limit:
        pairs = pairs[:args.limit]

    dataset = BuildingsDataset(pairs, class_map, image_transform=colour_adapter(args))
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_panels:
        (args.out / "panels").mkdir(exist_ok=True)

    tp = fp = fn = 0
    building_index = class_map.index_of("building")
    for i, pair in enumerate(pairs):
        scores = predict_tile(model, dataset, i, device, tta=args.tta)
        predicted = scores.argmax(-1)
        truth = decode_mask(pair[1], class_map)

        valid = truth != 255
        t, p = truth[valid], predicted[valid]
        tp += int(((p == building_index) & (t == building_index)).sum())
        fp += int(((p == building_index) & (t != building_index)).sum())
        fn += int(((p != building_index) & (t == building_index)).sum())

        if not args.no_panels:
            save_panel(args.out / "panels" / f"{pair[0].stem}.png",
                       read_rgb(pair[0]), class_map.to_rgb(truth), class_map.to_rgb(predicted))
        print(f"[{i + 1}/{len(pairs)}] {pair[0].stem}")

    metrics = {
        "split": args.split,
        "tiles": len(pairs),
        "colour": args.colour,
        "tta": bool(args.tta),
        "building_iou": tp / max(tp + fp + fn, 1),
        "building_precision": tp / max(tp + fp, 1),
        "building_recall": tp / max(tp + fn, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nbuilding IoU {metrics['building_iou']:.3f}   "
          f"precision {metrics['building_precision']:.3f}   "
          f"recall {metrics['building_recall']:.3f}   -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
