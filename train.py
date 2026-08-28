"""Train a U-Net on building footprints, or fine-tune one on a new dataset."""

import argparse
import json
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from data import DATA_ROOT, IGNORE_INDEX, ClassMap, decode_mask, find_pairs, read_rgb


#: ImageNet channel statistics - what the pretrained ResNet-34 encoder expects.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def train_augmentation(crop=256):
    """Random crop plus a random symmetry: a building is a building rotated."""
    return A.Compose([
        A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=0),
        A.RandomCrop(height=crop, width=crop),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=0.75),
    ])


class BuildingsDataset(Dataset):
    """Image/mask pairs, normalised for the ImageNet-pretrained encoder.

    `image_transform` runs on the raw uint8 tile before anything else, which is
    where colour alignment goes: the model must see the corrected radiometry,
    not a correction applied after normalisation.
    """

    def __init__(self, pairs, class_map, augmentation=None, image_transform=None):
        self.pairs = pairs
        self.class_map = class_map
        self.augmentation = augmentation
        self.image_transform = image_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image_path, mask_path = self.pairs[i]
        image = read_rgb(image_path)
        mask = decode_mask(mask_path, self.class_map)
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.augmentation is not None:
            out = self.augmentation(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD

        # A colour the palette does not know decodes to IGNORE_INDEX, which is no
        # class at all: its target row stays all-zero rather than indexing off the
        # end of the identity matrix. `Confusion` drops the same pixels when scoring.
        known = mask != IGNORE_INDEX
        one_hot = np.zeros((*mask.shape, self.class_map.n_classes), dtype=np.float32)
        one_hot[known] = np.eye(self.class_map.n_classes, dtype=np.float32)[mask[known]]
        return image.transpose(2, 0, 1).copy(), one_hot.transpose(2, 0, 1).copy()


class Confusion:
    """Truth-vs-prediction pixel counts, accumulated over a whole split."""

    def __init__(self, n_classes):
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, truth, predicted):
        """Add one batch, ignoring pixels whose class is unknown."""
        truth = np.asarray(truth).ravel()
        predicted = np.asarray(predicted).ravel()
        valid = (truth != 255) & (predicted != 255)
        np.add.at(self.matrix, (truth[valid], predicted[valid]), 1)

    def summary(self, names):
        """IoU per class plus the building precision and recall."""
        tp = np.diag(self.matrix).astype(float)
        predicted, actual = self.matrix.sum(0), self.matrix.sum(1)
        union = predicted + actual - tp

        def ratio(numerator, denominator):
            """Safe division: a class absent from both truth and prediction scores 0."""
            return np.divide(numerator, denominator,
                             out=np.zeros_like(numerator), where=denominator > 0)

        precision, recall, iou = ratio(tp, predicted), ratio(tp, actual), ratio(tp, union)
        return {
            "mean_iou": float(iou.mean()),
            **{f"{name}_iou": float(v) for name, v in zip(names, iou)},
            "building_precision": float(precision[-1]),
            "building_recall": float(recall[-1]),
        }


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`; trains when an optimizer is given, scores either way."""
    training = optimizer is not None
    model.train(training)
    confusion = Confusion(2)
    loss_sum, batches = 0.0, 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = criterion(out, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
            confusion.update(
                masks.argmax(1).cpu().numpy().ravel(),
                out.argmax(1).cpu().numpy().ravel(),
            )
    scores = confusion.summary(["background", "building"])
    scores["loss"] = loss_sum / max(batches, 1)
    return scores


def main():
    """Train, or fine-tune, and keep the checkpoint with the best validation IoU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start from this checkpoint instead of ImageNet (fine-tuning)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="use only this many training tiles (the few-shot curve)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    class_map = ClassMap.from_csv(args.data_root / "label_class_dict.csv")
    train_pairs, valid_pairs, _ = find_pairs(args.data_root)
    if args.train_limit:
        # Nested draws: N=8 contains N=4, so the curve measures what each extra
        # tile buys rather than the variance between unrelated draws.
        order = np.random.default_rng(args.seed).permutation(len(train_pairs))
        train_pairs = [train_pairs[i] for i in order[:args.train_limit]]
    print(f"train {len(train_pairs)} tiles, valid {len(valid_pairs)} tiles")

    train_set = BuildingsDataset(train_pairs, class_map, train_augmentation(args.crop))
    valid_set = BuildingsDataset(valid_pairs, class_map)
    batch_size = min(args.batch_size, max(len(train_set) // 2, 1))
    if batch_size != args.batch_size:
        print(f"batch size {args.batch_size} -> {batch_size}: only {len(train_set)} training tiles")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, drop_last=len(train_set) > batch_size)
    valid_loader = DataLoader(valid_set, batch_size=1, num_workers=4)

    if args.init_from:
        model = torch.load(args.init_from, map_location=device, weights_only=False)
        print(f"fine-tuning from {args.init_from}")
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         classes=2, activation="sigmoid")
    model = model.to(device)
    criterion = smp.losses.DiceLoss(mode="multilabel", from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_iou = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_scores = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_scores = run_epoch(model, valid_loader, criterion, device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_scores.items()},
                        **{f"valid_{k}": v for k, v in valid_scores.items()}})
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch:3d}  train mIoU {train_scores['mean_iou']:.4f}  "
              f"valid mIoU {valid_scores['mean_iou']:.4f}")

        if valid_scores["mean_iou"] > best_iou:
            best_iou = valid_scores["mean_iou"]
            torch.save(model, args.out / "best_model.pth")


if __name__ == "__main__":
    main()
