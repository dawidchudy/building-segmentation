# A guided tour of `building-segmentation`

This document explains, in plain language, how the code in this repository
works. It assumes you are **new to programming**: every concept is explained
the first time it appears.

## Before you start: Python concepts you will meet

**Modules and imports.** Every `.py` file is a *module*. Code from other files
is pulled in at the top of the file:

```python
import numpy as np              # whole numpy library, nicknamed "np"
from data import load_catalog   # one function from our own data.py
```

**Functions.** Reusable recipes. They take inputs (*arguments*) and hand back
a result:

```python
def normalise(image):
    """Squeeze pixel values into the range 0..1."""
    low, high = image.min(), image.max()
    return (image - low) / (high - low)
```

The triple-quoted string is a *docstring* - a note saying what the function
does. Every function in these projects has one; when opening an unfamiliar
file, read the docstrings first and you already understand 80% of it.

**Classes.** A way to bundle data together with the functions that operate on
it. Two examples you will meet:

- `ClassMap` (building-segmentation): stores class names and their colours,
  and knows how to translate between "colour picture" and "number picture".
- `MergerCNN` (galaxy-merger-classification): a neural network. Its two special
  functions are `__init__` ("build me") and `forward` ("run this input through").

**Command-line options (`argparse`).** Scripts accept settings without editing
code:

```bash
python train.py --epochs 50 --representation raw
```

Look for `parser.add_argument(...)` lines near the bottom of each script -
that is just a menu of options with sensible defaults.

**`if __name__ == "__main__":`.** Found at the bottom of most files. It means
"run this only when the file itself is executed" - so the same file works both
as a library others import from, and as a program you can run directly.

### The libraries used everywhere

| library | what it is |
|---|---|
| **NumPy** (`numpy`) | fast arrays of numbers. An **image is a NumPy array**: a grid of pixel values. `image[10:20, 30:40]` grabs a rectangle of pixels. |
| **pandas** (`pandas`) | tables with named columns, like a programmable spreadsheet. Galaxy catalogues live in these ("DataFrames"). |
| **PyTorch** (`torch`) | builds and trains neural networks. Wraps arrays in *tensors* that GPUs can process. |
| **matplotlib** | draws figures. **astropy** reads astronomy files. **PIL** reads normal images. **scikit-learn** provides ready-made quality metrics. |

### Words from machine learning

- **model / network** - a mathematical function with millions of adjustable
  numbers (*weights*). Adjusting those weights is *training*.
- **training loop** - show examples → measure how wrong the guesses are
  (the *loss*) → nudge weights to be less wrong → repeat thousands of times.
- **epoch** - one full pass through all training images.
- **batch** - images processed together in one go (e.g. 256).
- **train/validation/test split** - three disjoint piles: learn from the first,
  tune choices on the second, touch the third only once for the honest final
  score. Keeping data aside prevents *overfitting* (memorising instead of
  learning).


## How the code works

*Files: `data.py`, `train.py`, `predict.py`,
`adapt.py`, notebook*

### The problem

Project 2 answers "what is this image?"; segmentation answers "**which pixels**
are a building?" - the model outputs a picture, not a label. Trained on the
Massachusetts Buildings dataset (US aerial tiles), then tested *unchanged* on
Polish orthophotos of Wieliczka. Performance drops (different cameras, colours,
roof styles); measuring that gap and trying cheap ways to close it is the story.

### data.py - masks are just coloured pictures

Ground truth ("where buildings really are") is stored as an image where each
class has its own colour; `label_class_dict.csv` declares them:

```
name,r,g,b
background,0,0,0
building,255,0,0
```

(Massachusetts codes buildings **red**, not white - a classic trap that checking
this file prevents.) `ClassMap.to_index` converts colour-pictures to
number-pictures:

```python
keys = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
out  = np.full(keys.shape, IGNORE_INDEX, dtype=np.uint8)     # 255 = unknown
for i, (r, g, b) in enumerate(self.palette):
    out[keys == ((r << 16) | (g << 8) | b)] = i              # colour -> class number
```

The `<<` trick packs three 0-255 values into one big integer so each whole
colour can be matched in one comparison. Every pixel ends up as 0, 1 or 255.
`to_rgb` runs backwards for drawing predictions.

`find_pairs` matches image tiles to mask tiles (two folder layouts supported)
and falls back to its own random 75/12/13 split when no split is given.

### train.py - U-Net, augmentation and Dice loss

The model comes from a library rather than being hand-built:

```python
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                 classes=2, activation="sigmoid")
```

A **U-Net** squeezes the tile down through the ResNet-34 encoder and expands it
back up, outputting a score-map per class at full resolution - one painted layer
per class. Two ideas make it work well:

- **Transfer learning**: the encoder already knows how to see, from being
  pretrained on millions of ImageNet photos (`encoder_weights="imagenet"`).
  We reuse those skills instead of starting blind.
- **Augmentation** (`train_augmentation`): every tile is randomly cropped to
  256x256 and randomly flipped/rotated before being shown - free training
  variety, since aerial photos look valid in any orientation.

The loss is **Dice loss**, which directly rewards overlap between predicted and
true building regions. Plain accuracy would be misleading here: buildings cover
only a few percent of pixels, so a lazy model shouting "nothing is a building"
would score 95%.

Quality is measured by **IoU** (intersection over union) - how much the two
painted regions overlap, 1.0 perfect / 0.0 none. The small `Confusion` class
accumulates truth-vs-guess counts across all tiles and epochs, then:

```python
denom = matrix.sum(1) + matrix.sum(0) - tp      # size of the union
iou = tp / denom                                # intersection over union
```

Training saves `best_model.pth` whenever validation IoU improves.

### predict.py - using the trained model

Loads `best_model.pth`, runs it over the split you name, writes side-by-side
panels (*input / ground truth / prediction*) into `outputs/panels/`, and prints
building precision/recall/IoU. This is also the script used for the Wieliczka
cross-dataset experiment: point it at Polish tiles with the same checkpoint and
watch building IoU fall from 0.64 to 0.38.

### adapt.py - making the model travel

Two ways to win that back without any Polish labels. **Colour alignment**
measures the radiometry of the training set once (`DomainStats`: per-channel
mean, spread and a 256-level histogram) and rescales the new imagery onto it,
either by matching mean and spread (`match_moments`) or the whole tone curve
(`match_histogram`). **Test-time augmentation** (`tta_forward`) runs the tile
through the model in all eight rotations and mirrorings and averages the eight
answers back onto the original orientation.

Together they recover +0.143 building IoU. Then the result that matters:
`train.py --init-from checkpoint --train-limit 1`, fine-tuning on a *single*
labelled Polish tile, recovers +0.179 - more than both label-free tricks
combined. Measuring that is worth more than either method.

