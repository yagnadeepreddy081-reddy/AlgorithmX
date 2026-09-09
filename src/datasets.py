"""
Dataset interface for SAMUDRA NETRA slick segmentation.

Two backends behind one interface:

  source="sim"  physics-based Sentinel-1 synthesiser (src/sar_sim.py) -- used
                for the prototype because the labelled benchmark is behind a
                request form
  source="krestenitis"
                the Krestenitis et al. (2019) Sentinel-1 oil-spill benchmark,
                1112 annotated scenes, identical 5-class label ordering.
                Point DATA_ROOT at the extracted archive and switch the flag;
                nothing in unet.py, train.py or the inference pipeline changes.

Keeping the two behind one interface is deliberate: it is what makes the
prototype's training run reproducible against real data without a rewrite.
"""

import os
import numpy as np

CLASS_NAMES = ["sea", "oil spill", "look-alike", "ship", "land"]
DATA_ROOT = os.environ.get("SAMUDRA_DATA_ROOT", "data/krestenitis")


def _load_sim(n, seed0, size):
    import sar_sim

    return sar_sim.generate_batch(n, seed0=seed0, size=size)


def _load_krestenitis(split, size):
    """
    Expected layout after extracting the benchmark archive:
        DATA_ROOT/<split>/images/*.jpg
        DATA_ROOT/<split>/labels_1D/*.png     (pixel value == class index)
    """
    from PIL import Image

    img_dir = os.path.join(DATA_ROOT, split, "images")
    lbl_dir = os.path.join(DATA_ROOT, split, "labels_1D")
    names = sorted(os.listdir(img_dir))
    imgs = np.zeros((len(names), size, size), dtype=np.float32)
    masks = np.zeros((len(names), size, size), dtype=np.uint8)
    for i, nm in enumerate(names):
        im = Image.open(os.path.join(img_dir, nm)).convert("L").resize((size, size), Image.BILINEAR)
        lb = Image.open(os.path.join(lbl_dir, os.path.splitext(nm)[0] + ".png")).resize(
            (size, size), Image.NEAREST
        )
        imgs[i] = np.asarray(im, dtype=np.float32) / 255.0
        masks[i] = np.asarray(lb, dtype=np.uint8)
    return imgs, masks, [{} for _ in names]


def load_split(split, source="sim", size=256, n=None):
    """split in {'train','val','test'}.  Returns (images, masks, metas)."""
    if source == "krestenitis":
        return _load_krestenitis(split, size)
    counts = {"train": 1200, "val": 200, "test": 300}
    seeds = {"train": 100_000, "val": 500_000, "test": 900_000}
    return _load_sim(n or counts[split], seeds[split], size)


def augment(img, mask, rng):
    """Flips and 90 deg rotations only.  SAR slick appearance is rotation
    invariant, but brightness/contrast jitter is NOT applied -- absolute
    backscatter level carries the wind information the model needs."""
    k = rng.integers(0, 4)
    if k:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)
