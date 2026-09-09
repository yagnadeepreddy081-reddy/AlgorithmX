"""
Train the SAMUDRA NETRA slick segmentation U-Net.

Objective is cross-entropy (class-weighted by inverse square-root frequency)
plus soft multi-class Dice.  Oil is ~1 % of pixels; without both terms the
network converges to predicting sea everywhere and reports 94 % accuracy while
being completely useless.

Reported metrics are per-class IoU / precision / recall on a held-out test
split that the model never sees during training or model selection.

    python3 src/train.py --epochs 10 --source sim
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import datasets
from unet import UNet, dice_loss, confusion, metrics_from_confusion


def evaluate(model, imgs, masks, device, batch=4):
    model.eval()
    cm = torch.zeros(5, 5, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = torch.from_numpy(imgs[i : i + batch]).unsqueeze(1).to(device)
            y = torch.from_numpy(masks[i : i + batch].astype(np.int64))
            pred = model(x).argmax(1).cpu()
            cm += confusion(pred, y)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--source", default="sim")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = "cpu"
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tr_i, tr_m, _ = datasets.load_split("train", args.source, args.size)
    va_i, va_m, _ = datasets.load_split("val", args.source, args.size)
    te_i, te_m, _ = datasets.load_split("test", args.source, args.size)
    print("data: train %d val %d test %d  (%.1fs)" % (len(tr_i), len(va_i), len(te_i), time.time() - t0), flush=True)

    freq = np.bincount(tr_m.ravel(), minlength=5).astype(np.float64)
    freq = freq / freq.sum()
    print("train class pixel frequency:", np.round(freq, 5), flush=True)
    # inverse square-root frequency weighting: strong enough to rescue the rare
    # classes, gentle enough not to make the model trigger-happy on oil
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = w / w.mean()
    weights = torch.tensor(w, dtype=torch.float32)
    print("class weights:", np.round(w, 3), flush=True)

    model = UNet(1, 5, base=args.base).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr_i) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.25)

    rng = np.random.default_rng(1)
    history = []
    best = -1.0
    step = 0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_i))
        run, nb = 0.0, 0
        ep_t = time.time()
        for bi in range(0, len(order) - args.batch + 1, args.batch):
            idx = order[bi : bi + args.batch]
            xs, ys = [], []
            for j in idx:
                a, b = datasets.augment(tr_i[j], tr_m[j], rng)
                xs.append(a)
                ys.append(b)
            x = torch.from_numpy(np.stack(xs)).unsqueeze(1)
            y = torch.from_numpy(np.stack(ys).astype(np.int64))
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y, weight=weights) + dice_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            run += float(loss)
            nb += 1

        cm = evaluate(model, va_i, va_m, device, args.batch)
        iou, prec, rec = metrics_from_confusion(cm)
        # model selection on the oil class, because that is the class the whole
        # downstream attribution chain depends on
        score = iou[1]
        history.append({"epoch": ep + 1, "loss": run / max(nb, 1), "val_iou": iou, "val_oil_iou": score})
        print(
            "epoch %2d  loss %.4f  val IoU oil %.3f look-alike %.3f ship %.3f land %.3f sea %.3f  (%.0fs)"
            % (ep + 1, run / max(nb, 1), iou[1], iou[2], iou[3], iou[4], iou[0], time.time() - ep_t),
            flush=True,
        )
        if score > best:
            best = score
            torch.save(
                {"state_dict": model.state_dict(), "base": args.base, "size": args.size},
                os.path.join(args.out, "unet_slick.pt"),
            )

    ck = torch.load(os.path.join(args.out, "unet_slick.pt"), map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    cm = evaluate(model, te_i, te_m, device, args.batch)
    iou, prec, rec = metrics_from_confusion(cm)
    report = {
        "source": args.source,
        "parameters": n_par,
        "epochs": args.epochs,
        "train_scenes": len(tr_i),
        "test_scenes": len(te_i),
        "classes": datasets.CLASS_NAMES,
        "test_iou": iou,
        "test_precision": prec,
        "test_recall": rec,
        "mean_iou": float(np.mean(iou)),
        "confusion": cm.tolist(),
        "history": history,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== HELD-OUT TEST SET (%d scenes, never seen) ===" % len(te_i))
    for i, nm in enumerate(datasets.CLASS_NAMES):
        print("  %-11s IoU %.3f  precision %.3f  recall %.3f" % (nm, iou[i], prec[i], rec[i]))
    print("  mean IoU %.3f   parameters %s   wall %.0f s" % (np.mean(iou), f"{n_par:,}", time.time() - t0))


if __name__ == "__main__":
    main()
