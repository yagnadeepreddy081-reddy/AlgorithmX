"""
Export the trained U-Net so it can run inside the browser.

Two things happen here.

1. BatchNorm is folded into the preceding convolution.  Every DoubleConv block
   is Conv -> BN -> ReLU, and at inference time BN is just an affine map with
   fixed statistics, so it can be baked into the convolution weights:

       W' = W * gamma / sqrt(var + eps)          (per output channel)
       b' =    beta - gamma * mean / sqrt(var + eps)

   That removes every BN layer from the JavaScript engine and cuts the number
   of passes over each feature map, which matters when the whole forward pass
   is running in a browser tab in front of a judge.

2. Weights are written as float16.  753,385 parameters at float32 is 3.0 MB,
   which becomes 4.0 MB once base64-encoded into the page.  float16 halves
   that at a cost far below the noise floor of a 0.737 IoU model.

Output: web/weights.json  -- shapes manifest plus one base64 float16 blob.
"""

import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from unet import UNet


def fold(conv, bn):
    """Return (weight, bias) for a Conv2d with its following BatchNorm2d baked in."""
    w = conv.weight.detach().numpy().astype(np.float64)
    gamma = bn.weight.detach().numpy().astype(np.float64)
    beta = bn.bias.detach().numpy().astype(np.float64)
    mean = bn.running_mean.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    eps = float(bn.eps)
    scale = gamma / np.sqrt(var + eps)
    w = w * scale[:, None, None, None]
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


def main(ckpt="models/unet_slick.pt", out="web/weights.json"):
    ck = torch.load(ckpt, map_location="cpu")
    base = ck.get("base", 20)
    model = UNet(1, 5, base=base)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    tensors = []          # (name, ndarray)

    def add_double(prefix, block):
        # block.block = [Conv, BN, ReLU, Conv, BN, ReLU]
        w1, b1 = fold(block.block[0], block.block[1])
        w2, b2 = fold(block.block[3], block.block[4])
        tensors.append((prefix + ".c1.w", w1))
        tensors.append((prefix + ".c1.b", b1))
        tensors.append((prefix + ".c2.w", w2))
        tensors.append((prefix + ".c2.b", b2))

    add_double("enc1", model.enc1)
    add_double("enc2", model.enc2)
    add_double("enc3", model.enc3)
    add_double("bott", model.bott)

    for nm, layer in [("up3", model.up3), ("up2", model.up2), ("up1", model.up1)]:
        tensors.append((nm + ".w", layer.weight.detach().numpy().astype(np.float32)))
        tensors.append((nm + ".b", layer.bias.detach().numpy().astype(np.float32)))

    add_double("dec3", model.dec3)
    add_double("dec2", model.dec2)
    add_double("dec1", model.dec1)

    tensors.append(("head.w", model.head.weight.detach().numpy().astype(np.float32)))
    tensors.append(("head.b", model.head.bias.detach().numpy().astype(np.float32)))

    manifest = []
    flat = []
    offset = 0
    for nm, arr in tensors:
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        manifest.append({"name": nm, "shape": list(arr.shape), "offset": offset, "size": int(a.size)})
        flat.append(a)
        offset += int(a.size)

    allw = np.concatenate(flat).astype(np.float16)
    blob = base64.b64encode(allw.tobytes()).decode()

    payload = {
        "base": base,
        "in_size": ck.get("size", 256),
        "classes": ["sea", "oil spill", "look-alike", "ship", "land"],
        "dtype": "float16",
        "count": int(allw.size),
        "tensors": manifest,
        "data": blob,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print("tensors %d, params %s, float16 %.2f MB, base64 %.2f MB"
          % (len(manifest), f"{offset:,}", allw.nbytes / 1e6, len(blob) / 1e6))

    # ---- reference output for the JavaScript engine to be checked against ----
    import sar_sim
    img, _, _ = sar_sim.generate_scene(123456, size=256)
    with torch.no_grad():
        logits = model(torch.from_numpy(img).unsqueeze(0).unsqueeze(0))
        pred = logits.argmax(1)[0].numpy().astype(np.uint8)
    ref = {
        "input_sum": float(img.sum()),
        "input": [round(float(v), 6) for v in img[::16, ::16].ravel()],
        "logits_corner": [[round(float(logits[0, c, y, x]), 4) for c in range(5)]
                          for (y, x) in [(0, 0), (64, 64), (128, 128), (200, 90)]],
        "class_counts": [int((pred == c).sum()) for c in range(5)],
    }
    with open("out/reference_forward.json", "w") as f:
        json.dump(ref, f, indent=2)
    print("reference class counts:", ref["class_counts"])


if __name__ == "__main__":
    main()
