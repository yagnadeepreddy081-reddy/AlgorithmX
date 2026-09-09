"""
Rebuild everything from source, in order.

Running this on a fresh clone reproduces `public/index.html` exactly as
shipped -- the segmentation results, the 38-scene test set, the sample tiles,
the browser weights and the console itself.

Training is skipped by default because a trained checkpoint is committed at
models/unet_slick.pt; pass --train to redo it from scratch (about 80 minutes on
a CPU, no GPU needed or used).

    python3 src/build_all.py
    python3 src/build_all.py --train
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(label, args):
    print("\n" + "=" * 68)
    print(">> " + label)
    print("=" * 68, flush=True)
    t = time.time()
    r = subprocess.run([sys.executable] + args, cwd=ROOT)
    if r.returncode != 0:
        print("\nFAILED at: %s" % label)
        sys.exit(r.returncode)
    print("   (%.0f s)" % (time.time() - t), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="retrain the network instead of using the committed checkpoint")
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    t0 = time.time()

    if args.train:
        run("1/6  train the segmentation network",
            ["src/train.py", "--epochs", str(args.epochs)])
    else:
        ck = os.path.join(ROOT, "models", "unet_slick.pt")
        if not os.path.exists(ck):
            print("No checkpoint at models/unet_slick.pt -- rerun with --train")
            sys.exit(1)
        print("1/6  using the committed checkpoint (pass --train to retrain)")

    run("2/6  run three incidents end to end", ["src/pipeline.py"])
    run("3/6  build the 38-scene test set", ["src/build_library.py", "--n", "30", "--n-clean", "8"])
    run("4/6  build the native-scale sample tiles", ["src/build_samples.py"])
    run("5/6  export float16 weights for the browser", ["src/export_weights.py"])
    run("6/6  bake the console into one HTML file", ["src/build_console.py"])

    src = os.path.join(ROOT, "web", "samudra_console.html")
    dst = os.path.join(ROOT, "public", "index.html")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
        print("\ncopied -> public/index.html  (%.2f MB)" % (os.path.getsize(dst) / 1e6))

    print("\nAll done in %.0f s. Open public/index.html in any browser." % (time.time() - t0))


if __name__ == "__main__":
    main()
