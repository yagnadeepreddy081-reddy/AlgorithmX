"""
Build the scene library.

The point of this file is to answer one specific objection: "you only showed me
the three scenes that work."

It generates N independent scenes on sequential seeds with the release
parameters -- course, speed, wind, discharge duration -- drawn at random inside
operationally plausible ranges, plus a share of scenes containing no oil at all.
Every scene is kept, including the ones the pipeline gets wrong, and each one
records whether the true vessel was recovered so the console can say so on
screen.

Nothing here is selected, reordered or filtered by outcome.

    python3 src/build_library.py --n 30 --n-clean 8
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import geo
import pipeline as pl

ACQ_TIMES = [
    "2026-08-24T05:41:07Z", "2026-08-26T17:35:44Z", "2026-08-27T05:43:19Z",
    "2026-08-29T17:36:51Z", "2026-08-30T05:42:33Z", "2026-09-01T17:34:58Z",
]


def grade(inc):
    """Record the outcome of this scene, honestly, whatever it is."""
    dets = [d for d in inc["detections"] if d.get("candidates")]
    truth = inc.get("release_truth")
    if truth is None:
        strong = [d for d in inc["detections"]
                  if d["confidence"] >= 0.35 and d["area_km2"] >= 1.5]
        return {
            "kind": "no discharge",
            "outcome": "false alarm" if (strong or dets) else "correct rejection",
            "correct": not (strong or dets),
            "summary": ("Raised an incident on a scene with no oil in it."
                        if (strong or dets) else
                        "No oil-class detection cleared the threshold. No vessel named."),
        }
    if not dets or not any(c["status"] == "CANDIDATE" for c in dets[0]["candidates"]):
        return {
            "kind": "discharge",
            "outcome": "missed",
            "correct": False,
            "summary": "A discharge was injected but the pipeline produced no eligible vessel ranking.",
        }
    top = dets[0]["candidates"][0]
    hit = bool(top["status"] == "CANDIDATE" and (top.get("truth") or "").startswith("guilty"))
    d = dets[0]
    err = None
    usable = [f for f in d["drift"]["frames"] if f["usable"]]
    if usable and d.get("release_hour_estimate") is not None:
        h = int(round(d.get("release_hour_estimate") or 0))
        best = min(usable, key=lambda f: abs(f["hours_back"] - h))
        err = round(geo.haversine_km(best["lat"], best["lon"],
                                     truth["true_release_lat"], truth["true_release_lon"]), 2)
    return {
        "kind": "discharge",
        "outcome": "vessel recovered" if hit else "wrong vessel ranked first",
        "correct": hit,
        "origin_error_km": err,
        "summary": ("Ranked %s first with index %.0f/100; %d of %d other vessels excluded."
                    % (top["name"], top["score_pct"],
                       sum(1 for c in dets[0]["candidates"][1:] if c["status"] == "EXCLUDED"),
                       len(dets[0]["candidates"]) - 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-clean", type=int, default=8)
    ap.add_argument("--model", default="models/unet_slick.pt")
    ap.add_argument("--out", default="out/library.json")
    args = ap.parse_args()

    model = pl.load_model(args.model)
    rng = np.random.default_rng(2026090201)
    t0 = time.time()
    scenes = []

    for i in range(args.n):
        seed = 810000 + i * 41
        rel = {
            "start_h": float(rng.uniform(5.5, 10.5)),
            "speed_kn": float(rng.uniform(8.0, 14.0)),
            "course_deg": float(rng.uniform(0, 360)),
            "wind_ms": float(rng.uniform(3.5, 11.5)),
        }
        rel["end_h"] = max(1.5, rel["start_h"] - float(rng.uniform(1.2, 2.8)))
        inc = pl.build_incident(model, seed, "SCN-%04d" % (i + 1),
                                ACQ_TIMES[i % len(ACQ_TIMES)],
                                want_oil=True, release=rel, compact=True)
        inc["grade"] = grade(inc)
        inc["title"] = "Wind %.1f m/s, course %03d, valve open %.1f h" % (
            rel["wind_ms"], int(rel["course_deg"]), rel["start_h"] - rel["end_h"])
        scenes.append(inc)
        print("  %s  %-26s %s" % (inc["incident_id"], inc["grade"]["outcome"], inc["title"]), flush=True)

    for i in range(args.n_clean):
        seed = 690000 + i * 59
        inc = pl.build_incident(model, seed, "SCN-%04d" % (args.n + i + 1),
                                ACQ_TIMES[i % len(ACQ_TIMES)],
                                want_oil=False, release=None, compact=True)
        inc["grade"] = grade(inc)
        inc["title"] = "No discharge - look-alikes only"
        scenes.append(inc)
        print("  %s  %-26s %s" % (inc["incident_id"], inc["grade"]["outcome"], inc["title"]), flush=True)

    disc = [s for s in scenes if s["grade"]["kind"] == "discharge"]
    clean = [s for s in scenes if s["grade"]["kind"] == "no discharge"]
    errs = [s["grade"]["origin_error_km"] for s in disc if s["grade"].get("origin_error_km") is not None]
    summary = {
        "scenes": len(scenes),
        "with_discharge": len(disc),
        "clean": len(clean),
        "recovered": sum(1 for s in disc if s["grade"]["correct"]),
        "missed": sum(1 for s in disc if s["grade"]["outcome"] == "missed"),
        "wrong_vessel": sum(1 for s in disc if s["grade"]["outcome"] == "wrong vessel ranked first"),
        "false_alarms": sum(1 for s in clean if not s["grade"]["correct"]),
        "median_origin_error_km": round(float(np.median(errs)), 2) if errs else None,
        "protocol": "observation_boundary_v2",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": scenes}, f, separators=(",", ":"))

    print("\n=== LIBRARY ===")
    for k, v in summary.items():
        print("  %-24s %s" % (k, v))
    print("  file %.2f MB, %.0f s" % (os.path.getsize(args.out) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
