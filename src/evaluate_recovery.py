"""
End-to-end recovery evaluation.

Segmentation IoU says whether the network can find oil.  It says nothing about
whether the *system* can name the ship, which is the actual claim.  This script
measures that claim directly.

For N independently seeded incidents it runs the whole chain -- forward release
simulation, SAR rendering, segmentation, characterisation, backward drift
ensemble, AIS correlation, scoring -- and reports:

    detection rate      fraction of injected discharges the segmenter found
    rank-1 accuracy     fraction where the true vessel was ranked first
    clearance rate      fraction of innocent vessels correctly cleared
    origin error        distance between the true release point and the
                        hindcast envelope centre at the estimated release hour
    false-alarm rate    fraction of no-oil scenes that produced an incident

The default traffic has constructed competitors; this is synthetic evaluation.
Discharge parameters are not selected by outcome.  Seeds are sequential, the release parameters are drawn at
random inside operationally plausible ranges, and every run is counted.

    python3 src/evaluate_recovery.py --n 12
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from experiments import summarise_comparisons
import geo
import pipeline as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-clean", type=int, default=8)
    ap.add_argument("--model", default="models/unet_slick.pt")
    ap.add_argument("--out", default="out/recovery.json")
    ap.add_argument("--traffic", choices=["constructed", "random"], default="constructed")
    ap.add_argument("--seed-base", type=int, default=770000)
    ap.add_argument("--parameter-seed", type=int, default=20260902)
    ap.add_argument("--results", default="results/research-comparison.json")
    args = ap.parse_args()
    if args.n < 1 or args.n_clean < 1:
        ap.error("Both trial counts must be positive")

    model = pl.load_model(args.model)
    rng = np.random.default_rng(args.parameter_seed)
    t0 = time.time()

    detected = 0
    rank1 = 0
    cleared_ok = 0
    cleared_tot = 0
    errors = []
    trials = []

    for i in range(args.n):
        seed = args.seed_base + i * 37
        rel = {
            "start_h": float(rng.uniform(6.0, 10.0)),
            "speed_kn": float(rng.uniform(8.5, 13.5)),
            "course_deg": float(rng.uniform(0, 360)),
            "wind_ms": float(rng.uniform(4.5, 10.5)),
        }
        rel["end_h"] = rel["start_h"] - float(rng.uniform(1.2, 2.6))
        inc = pl.build_incident(model, seed, "EVAL-%03d" % i, "2026-09-01T05:42:18Z",
                                want_oil=True, release=rel, traffic_mode=args.traffic)
        t = inc["release_truth"]
        dets = [d for d in inc["detections"] if d.get("candidates")]
        rec = {"seed": seed, "course": round(rel["course_deg"], 1),
               "wind": round(rel["wind_ms"], 1), "detected": bool(dets)}
        if dets:
            detected += 1
            d = dets[0]
            cands = d["candidates"]
            top = cands[0]
            hit = bool(top["status"] == "CANDIDATE" and (top.get("truth") or "").startswith("guilty"))
            rank1 += 1 if hit else 0
            rec["rank1"] = hit
            rec["top"] = top["name"]
            rec["score"] = top["score_pct"]
            for c in cands:
                if not c.get("truth", "").startswith("guilty"):
                    cleared_tot += 1
                    cleared_ok += 1 if c["status"] == "EXCLUDED" else 0
            # one of the innocents is a deliberate hard case: it crossed the
            # release area during the release window, went dark while it was
            # there, and is a tanker.  It is meant to survive as a candidate --
            # what must not happen is it outranking the vessel that did it.
            rec["innocent_above_guilty"] = sum(
                1 for c in cands
                if not c.get("truth", "").startswith("guilty") and c["score"] > (
                    [x["score"] for x in cands if x.get("truth", "").startswith("guilty")] or [1]
                )[0]
            )
            usable = [f for f in d["drift"]["frames"] if f["usable"]]
            h = int(round(d.get("release_hour_estimate") or 0))
            if usable and d.get("release_hour_estimate") is not None:
                best = min(usable, key=lambda f: abs(f["hours_back"] - h))
                err = geo.haversine_km(best["lat"], best["lon"],
                                       t["true_release_lat"], t["true_release_lon"])
                errors.append(err)
                rec["origin_error_km"] = round(err, 2)
        guilty = next((c["mmsi"] for d in inc["detections"] for c in d.get("candidates", [])
                       if (c.get("truth") or "").startswith("guilty")), "419087341")
        methods = dets[0].get("method_comparisons", {}) if dets else {}
        rec["methods"] = {name: dict(result, correct=result.get("mmsi") == guilty)
                          for name, result in methods.items()}
        trials.append(rec)
        print("  %-10s detected=%-5s rank1=%-5s err=%s" % (
            rec["seed"], rec["detected"], rec.get("rank1", "-"),
            rec.get("origin_error_km", "-")), flush=True)

    false_alarms = 0
    for i in range(args.n_clean):
        seed = 660000 + i * 53
        inc = pl.build_incident(model, seed, "CLEAN-%03d" % i, "2026-09-02T05:44:51Z",
                                want_oil=False, release=None)
        strong = [d for d in inc["detections"] if d["confidence"] >= 0.35 and d["area_km2"] >= 1.5]
        if strong:
            false_alarms += 1
        print("  clean %d: %d oil regions, %d above escalation, %d look-alikes rejected"
              % (i, inc["n_detections"], len(strong), len(inc["lookalikes"])), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = {
        "protocol": "observation_boundary_v2",
        "data_source": "synthetic",
        "traffic_mode": args.traffic, "seed_base": args.seed_base,
        "parameter_seed": args.parameter_seed,
        "comparisons": summarise_comparisons(trials),
        "incidents": args.n,
        "clean_scenes": args.n_clean,
        "detection_rate": round(detected / max(args.n, 1), 3),
        "rank1_accuracy": round(rank1 / max(detected, 1), 3),
        "rank1_over_all": round(rank1 / max(args.n, 1), 3),
        "counts": {"discharges": args.n, "detected": detected, "rank1": rank1,
                   "false_alarms": false_alarms, "clean": args.n_clean},
        "innocent_clearance_rate": round(cleared_ok / max(cleared_tot, 1), 3),
        "innocent_above_guilty": sum(t.get("innocent_above_guilty", 0) for t in trials),
        "median_origin_error_km": round(float(np.median(errors)), 2) if errors else None,
        "p90_origin_error_km": round(float(np.percentile(errors, 90)), 2) if errors else None,
        "false_alarm_rate": round(false_alarms / max(args.n_clean, 1), 3),
        "trials": trials,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(report, f, indent=2)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== END-TO-END RECOVERY (%d incidents, %d clean scenes) ===" % (args.n, args.n_clean))
    print("  discharge detected            %.0f%%" % (report["detection_rate"] * 100))
    print("  true vessel ranked first      %.0f%% of detections (%.0f%% of all incidents)"
          % (report["rank1_accuracy"] * 100, report["rank1_over_all"] * 100))
    print("  innocent vessels excluded     %.0f%%" % (report["innocent_clearance_rate"] * 100))
    print("  release-origin error          median %s km, p90 %s km"
          % (report["median_origin_error_km"], report["p90_origin_error_km"]))
    print("  false alarms on clean scenes  %.0f%%" % (report["false_alarm_rate"] * 100))
    print("  wall %.0f s" % report["wall_seconds"])


if __name__ == "__main__":
    main()
