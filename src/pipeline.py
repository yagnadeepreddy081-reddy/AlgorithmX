"""
SAMUDRA NETRA -- end-to-end incident pipeline.

    SAR tile -> U-Net segmentation -> characterisation + look-alike filter
             -> backward drift ensemble -> release envelope
             -> AIS correlation -> evidence scoring -> incident dossier (JSON)

Run:
    python3 src/pipeline.py --scenes 3 --out out/incidents.json
"""

import argparse
import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import ais as ais_mod
import characterise as ch
import drift as drift_mod
import fields
import incident_sim
import sar_sim
import score as score_mod
from unet import UNet
from experiments import compare_methods

CLASS_COLOURS = {
    1: (255, 62, 62),     # oil
    2: (255, 208, 61),    # look-alike
    3: (78, 255, 154),    # ship
    4: (150, 122, 84),    # land
}


def load_model(path="models/unet_slick.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_num_threads(2)
    m = UNet(1, 5, base=ck.get("base", 20))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def segment(model, img):
    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].numpy()
    return probs.argmax(0).astype(np.uint8), probs


def _png_b64(arr_rgb, scale=1):
    im = Image.fromarray(arr_rgb)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpg_b64(arr_rgb, quality=74):
    """JPEG for the raw SAR quicklook.

    Speckle is high-entropy noise, so PNG barely compresses it -- a 512 px tile
    costs ~250 kB as PNG and ~45 kB as JPEG.  With three dozen scenes in the
    library that difference is what keeps the page under its size budget.  The
    segmentation overlay stays PNG, because its class colours must not smear.
    """
    im = Image.fromarray(arr_rgb).convert("L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _png_pal_b64(arr_rgb, colours=16):
    """Palette-quantised PNG for the class overlay."""
    im = Image.fromarray(arr_rgb).quantize(colors=colours, method=Image.MEDIANCUT)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def quicklook(img):
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def overlay(img, class_map, alpha=0.55):
    rgb = quicklook(img).astype(np.float32)
    for c, col in CLASS_COLOURS.items():
        m = class_map == c
        if m.any():
            for k in range(3):
                rgb[..., k][m] = rgb[..., k][m] * (1 - alpha) + col[k] * alpha
    return rgb.astype(np.uint8)


def _strip(obj):
    """Remove internal-only keys (leading underscore) before serialisation."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def analyse_observation(model, img, traffic, provider, incident_id, acquired_utc,
                        lat0, lon0, compact=False):
    """Analysis boundary: no masks, release times, truth labels or simulator inputs."""
    if any("truth" in v for v in traffic["vessels"]) or "release_hour_estimate" in traffic:
        raise ValueError("Ground truth is not an analysis input")
    size = img.shape[0]
    if img.shape != (size, size) or size % 8 or not np.isfinite(img).all():
        raise ValueError("Image must be a finite square array with size divisible by 8")
    class_map, probs = segment(model, img)
    dets = ch.characterise(img, class_map, probs, provider.mean_wind,
                           lat0, lon0, acquired_utc)
    prov = provider
    half_km = size * sar_sim.PIXEL_METRES / 2000.0
    import geo as geo_mod
    sw = geo_mod.xy_to_lonlat(-half_km * 1000, -half_km * 1000, lat0, lon0)
    ne = geo_mod.xy_to_lonlat(half_km * 1000, half_km * 1000, lat0, lon0)

    incident = {
        "incident_id": incident_id,
        "acquired_utc": acquired_utc,
        "region": fields.PILOT["name"],
        "centre": [lat0, lon0],
        "bounds": [[round(sw[1], 6), round(sw[0], 6)], [round(ne[1], 6), round(ne[0], 6)]],
        "pixel_metres": sar_sim.PIXEL_METRES,
        "tile_km": round(size * sar_sim.PIXEL_METRES / 1000.0, 1),
        "wind_ms": round(float(provider.mean_wind), 1),
        # Library scenes carry the raw SAR tile only.  The segmentation is sent
        # as polygons instead of a second raster: a speckled overlay image costs
        # 169 kB even palette-quantised, and 38 of those will not fit in a page.
        "sar_png": _jpg_b64(quicklook(img), quality=56) if compact else _png_b64(quicklook(img)),
        "overlay_png": None if compact else _png_b64(overlay(img, class_map)),
        "truth_png": None,
        "n_detections": len(dets),
        "lookalikes": ch.characterise_lookalikes(class_map, lat0, lon0, size),
        "detections": [],
        "field": prov.describe(),
    }

    # Attribution is expensive and only defensible on a well-formed detection,
    # so it runs on the strongest few.  Everything else is characterised and
    # logged -- which is what an operational system does with a marginal patch.
    n_attributed = 0
    for d in dets:
        entry = {k: v for k, v in d.items() if not k.startswith("_")}
        max_attr = 1 if compact else 2
        if (d["confidence"] < 0.35
                or d["area_km2"] < 1.5 or n_attributed >= max_attr):
            entry["drift"] = None
            entry["candidates"] = []
            entry["note"] = "Characterised and logged; below the attribution threshold."
            incident["detections"].append(entry)
            continue
        n_attributed += 1

        # Run further back than the system can defend on purpose: the point at
        # which added uncertainty passes the 12 km threshold is itself a result,
        # and the timeline shows those hours being computed and then discarded.
        dr = drift_mod.backward_drift(d, prov, lat0, lon0, size, hours_back=30,
                                      n_particles=900, n_members=12, seed=0)
        scen = traffic
        cands = score_mod.score_vessels(d, dr, scen, lat0, lon0)

        if compact:
            for fr in dr["frames"]:
                fr["particles"] = [[round(p[0], 4), round(p[1], 4)] for p in fr["particles"][::3]]
                fr.pop("ellipse", None)
        entry["drift"] = _strip(dr)
        entry["release_hour_estimate"] = next((c["release_hour_estimate"] for c in cands
            if c["status"] == "CANDIDATE"), None)
        entry["candidates"] = cands
        entry["method_comparisons"] = compare_methods(d, dr, scen, lat0, lon0)
        incident["detections"].append(entry)

    if compact:
        # keep only what the console renders for a library scene
        incident["detections"] = [
            e for e in incident["detections"] if e.get("candidates") or e["confidence"] >= 0.35
        ][:3]
        for e in incident["detections"]:
            e.pop("factors", None) if False else None
            for c in e.get("candidates", []):
                c.pop("corridor_hits", None)
        # look-alike outlines are kept: with no overlay raster they are the only
        # way the console can show what the discriminator rejected
        incident["lookalikes"] = incident["lookalikes"][:10]

    return _strip(incident)


def build_incident(model, seed, incident_id, acquired_utc, want_oil=True, size=512,
                   release=None, compact=False, traffic_mode="constructed"):
    """Simulator and post-analysis grading wrapper; not the production input API."""
    lat0, lon0 = fields.PILOT["lat0"], fields.PILOT["lon0"]
    incident_sim.set_origin(lat0, lon0)
    scene_wind = release.get("wind_ms", 7.4) if release else 7.4
    prov = fields.FieldProvider(seed=seed, mean_wind=scene_wind, wind_dir=290.0)
    oil_cov, release_truth = None, None
    if release:
        oil_cov, release_truth = incident_sim.simulate_release(
            prov, size, seed=seed, release_start_h=release["start_h"],
            release_end_h=release["end_h"], speed_kn=release["speed_kn"],
            course_deg=release.get("course_deg"))
    img, mask, meta = sar_sim.generate_scene(seed, size=size, force_oil=want_oil,
                                            oil_cov=oil_cov, mean_wind=scene_wind)
    # Traffic generation precedes segmentation. The simulator may know truth;
    # the analysis function receives only the same ping fields as an AIS archive.
    labelled = {"vessels": []}
    if release_truth:
        ys, xs = np.nonzero(oil_cov > 0.42)
        cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) else (size/2, size/2)
        import geo
        lo, la = geo.xy_to_lonlat((cx-size/2)*80, (size/2-cy)*80, lat0, lon0)
        reference = {"centroid": [la, lo], "orientation_deg": release_truth["course_deg"] % 180}
        if traffic_mode == "constructed":
            labelled = ais_mod.build_truth_scenario(release_truth, reference, lat0, lon0, seed)
        elif traffic_mode == "random":
            labelled = ais_mod.build_random_scenario(release_truth, lat0, lon0, seed)
        else:
            raise ValueError("Unknown traffic mode")
    traffic = ais_mod.observation_tracks(labelled)
    inc = analyse_observation(model, img, traffic, prov, incident_id, acquired_utc,
                              lat0, lon0, compact)
    # Evaluation-only information is attached AFTER predictions are final.
    labels = {v["mmsi"]: v.get("truth") for v in labelled["vessels"]}
    for d in inc["detections"]:
        for c in d.get("candidates", []): c["truth"] = labels.get(c["mmsi"])
    inc["scene_truth"] = {"has_oil": meta["has_oil"], "n_lookalikes": meta["n_lookalikes"]}
    inc["release_truth"] = release_truth
    inc["truth_png"] = None if compact else _png_b64(overlay(img, mask))
    inc["provenance"] = {"data": "synthetic", "evaluation_protocol": "observation_boundary_v2",
        "seed": seed, "traffic": traffic_mode + " simulation; not real AIS",
        "environment": "shared synthetic fields; not independent observed forcing"}
    return inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/unet_slick.pt")
    ap.add_argument("--out", default="out/incidents.json")
    args = ap.parse_args()

    model = load_model(args.model)
    t0 = time.time()

    scenarios = [
        {
            "seed": 900123, "id": "INC-2026-0417", "utc": "2026-09-01T05:42:18Z",
            "title": "Sustained discharge, 41 km tile, Mumbai approaches",
            "release": {"start_h": 8.5, "end_h": 7.0, "speed_kn": 9.5,
                        "course_deg": 62.0, "wind_ms": 7.4},
        },
        {
            "seed": 900357, "id": "INC-2026-0418", "utc": "2026-09-01T17:36:02Z",
            "title": "Shorter discharge, higher wind, oblique course",
            "release": {"start_h": 6.5, "end_h": 5.0, "speed_kn": 11.0,
                        "course_deg": 148.0, "wind_ms": 9.6},
        },
        {
            "seed": 900461, "id": "INC-2026-0419", "utc": "2026-09-02T05:44:51Z",
            "title": "Look-alike scene, no discharge - correct rejection test",
            "release": None, "want_oil": False,
        },
    ]
    incidents = []
    for sc in scenarios:
        inc = build_incident(model, sc["seed"], sc["id"], sc["utc"],
                             want_oil=sc.get("want_oil", True), release=sc["release"])
        inc["title"] = sc["title"]
        top = inc["detections"][0] if inc["detections"] else None
        cand = (top or {}).get("candidates") or []
        print("%s  det=%d  top=%s  rank1=%s" % (
            sc["id"], inc["n_detections"],
            (top["verdict"] if top else "none"),
            (cand[0]["name"] + " %.0f%%" % cand[0]["score_pct"]) if cand else "-"), flush=True)
        incidents.append(inc)

    metrics = {}
    if os.path.exists("results/metrics.json"):
        metrics = json.load(open("results/metrics.json"))

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_box": fields.PILOT,
        "model_metrics": metrics,
        "incidents": incidents,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print("wrote %s (%.1f MB) in %.1fs" % (args.out, os.path.getsize(args.out) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
