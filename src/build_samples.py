"""
Sample tiles for the live-analysis demo.

These are written at the network's native working scale -- 256 x 256 pixels at
80 m ground sampling, which is exactly the tile a full Sentinel-1 scene is cut
into in production.  The library scenes on the main console are 512 px, and
resampling one of those down to 256 halves the apparent pixel scale, which
makes a slick two pixels wide instead of four and quietly degrades the result.

Fresh seeds, unrelated to anything in the training set or the test set, so a
judge picking one of these is picking a scene the system has genuinely never
been run on.  A mix of discharges, look-alike-only scenes and one empty sea.

Files land in deploy/sample_tiles/ so the judge can also open them from disk.
"""

import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import fields
import incident_sim
import sar_sim

SPECS = [
    # (seed, label, release or None)
    (515101, "Discharge, moderate wind", {"start_h": 7.0, "end_h": 5.6, "speed_kn": 10.0, "course_deg": 41.0, "wind_ms": 7.2}),
    (515233, "Discharge, brisk wind, oblique", {"start_h": 6.5, "end_h": 5.2, "speed_kn": 11.5, "course_deg": 133.0, "wind_ms": 9.8}),
    (515377, "Short discharge, light wind", {"start_h": 6.0, "end_h": 5.1, "speed_kn": 9.0, "course_deg": 78.0, "wind_ms": 5.4}),
    (515489, "Discharge near a coastline", {"start_h": 7.5, "end_h": 6.0, "speed_kn": 10.5, "course_deg": 196.0, "wind_ms": 8.1}),
    (515601, "Look-alikes only, no oil", None),
    (515733, "Look-alikes only, no oil", None),
    (515877, "Calm sea, wind shadow", None),
    (515941, "Busy shipping lane, no oil", None),
]

OUT_DIR = "public/sample_tiles"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("out", exist_ok=True)
    lat0, lon0 = fields.PILOT["lat0"], fields.PILOT["lon0"]
    incident_sim.set_origin(lat0, lon0)
    entries = []

    for seed, label, rel in SPECS:
        oil_cov = None
        wind = 7.4
        if rel:
            wind = rel["wind_ms"]
            prov = fields.FieldProvider(seed=seed, mean_wind=wind, wind_dir=290.0)
            oil_cov, _ = incident_sim.simulate_release(
                prov, 256, seed=seed,
                release_start_h=rel["start_h"], release_end_h=rel["end_h"],
                speed_kn=rel["speed_kn"], course_deg=rel["course_deg"],
            )
        img, mask, meta = sar_sim.generate_scene(
            seed, size=256, force_oil=bool(rel), oil_cov=oil_cov, mean_wind=wind)

        g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(g, mode="L")
        name = "tile_%d.png" % seed
        im.save(os.path.join(OUT_DIR, name), optimize=True)

        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        entries.append({
            "id": "TILE-%d" % seed,
            "label": label,
            "wind_ms": round(float(wind), 1),
            "has_oil": bool(rel),
            "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        })
        print("  %-28s %s  %5.1f kB" % (label, name, os.path.getsize(os.path.join(OUT_DIR, name)) / 1000))

    with open("out/samples.json", "w") as f:
        json.dump({"tiles": entries}, f, separators=(",", ":"))

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(
            "SAMUDRA NETRA - sample Sentinel-1 tiles\n"
            "======================================\n\n"
            "256 x 256 pixels, 80 m ground sampling, 20.5 km across. This is the tile\n"
            "size the segmentation network works on; a full Sentinel-1 scene is cut\n"
            "into these before processing.\n\n"
            "Use them with the 'Analyse a new image' button on the console. Four of\n"
            "these eight contain an oil discharge and four do not. They were generated\n"
            "from seeds unrelated to the training set and the test set, so the system\n"
            "has never been run on them.\n\n"
            "You can also hand the console any other image at all - it will tell you\n"
            "whether what it received looks like radar data before it reports anything.\n"
        )
    print("wrote %d tiles, %.2f MB embedded" % (len(entries), os.path.getsize("out/samples.json") / 1e6))


if __name__ == "__main__":
    main()
