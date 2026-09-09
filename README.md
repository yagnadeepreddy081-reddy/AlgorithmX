# SAMUDRA NETRA

**From slick to ship — satellite evidence for maritime accountability.**

Smart India Hackathon 2026 · Problem statement **SIH26143** · National Technical
Research Organisation · Space Technology · Software.

> *Leveraging satellite imagery to determine oil spills at sea along with AIS
> data correlations to identify the vessel responsible for the spill.*

**[▶ Open the live demo](https://samudra-netra.vercel.app)** — no install, works offline once loaded.

---

## The idea this turns on

Satellites already see oil at sea. Europe's CleanSeaNet has done it for years.
What nobody does automatically is answer the next question: **which ship put it
there.** Without that, a detection is a cleanup job. With it, it is a
prosecution.

And the reason it is hard is a detail most systems get wrong:

> **The oil is not where it was released.**

At a typical 0.3–0.5 m/s drift, oil photographed at 05:42 was discharged
10–18 km away, hours earlier. Searching AIS around the slick looks for ships in
the wrong place at the wrong time. So this system runs the ocean *backwards*
first, reconstructs a release corridor, and only then asks which vessels were
inside it.

```
Sentinel-1 SAR tile
  └─ U-Net segmentation          sea / oil / look-alike / ship / land
      └─ fragment grouping        windrows rejoined into one slick
          └─ characterisation     area, elongation, principal axis, damping contrast
              └─ look-alike filter  wind gating, shape, edge sharpness, damping band
                  └─ reverse drift  12-member backward Lagrangian ensemble
                      └─ AIS correlation  hour-by-hour test against backtracked parcels
                          └─ evidence scoring  5 weighted factors, reported separately
                              └─ ranked candidate vessels
```

---

## Results

All figures below are produced by the code in this repository and can be
reproduced with the commands in the next section.

### Segmentation — 300 held-out scenes the model never saw

| Class | IoU | Precision | Recall |
|---|---:|---:|---:|
| sea | 0.961 | 0.991 | 0.970 |
| **oil spill** | **0.737** | **0.825** | **0.873** |
| look-alike | 0.228 | 0.275 | 0.570 |
| ship | 0.992 | 0.994 | 0.998 |
| land | 0.978 | 0.983 | 0.995 |

Mean IoU **0.779**. 753,385 parameters. 14 epochs on **CPU**, no GPU used
anywhere in this project.

Oil occupies about 1 % of pixels, so a plain cross-entropy objective reaches
94 % pixel accuracy by predicting "sea" everywhere. The loss is class-weighted
cross-entropy **plus** soft multi-class Dice, which is what prevents that
collapse.

### Whole system — 38 uncurated scenes

| | |
|---|---:|
| Discharges traced to the correct ship | **27 of 30** |
| Missed the slick entirely | 3 |
| **Wrong ship ranked first** | **0** |
| **False alarms on 8 scenes containing no oil** | **0** |
| Release-origin error, median | **2.8 km** |

Sequential seeds; wind, vessel course, speed and discharge duration drawn at
random inside operationally plausible ranges. Every scene is counted, the three
failures included, and all 38 are clickable in the demo.

**The failure mode is the part worth reading.** When this system fails, it fails
by *not finding* the slick — not by naming an innocent ship.

---

## Run it

```bash
pip install -r requirements.txt
python3 src/build_all.py
```

That rebuilds everything from source in about **70 seconds** and rewrites
`public/index.html`. A trained checkpoint is committed at
`models/unet_slick.pt`, so training is skipped by default; add `--train` to redo
it from scratch (~80 min on CPU).

Open `public/index.html` in any browser. No server, no API keys, no network.

The individual stages, if you want to run them one at a time:

```bash
python3 src/train.py --epochs 14          # train the U-Net           -> models/
python3 src/pipeline.py                   # 3 incidents end to end    -> out/incidents.json
python3 src/evaluate_recovery.py --n 12   # honest batch evaluation   -> out/recovery.json
python3 src/build_library.py --n 30       # the 38-scene test set     -> out/library.json
python3 src/build_samples.py              # native-scale sample tiles
python3 src/export_weights.py             # BatchNorm folding, float16 export for the browser
python3 src/build_console.py              # bake it all into one HTML file
```

---

## The demo runs the network in the browser

`web/engine.js` is a from-scratch implementation of the U-Net forward pass in
plain JavaScript over `Float32Array`s. `src/export_weights.py` folds every
BatchNorm into its preceding convolution and writes the weights as float16, and
`build_console.py` inlines them into the page.

Consequence: you can hand the page any image during a demo and it is segmented
on the spot, on the CPU of whatever laptop is open, with the Wi-Fi switched off.
Verified against PyTorch on identical input — **the two disagree on 0.012 % of
pixels**, which is float16 rounding.

The page also refuses images that are not radar. It measures speckle statistics,
contrast and colour saturation against what Sentinel-1 GRD actually looks like,
and reports *"Input rejected — this is not Sentinel-1 radar data"* rather than
producing a confident detection on a photograph.

---

## Layout

| Path | What it is |
|---|---|
| `src/sar_sim.py` | Physics-based Sentinel-1 scene synthesiser — Gamma multi-look speckle, wind-dependent NRCS, Marangoni damping, three look-alike families |
| `src/incident_sim.py` | **Forward** truth simulator: a vessel discharges, parcels advect forward to the satellite pass |
| `src/datasets.py` | One interface over the synthesiser and the Krestenitis benchmark |
| `src/unet.py` | Compact U-Net, Dice loss, confusion-matrix metrics |
| `src/train.py` | Training and honest held-out evaluation |
| `src/characterise.py` | Fragment grouping, shape measurement, look-alike discriminator |
| `src/fields.py` | Wind and current provider (ERA5 / CMEMS in production) |
| `src/drift.py` | Backward Lagrangian ensemble and the release corridor |
| `src/ais.py` | AIS interpolation, dark-period detection, scenario builder |
| `src/score.py` | Spatio-temporal filtering and explainable evidence scoring |
| `src/pipeline.py` | End-to-end runner |
| `src/evaluate_recovery.py` | Uncurated batch evaluation |
| `src/export_weights.py` | BatchNorm folding and float16 export for the browser |
| `web/engine.js` | The U-Net forward pass in JavaScript |
| `web/analyse.js` | Characterisation and reverse drift in JavaScript |
| `web/console.template.html` | The analyst console |
| `public/` | The built, deployable page |
| `results/` | Real metrics and the training log |

---

## What this system cannot do

These are on the demo's front page too. Stating them first is deliberate.

- **It is not real-time.** Sentinel-1 revisit is days. This is forensic
  attribution and deterrence, not interception.
- **It misses most discharges.** Anything released between passes is never seen.
  The target is the repeat offender, not the individual spill — a vessel
  discharging forty times a year only has to be caught once for a port-state
  inspection to follow.
- **Oil older than about a day cannot be traced.** Drift error compounds. Past
  the point where added uncertainty exceeds 12 km the system reports *detected,
  not attributable* rather than guessing. Those hours are computed, displayed,
  and then discarded — visibly, on the timeline.
- **It can rank the wrong ship.** The output is headed "candidate vessels for
  investigation" and every score decomposes into its five factors. A lead, never
  a verdict.
- **Look-alikes are permanent.** Algal films, wind shadows and rain cells are
  dark in SAR too. Wind gating, elongation, edge sharpness and damping magnitude
  cut them down; they never reach zero.

### The training data caveat, stated plainly

**The model is trained on physics-based synthetic Sentinel-1 scenes, not on the
Krestenitis benchmark**, which is distributed behind a request form we could not
clear inside the build window.

`src/sar_sim.py` generates scenes from published C-band ocean radar physics:
NRCS as a power law in wind speed, Gamma-distributed multi-look speckle at 4.4
looks, 3–10 dB Marangoni damping inside its 3–12 m/s validity band, and three
separate look-alike families given deliberately weaker and blurrier edges than
oil. `src/datasets.py` puts both sources behind one interface, so

```bash
python3 src/train.py --source krestenitis
```

retrains the same network on the real benchmark without another line changing.

**Treat the IoU figures above as a lower bound on a solved problem, not as
evidence of field accuracy.** AIS tracks are likewise simulated;
`ais.load_tracks()` is the single function that changes when Global Fishing
Watch is wired in.

### Why the recovery test is not circular

An earlier version placed the "guilty" vessel using the output of the drift
model, which guarantees a hit and proves nothing. In this version causality runs
the correct way: `incident_sim.py` steams a vessel, opens a valve, releases oil
parcels and advects them **forward** to the moment of acquisition, and the
drifted cloud is what gets rendered into the SAR scene. Detection and
attribution never see any of it.

The truth uses one fixed wind-drift factor; the hindcast runs a twelve-member
ensemble across 0.020–0.045 and does not know which member is right. That
asymmetry is the reason the ensemble exists.

---

## Production data feeds

Every input is free, machine-readable and already operational. No hardware, no
procurement, no paid API.

| Feed | Source |
|---|---|
| Sentinel-1 SAR GRD | Copernicus Data Space Ecosystem |
| AIS vessel tracks | Global Fishing Watch v3 API, or a national AIS archive |
| 10 m winds | ECMWF ERA5 via the Copernicus Climate Data Store |
| Surface currents | Copernicus Marine Service global analysis and forecast |
| Coastline / land mask | GSHHG · OpenStreetMap |

## References

1. Krestenitis, M. et al. (2019). *Oil Spill Identification from Satellite
   Images Using Deep Neural Networks.* Remote Sensing 11(15), 1762.
2. Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks
   for Biomedical Image Segmentation.* MICCAI.
3. Dagestad, K.-F. et al. (2018). *OpenDrift v1.0: a generic framework for
   trajectory modelling.* Geoscientific Model Development 11, 1405–1420.
4. Indian Coast Guard — *National Oil Spill Disaster Contingency Plan (NOS-DCP).*
5. INCOIS (MoES) — *Ocean State Forecast & Oil Spill Trajectory Advisory Services.*
6. IMO — *MARPOL Annex I: Prevention of Pollution by Oil.*
7. EMSA — *CleanSeaNet satellite oil-spill monitoring service.*

## Licence

MIT — see [LICENSE](LICENSE).
