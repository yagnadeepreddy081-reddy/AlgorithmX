# Samudra Netra

Radar-like oil-slick segmentation, backward drift and explainable candidate-vessel ranking. The repository is AlgorithmX; the prototype is Samudra Netra.

**Research prototype using synthetic data.** Images, AIS traffic and environmental fields in the demo are generated. Evidence scores are manually weighted indices, not probabilities of responsibility. Real-world accuracy has not been measured.

## What works

- Compact five-class U-Net, trained from scratch using PyTorch; the existing checkpoint is preserved.
- Oil-fragment grouping, footprint/shape measurements and heuristic look-alike screening.
- Twelve-member backward particle ensemble, midpoint integration and explicit uncertainty gate.
- Time-matched traffic analysis with long-gap abstention and distinct missing-data/model-exclusion statuses.
- Six predeclared comparison methods, per-trial results and uncertainty intervals.
- Browser image inference, deterministic illustrative drift, stored scene exploration, interactive evidence weights and downloadable case JSON.
- Complete rebuild, provenance manifest, pinned local environment and automated reliability checks.

## Quick start

Open `public/index.html`. The core app uses embedded data and code and no external font request. Library results are precomputed; the new-image button runs the network locally. Uploaded images require the demo's assumed scale/stretch; they are not automatically calibrated or georeferenced and do not receive vessel attribution.

For reproducible preparation from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python src/build_all.py
```

The lock records the verified Python 3.13 macOS environment; other platforms may need a compatible PyTorch wheel. `requirements.txt` gives broader supported minimum versions. The generated scene library includes failures, and builds regenerate recovery comparisons. Retraining is optional: `python src/build_all.py --train --epochs 14`.

## Results and evidence

Historical segmentation training is documented in `results/metrics.json` and `results/training.log`: 1,200 training, 200 validation and 300 test synthetic scenes; 753,385 trainable parameters; 14 epochs. Historical oil IoU is 0.737. This is not field accuracy.

The changed pipeline uses protocol `observation_boundary_v2`. New comparison results are generated in `results/research-comparison.json`; they include misses and abstentions. Do not reuse the original 27/30 recovery claim for this modified protocol.

Current measured results: the 12 constructed discharge cases recovered 12/12 for **all six methods**. On the separately seeded 24-case random-traffic batch, inspect `results/stress-comparison.json` for the fixed-method results; this is still simplified synthetic traffic. The 300-scene segmentation rerun confirms oil IoU **0.737** and look-alike IoU **0.228**, a significant weakness. Neither batch establishes field accuracy. No retraining was performed in this improvement pass.

Read [the research protocol](docs/RESEARCH_PROTOCOL.md) for experiment definitions and remaining threats to validity. Ground-truth roles/times enter simulation and post-hoc grading only. The original traffic batch is constructed; an additional 24-scene batch has independently randomised background traffic, and the forward/hindcast environmental fields are shared.

## Real labelled data

No real dataset is included. Obtain a dataset with suitable permission and verify labels and independent splits. Place images under `data/krestenitis/test/images` and single-channel class masks (0 sea, 1 oil, 2 look-alike, 3 ship, 4 land) under `test/labels_1D`, then:

```sh
SAMUDRA_DATA_ROOT=data/krestenitis python src/evaluate_segmentation.py --source krestenitis
```

That evaluates the existing synthetic-trained checkpoint; it does not make it a real-trained model. To retrain, provide train/val/test directories and use `python src/train.py --source krestenitis --epochs 14`. No ERA5, CMEMS or live AIS backend is implemented. The observation-analysis API creates an integration boundary, not an operational data service.

## File map

`src/` contains simulation, model, analysis, build and evaluation modules. `web/` contains the template and browser inference/analysis engines. `public/` is the built static site. `tests/` verifies truth separation, wind consistency, gap handling, score invariance, browser repeatability and network parity. `results/` stores historical training and newly generated comparison evidence.

Run `python -m pytest -q` for the reliability checks. A ready-to-use GitHub Actions workflow was prepared separately but could not be uploaded because the saved GitHub token lacks workflow permission. No remote CI run is claimed.

## Limitations

The model may miss oil and confuse look-alikes. Drift assumptions, coarse timing, hand-chosen scores, missing coverage and simplified traffic can yield wrong rankings. Unknown data is not evidence of innocence. A lead is for investigation, never a verdict. The 12 km uncertainty gate is a chosen rule rather than a validated confidence guarantee.

The project uses the established U-Net design and a custom simplified drift implementation, not a novel U-Net architecture or the OpenDrift package. Existing services such as CleanSeaNet already combine satellite surveillance and vessel information; this project does not claim to invent source attribution.

## Deployment

Serve `public/` as static files. Vercel configuration is included. App Engine deployment has not been completed in this work. Local validation does not imply a publicly deployed site or accepted research paper.

MIT licence; see LICENSE.
