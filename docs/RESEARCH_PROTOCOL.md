# Research protocol v2

This protocol is defined before examining the new comparison results. It does not claim a superior method in advance.

## Question

Does comparing vessel positions with backward-drifted oil improve first-ranked vessel recovery relative to comparing them with the visible slick? What changes when individual evidence factors are removed?

## Analysis boundary

`pipeline.analyse_observation` receives only an image, environmental provider, observed traffic fields, location and acquisition time. It rejects traffic truth labels and supplied release-time estimates. The simulation wrapper builds traffic before segmentation, strips role labels, runs analysis and only then attaches truth for evaluation/display.

The release time is inferred from the matching hours of the highest-ranked eligible candidate, not from the simulated valve midpoint. The origin-error metric uses the nearest available usable frame to that inferred time. This remains an approximate cloud-centroid metric, not exact release-footprint recovery.

## Compared methods

- Visible-slick baseline: minimum vessel distance to observed slick centroid over usable sampled history.
- Drift geometry: equal corridor/proximity weights.
- Full: fixed weights 0.34/0.26/0.22/0.11/0.07.
- Without course axis, without AIS gap, without type prior: remove the named weight and renormalise the rest.

Methods receive identical images and available hourly track samples. The nearest-slick baseline always ranks an available track; drift methods can abstain if no vessel enters the corridor. Report both coverage/abstention and success across all discharge incidents.

## Experiment

`evaluate_recovery.py` keeps the original 12 discharge seeds (770000 + 37i) and eight clean seeds (660000 + 53i). Its parameter generator uses seed 20260902. Do not tune thresholds or weights on this evaluation batch. The 38-scene display library is a separate batch; report it separately. New results must not be combined with historical protocol-v1 results.

Outputs include per-trial method winners, failures and abstentions, all-incident and ranked-only denominators, Wilson intervals for all-incident success, origin errors and data-source/protocol labels. Wilson intervals represent sampling variation in these simulated trials; they do not quantify synthetic-to-real uncertainty.

## Remaining threats to validity

The generated traffic contains deliberately constructed competitors, uses the same vessel roles between scenes, and can be much easier than real dense traffic. Forward simulation and hindcast share generated environmental fields. Wind consistency is fixed, but field accuracy is not established. Test splits share simulator assumptions. The checkpoint has not been retrained on real imagery. Screening thresholds and ranking factors are uncalibrated heuristics. AIS gaps can arise from coverage, not deliberate behaviour. Real incident evaluation is necessary before operational claims.

## Next experiment

Freeze method settings, add an independently seeded larger batch and more varied traffic without selecting successful outcomes, and test paired per-scene differences. Then evaluate representative real labelled SAR imagery and documented incidents with matched, permitted AIS and measured/reanalysis environmental inputs. Do not reuse test cases for selecting weights.

## Additional stress batch (settings frozen before running)

Keep all six methods and thresholds unchanged. Use 24 discharge seeds `920000 + 37i`, parameter seed `20261010`, and ten random background vessels per incident. Their acquisition positions are uniform in a 110 km square, courses uniform 0–360°, speeds 5–18 knots; 35% receive an observation gap. Background generation never receives the image, slick reference or corridor. The true vessel is still reconstructed from simulation, and environmental fields are still shared. Eight clean seeds repeat the original controls and must not be presented as eight additional independent controls. This is a traffic stress test, not real validation or a claim about all possible traffic.

Command: `python src/evaluate_recovery.py --n 24 --traffic random --seed-base 920000 --parameter-seed 20261010 --out out/stress.json --results results/stress-comparison.json`. Report every outcome, including ties and failures. Do not tune methods using this batch.

The AIS-gap factor counts missing hours between the first and last supported corridor matches, excluding observed endpoints. It does not impute a location inside the gap; a single match receives no gap credit. This arithmetic was corrected before the final archived rerun; settings and seeds remain unchanged.
