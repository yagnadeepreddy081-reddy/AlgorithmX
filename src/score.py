"""
Stage 6 -- spatio-temporal filtering and explainable evidence scoring.

Every candidate is scored on five independent factors.  They are kept separate
and reported separately on purpose: an officer has to be able to say *why* a
vessel is ranked first, and a judge has to be able to attack any single factor
without the whole thing collapsing into an unexplainable number.

  corridor_hit   fraction of usable hindcast hours in which the vessel's
                 interpolated position lay inside the 90 % release envelope
  proximity      how close the vessel came to the envelope centre, relative to
                 the envelope radius at that hour
  axis_alignment agreement between the vessel's mean course and the slick's
                 principal axis.  An operational discharge is released while
                 underway, so the slick is a ribbon lying *along* the track.
                 A vessel that crossed the corridor perpendicular did not make
                 a ribbon in that direction.
  dark_period    transponder silence overlapping the release window
  vessel_type    tankers and chemical carriers carry the cargo and the slop
                 tanks; this is a weak prior, weighted accordingly

Deliberate design choices
-------------------------
* Hours where the drift envelope has grown past the usability radius are
  DISCARDED before scoring, not down-weighted.  If the search circle is 20 km
  wide, half the Arabian Sea's traffic is "inside" it and the match means
  nothing.
* The output is headed "candidate vessels for investigation".  The system
  produces a lead, never a verdict.
* A vessel with no usable corridor hours is reported as CLEARED with the
  physical reason, rather than silently dropped -- showing the exclusions is
  what makes the ranking credible.
"""

import math
import numpy as np

import geo

WEIGHTS = {
    "corridor_hit": 0.34,
    "proximity": 0.26,
    "axis_alignment": 0.22,
    "dark_period": 0.11,
    "vessel_type": 0.07,
}

TYPE_PRIOR = {
    "Crude oil tanker": 1.00,
    "Chemical tanker": 0.95,
    "Product tanker": 0.95,
    "Bulk carrier": 0.55,
    "Container ship": 0.45,
    "General cargo": 0.45,
    "Fishing": 0.30,
}


def _clip_track(pings, lat0, lon0, half_km):
    """Keep only the part of a track inside a square window about the tile."""
    out = []
    for p in sorted(pings, key=lambda q: -q["t_h"])[::2]:
        x, y = geo.lonlat_to_xy(p["lon"], p["lat"], lat0, lon0)
        if abs(x) / 1000.0 <= half_km and abs(y) / 1000.0 <= half_km:
            out.append([round(p["lat"], 5), round(p["lon"], 5)])
    return out


def _mean_course(samples):
    """Circular mean of course over ground, as an undirected axis in 0-180."""
    ang = [math.radians(2 * s["cog_deg"]) for s in samples]
    if not ang:
        return None
    s = sum(math.sin(a) for a in ang)
    c = sum(math.cos(a) for a in ang)
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def score_vessels(det, drift_result, ais_scenario, lat0, lon0):
    frames = [f for f in drift_result["frames"] if f["usable"] and f["hours_back"] > 0]
    usable_hours = [f["hours_back"] for f in frames]
    n_usable = len(usable_hours)
    slick_axis = det["orientation_deg"]

    results = []
    for v in ais_scenario["vessels"]:
        pings = v["pings"]
        dark = __import__("ais").detect_dark_periods(pings)
        samples = __import__("ais").resample_track(pings, np.array(usable_hours, dtype=float))

        # ------------------------------------------------------------------
        # Corridor test.
        #
        # The release envelope is NOT a circle around a centroid.  A slick from
        # an operational discharge is a ribbon: its head and its tail were
        # released at different times and from different points along the
        # ship's track, so backtracking it produces an elongated parcel cloud,
        # not a blob.  Testing a vessel against the distance to the cloud's
        # centroid throws that structure away and misses true matches sitting
        # on the far end of the cloud.
        #
        # So the test is distance to the NEAREST backtracked parcel, with a
        # tolerance that grows with the cloud's own spread.
        # ------------------------------------------------------------------
        inside = []
        rel_dists = []
        abs_dists = []
        for f, s in zip(frames, samples):
            if s is None:
                continue
            vx, vy = geo.lonlat_to_xy(s["lon"], s["lat"], lat0, lon0)
            px, py = f["_xy"]
            d_km = float(np.min(np.hypot(px - vx, py - vy))) / 1000.0
            abs_dists.append(d_km)
            # tolerance: AIS interpolation error + parcel-cloud coarseness,
            # widened slightly as the ensemble spread grows
            tol = max(1.5, 0.6 * f.get("uncertainty_km", 0.0))
            rel_dists.append(d_km / tol)
            if d_km <= tol:
                inside.append(
                    {"hours_back": f["hours_back"], "dist_km": round(d_km, 2), "tolerance_km": round(tol, 2)}
                )

        if not abs_dists:
            results.append(
                {
                    "mmsi": v["mmsi"], "name": v["name"], "type": v["type"], "flag": v["flag"],
                    "length_m": v["length_m"], "score": 0.0, "status": "CLEARED",
                    "reason": "No AIS coverage within the usable hindcast window.",
                    "factors": {}, "hours_in_corridor": 0, "usable_hours": n_usable,
                    "closest_km": None, "dark_periods": dark, "track": [], "truth": v.get("truth"),
                }
            )
            continue

        # A sustained discharge produces a RUN of consecutive matching hours.
        # An isolated single-hour hit is what coincidence looks like, so the
        # corridor factor is scored on the longest consecutive run, normalised
        # against a 4-hour reference discharge -- not on a raw hit count.
        hit_hours = sorted(h["hours_back"] for h in inside)
        longest, run = 0, 0
        prev = None
        for h in hit_hours:
            run = run + 1 if (prev is not None and h - prev == 1) else 1
            prev = h
            longest = max(longest, run)
        # How long *should* the run be?  The slick's own length answers it: oil
        # released continuously while underway makes a ribbon whose length is
        # the vessel's speed times the time the valve was open.  So the expected
        # number of matching hours is slick_length / vessel_speed.  Scoring the
        # run against that, instead of a fixed constant, means a short slick
        # from a short discharge is not punished for being short.
        v_kmh = max(3.0, float(np.mean([s["sog_kn"] for s in samples if s is not None])) * 1.852)
        # Floored at 1.5 h: a single isolated matching hour is never full
        # corridor credit, however short the slick.  One hour is what
        # coincidence looks like.
        expected_h = float(np.clip(det["length_km"] / v_kmh, 1.5, 6.0))
        hit = float(np.clip(longest / expected_h, 0.0, 1.0))
        closest = min(abs_dists)
        best_rel = min(rel_dists)
        proximity = float(np.clip(1.0 - (best_rel - 0.35) / 1.65, 0.0, 1.0))

        course_axis = _mean_course([s for s in samples if s is not None])
        if course_axis is None:
            align_deg, alignment = None, 0.0
        else:
            align_deg = geo.axis_difference_deg(course_axis, slick_axis)
            # full credit within 15 deg, zero beyond 55 deg
            alignment = float(np.clip((55.0 - align_deg) / 40.0, 0.0, 1.0))

        # Transponder silence is scored against the DISCHARGE window, not the
        # whole hindcast.  A vessel dark for the two hours the oil was going
        # into the water is the case this factor exists to catch; dividing that
        # by an eighteen-hour hindcast dilutes it to nothing and the factor
        # stops meaning anything.
        window = set(usable_hours)
        dark_score = 0.0
        for d in dark:
            lo, hi = d["to_h_before"], d["from_h_before"]
            covered = sum(1 for h in window if lo <= h <= hi)
            dark_score = max(dark_score, float(np.clip(covered / max(expected_h, 1.5), 0.0, 1.0)))

        type_score = TYPE_PRIOR.get(v["type"], 0.5)

        factors = {
            "corridor_hit": round(hit, 3),
            "proximity": round(proximity, 3),
            "axis_alignment": round(alignment, 3),
            "dark_period": round(dark_score, 3),
            "vessel_type": round(type_score, 3),
        }
        raw = sum(WEIGHTS[k] * factors[k] for k in WEIGHTS)

        # A vessel that was never inside the envelope in any usable hour is
        # physically excluded.  No amount of vessel-type prior or course
        # coincidence should be able to promote it.
        if len(inside) == 0:
            status = "CLEARED"
            score = 0.0
            if closest > 15:
                reason = "Never within the release envelope; closest approach %.1f km." % closest
            else:
                reason = (
                    "Approached to %.1f km but only outside the release window - "
                    "the vessel was not in the origin area at the hours the oil was released."
                    % closest
                )
        else:
            score = raw
            status = "CANDIDATE"
            reason = (
                "Inside the release corridor for %d of %d usable hindcast hours, "
                "longest unbroken run %d h. Closest approach to the backtracked oil %.1f km."
                % (len(inside), n_usable, longest, closest)
            )

        results.append(
            {
                "mmsi": v["mmsi"], "name": v["name"], "type": v["type"], "flag": v["flag"],
                "length_m": v["length_m"],
                "score": round(float(score), 4),
                "score_pct": round(float(score) * 100.0, 1),
                "status": status,
                "reason": reason,
                "factors": factors,
                "weights": WEIGHTS,
                "contributions": {k: round(WEIGHTS[k] * factors[k], 4) for k in WEIGHTS},
                "hours_in_corridor": len(inside),
                "longest_run_h": longest,
                "expected_run_h": round(expected_h, 1),
                "implied_discharge_minutes": round(expected_h * 60),
                "usable_hours": n_usable,
                "closest_km": round(closest, 2),
                "mean_course_axis_deg": None if course_axis is None else round(course_axis, 1),
                "slick_axis_deg": round(slick_axis, 1),
                "axis_difference_deg": None if align_deg is None else round(align_deg, 1),
                "dark_periods": dark,
                "corridor_hits": inside,
                "track": [
                    {"t_h": s["t_h"], "lat": s["lat"], "lon": s["lon"], "observed": s["observed"]}
                    for s in samples if s is not None
                ],
                # Track clipped to a window around the tile: a vessel's 14-hour
                # history runs for 250+ km and would make the map unreadable.
                "full_track": _clip_track(pings, lat0, lon0, 34.0),
                "truth": v.get("truth"),
            }
        )

    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results
