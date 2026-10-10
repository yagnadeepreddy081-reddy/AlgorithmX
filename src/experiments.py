"""Predeclared baselines/ablations on identical observed inputs; no truth access."""
import numpy as np
import ais
import geo
import score

METHODS = {
    "drift_geometry": {"corridor_hit": .5, "proximity": .5, "axis_alignment": 0,
                       "dark_period": 0, "vessel_type": 0},
    "full": score.WEIGHTS,
    "no_axis": dict(score.WEIGHTS, axis_alignment=0),
    "no_ais_gap": dict(score.WEIGHTS, dark_period=0),
    "no_type_prior": dict(score.WEIGHTS, vessel_type=0),
}


def _winner(candidates):
    ranked = [c for c in candidates if c["status"] == "CANDIDATE"]
    if not ranked: return {"mmsi": None, "score": None, "abstained": True}
    top = ranked[0]
    return {"mmsi": top["mmsi"], "score": top["score"], "abstained": False}


def compare_methods(det, drift, traffic, lat0, lon0):
    """The visible-slick baseline uses minimum historical distance to its centroid.

    All methods use the same available, short-gap AIS samples at usable hours.
    This isolates changing the spatial hypothesis, then the scoring features.
    """
    hours = np.array([f["hours_back"] for f in drift["frames"]
                      if f["usable"] and f["hours_back"] > 0], dtype=float)
    distances = []
    for v in traffic["vessels"]:
        samples = [s for s in ais.resample_track(v["pings"], hours) if s is not None]
        if samples:
            distance = min(geo.haversine_km(det["centroid"][0], det["centroid"][1],
                                           s["lat"], s["lon"]) for s in samples)
            distances.append((distance, v["mmsi"]))
    baseline = {"mmsi": None, "score": None, "abstained": True}
    if distances:
        distance, identity = min(distances)
        baseline = {"mmsi": identity, "distance_km": round(distance, 3), "abstained": False}
    out = {"visible_slick_nearest": baseline}
    for name, weights in METHODS.items():
        out[name] = _winner(score.score_vessels(det, drift, traffic, lat0, lon0, weights))
    return out


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a binomial proportion; None for no trials."""
    if total == 0: return None
    p = successes / total
    den = 1 + z*z/total
    centre = (p + z*z/(2*total))/den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [round(max(0,centre-half),4), round(min(1,centre+half),4)]


def summarise_comparisons(trials):
    names = ["visible_slick_nearest", *METHODS]
    out = {}
    for name in names:
        wins = sum(t.get("methods",{}).get(name,{}).get("correct",False) for t in trials)
        attempted = sum(not t.get("methods",{}).get(name,{"abstained":True}).get("abstained",True)
                        for t in trials)
        out[name] = {"correct": wins, "total_incidents": len(trials), "ranked": attempted,
                     "abstained_or_missed": len(trials)-attempted,
                     "rank1_over_all": round(wins/len(trials),4) if trials else None,
                     "rank1_when_ranked": round(wins/attempted,4) if attempted else None,
                     "rank1_over_all_95pct_wilson": wilson_interval(wins,len(trials))}
    return out
