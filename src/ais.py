"""
Stage 5 -- AIS ingest, track reconstruction and dark-period detection.

Production source: Global Fishing Watch `/v3/events` and `/v3/vessels` APIs, or
a national AIS archive.  Both deliver the same primitive -- a sequence of
(timestamp, lat, lon, sog, cog, mmsi) pings at irregular intervals.  This module
consumes that primitive; `load_tracks()` is the only function that needs to
change when the live feed is connected.

Two things happen here that matter for attribution:

1. Interpolation.  AIS pings arrive every 2-10 minutes when the transponder is
   on, but the drift model works on a fixed hourly grid.  Tracks are resampled
   onto that grid by local linear interpolation, so "was this vessel inside the
   release envelope at 03:00" becomes an answerable question.

2. Dark-period detection.  A gap longer than DARK_GAP_MIN minutes while the
   vessel is at sea is recorded as a dark period.  A gap may reflect receiver coverage or equipment issues; it does not prove
   intent. Positions inside gaps longer than the threshold are left unknown.
   Short-gap interpolations are labelled separately from actual observations.
"""

import math
import numpy as np

import geo

DARK_GAP_MIN = 45.0  # minutes of silence at sea that counts as a dark period


def _interp_position(p0, p1, frac):
    return (
        p0[0] + (p1[0] - p0[0]) * frac,
        p0[1] + (p1[1] - p0[1]) * frac,
    )


def resample_track(pings, t_hours):
    """
    pings   : list of dicts {t_h (hours before acquisition, positive), lat, lon,
              sog_kn, cog_deg}
    t_hours : array of hours-before-acquisition to resample onto

    Returns list of dicts per requested hour, each flagged observed/interpolated.
    """
    if not pings:
        return [None for _ in t_hours]
    pings = sorted(pings, key=lambda p: -p["t_h"])  # oldest first
    ts = np.array([p["t_h"] for p in pings])
    out = []
    for th in t_hours:
        if th > ts.max() or th < ts.min():
            out.append(None)
            continue
        j = int(np.argmin(np.abs(ts - th)))
        # bracketing pings
        lo = np.where(ts >= th)[0]
        hi = np.where(ts <= th)[0]
        if len(lo) == 0 or len(hi) == 0:
            out.append(None)
            continue
        i0 = lo[np.argmin(ts[lo])]
        i1 = hi[np.argmax(ts[hi])]
        p0, p1 = pings[i0], pings[i1]
        gap_min = abs(p0["t_h"] - p1["t_h"]) * 60.0
        if gap_min > DARK_GAP_MIN:
            out.append(None)
            continue
        if i0 == i1:
            lat, lon = p0["lat"], p0["lon"]
        else:
            span = p0["t_h"] - p1["t_h"]
            frac = 0.0 if span == 0 else (p0["t_h"] - th) / span
            lat, lon = _interp_position((p0["lat"], p0["lon"]), (p1["lat"], p1["lon"]), frac)
        out.append(
            {
                "t_h": float(th),
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "sog_kn": float(pings[j]["sog_kn"]),
                "cog_deg": float(pings[j]["cog_deg"]),
                "observed": bool(i0 == i1),
                "interpolated": bool(i0 != i1),
                "gap_min": round(float(gap_min), 1),
            }
        )
    return out


def detect_dark_periods(pings):
    """Return list of (start_h_before, end_h_before, duration_min)."""
    pings = sorted(pings, key=lambda p: -p["t_h"])
    dark = []
    for a, b in zip(pings[:-1], pings[1:]):
        gap = (a["t_h"] - b["t_h"]) * 60.0
        if gap > DARK_GAP_MIN:
            dark.append(
                {
                    "from_h_before": round(a["t_h"], 2),
                    "to_h_before": round(b["t_h"], 2),
                    "duration_min": round(gap, 0),
                }
            )
    return dark


# ---------------------------------------------------------------------------
# offline scenario builder
# ---------------------------------------------------------------------------

def _straight_track(lat0, lon0, start_xy, course_deg, speed_kn, t_from, t_to,
                    ping_min=6.0, dark=None, rng=None):
    """Build a ping list for a vessel steaming a constant course."""
    rng = rng or np.random.default_rng(0)
    v = speed_kn * 0.514444  # m/s
    th = math.radians(course_deg)
    pings = []
    t = t_from
    while t >= t_to - 1e-9:
        dt_s = (t_from - t) * 3600.0
        x = start_xy[0] + v * math.sin(th) * dt_s
        y = start_xy[1] + v * math.cos(th) * dt_s
        # small navigational wander
        x += rng.normal(0, 120)
        y += rng.normal(0, 120)
        lon, lat = geo.xy_to_lonlat(x, y, lat0, lon0)
        in_dark = dark is not None and dark[1] <= t <= dark[0]
        if not in_dark:
            pings.append(
                {
                    "t_h": round(t, 3),
                    "lat": lat,
                    "lon": lon,
                    "sog_kn": round(speed_kn + rng.normal(0, 0.25), 2),
                    "cog_deg": round((course_deg + rng.normal(0, 1.5)) % 360, 1),
                }
            )
        t -= ping_min / 60.0
    return pings


def build_truth_scenario(truth, det, lat0, lon0, seed=0):
    """
    Build the AIS picture from GROUND TRUTH, with no reference to the drift
    model's output.

    This is the important difference from a staged demo.  The guilty vessel's
    track is the one the forward incident simulator actually used to release the
    oil; the backward drift model has never seen it and has to find it.  The
    three innocent vessels are placed relative to the *observed* slick, which is
    what a real analyst sees, so they are exactly the traffic that looks
    suspicious to the naked eye and must be cleared by physics.

      GUILTY  : the real discharging vessel
      DECOY   : passes closer to the visible slick than the guilty vessel does,
                but two hours after the acquisition window that matters -- the
                oil was already on the water before it arrived
      DARK    : goes dark inside the release window, but 30+ km away
      TRANSIT : ordinary lane traffic
    """
    rng = np.random.default_rng(seed + 7)
    course = truth["course_deg"]
    speed = truth["speed_kn"]
    v = speed * 0.514444
    th = math.radians(course)

    # reconstruct the vessel's position history from the truth track, then
    # convert it into realistic AIS pings (6 min interval, positional noise)
    guilty_pings = []
    mid_h = 0.5 * (truth["valve_open_h_before"] + truth["valve_close_h_before"])
    rx, ry = geo.lonlat_to_xy(truth["true_release_lon"], truth["true_release_lat"], lat0, lon0)
    t = truth["valve_open_h_before"] + 8.0
    while t >= 0.0:
        dt = (mid_h - t) * 3600.0
        x = rx + math.sin(th) * v * dt + rng.normal(0, 55)
        y = ry + math.cos(th) * v * dt + rng.normal(0, 55)
        lon, lat = geo.xy_to_lonlat(x, y, lat0, lon0)
        guilty_pings.append(
            {
                "t_h": round(t, 3), "lat": lat, "lon": lon,
                "sog_kn": round(speed + rng.normal(0, 0.2), 2),
                "cog_deg": round((course + rng.normal(0, 1.4)) % 360, 1),
            }
        )
        t -= 0.1
    vessels = [
        {
            "mmsi": "419087341", "name": "MT ARJUN PRIDE", "type": "Crude oil tanker",
            "flag": "India", "length_m": 183, "pings": guilty_pings,
            "truth": "guilty - released the oil",
        }
    ]

    # --- decoy: crosses the visible slick, but after the release window ----
    sx, sy = geo.lonlat_to_xy(det["centroid"][1], det["centroid"][0], lat0, lon0)
    c_d = (det["orientation_deg"] + 74.0) % 360
    sp_d = 15.5
    vessels.append(
        {
            "mmsi": "477553900", "name": "MV HAI FENG 09", "type": "Container ship",
            "flag": "Hong Kong", "length_m": 232,
            "pings": _straight_track(lat0, lon0,
                                     (sx - math.sin(math.radians(c_d)) * sp_d * 0.514444 * 2.2 * 3600,
                                      sy - math.cos(math.radians(c_d)) * sp_d * 0.514444 * 2.2 * 3600),
                                     c_d, sp_d, 2.2, 0.0, rng=rng),
            "truth": "innocent - crossed the slick after the release window",
        }
    )

    # --- the hard case: a second vessel that really was there --------------
    #
    # A demo where only the guilty ship ever enters the corridor proves
    # nothing about the scoring, because any weighting produces the same
    # answer.  This vessel crosses the release area *during* the release
    # window, is a chemical tanker, and goes dark while it is there -- three
    # of the five factors point straight at it.
    #
    # What clears it is the fourth: the slick is a ribbon lying along a course
    # this ship was not steering.  Oil released from it while underway would
    # have made a ribbon almost sixty degrees away from the one in the image.
    # That is the whole argument for measuring the slick's principal axis, and
    # it is the factor an analyst can attack or defend in front of a court.
    c_x = (course + 58.0) % 360
    spd_x = 12.5
    vx = spd_x * 0.514444
    thx = math.radians(c_x)
    vessels.append(
        {
            "mmsi": "563112400", "name": "MT SEROJA LIMA", "type": "Chemical tanker",
            "flag": "Singapore", "length_m": 169,
            # start the track 4 h of steaming short of the release point, so
            # that the vessel is standing on it at the moment the oil went in
            "pings": _straight_track(
                lat0, lon0,
                (rx - math.sin(thx) * vx * 4.0 * 3600.0,
                 ry - math.cos(thx) * vx * 4.0 * 3600.0),
                c_x, spd_x, mid_h + 4.0, 0.0,
                dark=(mid_h + 1.2, mid_h - 1.2), rng=rng,
            ),
            "truth": "innocent - crossed the release area, but on a course that cannot have made this slick",
        }
    )

    # --- dark-period vessel, far away --------------------------------------
    kx, ky = 33000.0, -26000.0
    c_k = (math.degrees(math.atan2(kx, ky)) + 360.0) % 360.0  # steaming outbound
    vessels.append(
        {
            "mmsi": "538007712", "name": "MT GULF SENTINEL", "type": "Chemical tanker",
            "flag": "Marshall Islands", "length_m": 176,
            "pings": _straight_track(lat0, lon0, (rx + kx, ry + ky), c_k, 10.8,
                                     truth["valve_open_h_before"] + 6.0, 0.0,
                                     dark=(mid_h + 1.5, mid_h - 1.5), rng=rng),
            "truth": "innocent - dark period inside the window, but 30+ km away",
        }
    )

    # --- ordinary lane traffic ---------------------------------------------
    # Lane traffic is placed on the outbound shipping lane, offset from the
    # incident and steaming away from it, so it never enters the corridor.
    for nm, mm, tp, fl, ln, (ox_km, oy_km) in [
        ("MV SAGAR MOTI", "419001234", "Bulk carrier", "India", 199, (-46.0, 38.0)),
        ("MV NORD SPICA", "219871200", "General cargo", "Denmark", 158, (52.0, -41.0)),
    ]:
        ox, oy = ox_km * 1000.0, oy_km * 1000.0
        outbound = (math.degrees(math.atan2(ox, oy)) + 360.0) % 360.0
        vessels.append(
            {
                "mmsi": mm, "name": nm, "type": tp, "flag": fl, "length_m": ln,
                "pings": _straight_track(
                    lat0, lon0, (rx + ox, ry + oy), outbound, rng.uniform(9.5, 13.5),
                    truth["valve_open_h_before"] + 8.0, 0.0, rng=rng),
                "truth": "innocent - lane traffic, never near the corridor",
            }
        )

    return {"vessels": vessels, "release_hour_estimate": round(mid_h, 1)}


def observation_tracks(scenario):
    """Copy only observed traffic fields across the simulation boundary."""
    import copy
    return {"vessels": [{k: copy.deepcopy(v[k]) for k in
             ("mmsi", "name", "type", "flag", "length_m", "pings")}
             for v in scenario["vessels"]]}


def load_tracks(*args, **kwargs):
    raise NotImplementedError(
        "Supply observed traffic to analyse_observation; simulation truth must not enter analysis")


def build_random_scenario(truth, lat0, lon0, seed=0, n_background=10):
    """Synthetic stress traffic; background courses/locations are independent of oil.

    The discharger follows the forward-simulator vessel. Competitors use a
    separately seeded regional traffic generator, never the predicted corridor.
    All tracks cover 30 hours, except explicitly generated observation gaps.
    """
    rng = np.random.default_rng(seed + 901)
    mid_h = (truth["valve_open_h_before"] + truth["valve_close_h_before"]) / 2
    rx, ry = geo.lonlat_to_xy(truth["true_release_lon"], truth["true_release_lat"], lat0, lon0)
    angle = math.radians(truth["course_deg"])
    speed = truth["speed_kn"] * .514444
    start = (rx + math.sin(angle)*speed*(mid_h-30)*3600,
             ry + math.cos(angle)*speed*(mid_h-30)*3600)
    vessels = [dict(mmsi="419087341", name="SIMULATED DISCHARGER", type="Crude oil tanker",
                    flag="Synthetic", length_m=183, truth="guilty - released the oil",
                    pings=_straight_track(lat0,lon0,start,truth["course_deg"],
                                          truth["speed_kn"],30,0,rng=rng))]
    types = ["Container ship", "Chemical tanker", "Crude oil tanker", "Fishing vessel"]
    for i in range(n_background):
        course = float(rng.uniform(0,360)); knots = float(rng.uniform(5,18))
        # Position at acquisition is distributed in an independent regional box.
        end_x,end_y = rng.uniform(-55000,55000,size=2)
        a = math.radians(course); velocity = knots*.514444
        start = (end_x-math.sin(a)*velocity*30*3600,
                 end_y-math.cos(a)*velocity*30*3600)
        dark = None
        if rng.random() < .35:
            older = float(rng.uniform(2,25)); dark=(older,older-float(rng.uniform(.8,3)))
        vessels.append(dict(mmsi=str(900000000+i),name="BACKGROUND %02d" % i,
            type=str(rng.choice(types)),flag="Synthetic",length_m=int(rng.integers(50,250)),
            truth="innocent - random regional traffic",
            pings=_straight_track(lat0,lon0,start,course,knots,30,0,dark=dark,rng=rng)))
    return {"vessels": vessels}
