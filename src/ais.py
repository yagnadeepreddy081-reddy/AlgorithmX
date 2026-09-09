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
   onto that grid by great-circle interpolation, so "was this vessel inside the
   release envelope at 03:00" becomes an answerable question.

2. Dark-period detection.  A gap longer than DARK_GAP_MIN minutes while the
   vessel is at sea is recorded as a dark period.  This is treated as evidence
   *for* suspicion rather than as missing data -- switching the transponder off
   is exactly what a vessel about to discharge does.  Positions inside a dark
   period are marked interpolated and are never presented as observed fact.
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
                "observed": bool(gap_min <= DARK_GAP_MIN),
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


def build_scenario(det, drift_result, lat0, lon0, seed=0):
    """
    Construct the AIS picture for the incident.

    Deliberately includes four hard cases, because a demo that only contains an
    obvious culprit proves nothing:

      * GUILTY   -- transited the release envelope during the release window on
                    a course aligned with the slick's principal axis
      * DECOY    -- passed *closer* to the slick than the guilty vessel, but
                    hours after the release window, so it is physically
                    impossible.  The system must clear it.
      * DARK     -- went dark inside the window but was on the wrong side of the
                    envelope.  Suspicion is raised but distance rules it out.
      * TRANSIT  -- ordinary lane traffic, never near the envelope.
    """
    rng = np.random.default_rng(seed)
    frames = {f["hours_back"]: f for f in drift_result["frames"]}
    usable = [f for f in drift_result["frames"] if f["usable"]]
    # release window: the usable part of the hindcast, older half
    h_max = max(f["hours_back"] for f in usable) if usable else 6
    h_rel = max(3, int(h_max * 0.75))
    origin = frames[h_rel]
    ox, oy = geo.lonlat_to_xy(origin["lon"], origin["lat"], lat0, lon0)

    slick_axis = det["orientation_deg"]
    vessels = []

    # --- guilty: on the slick axis, through the origin at h_rel -------------
    course = (slick_axis + rng.uniform(-6, 6)) % 360
    spd = 12.4
    v = spd * 0.514444
    t_from = h_rel + 7.0
    dx = -v * math.sin(math.radians(course)) * 7.0 * 3600.0
    dy = -v * math.cos(math.radians(course)) * 7.0 * 3600.0
    vessels.append(
        {
            "mmsi": "419087341",
            "name": "MT ARJUN PRIDE",
            "type": "Crude oil tanker",
            "flag": "India",
            "length_m": 183,
            "pings": _straight_track(lat0, lon0, (ox + dx, oy + dy), course, spd, t_from, 0.0, rng=rng),
            "truth": "guilty",
        }
    )

    # --- decoy: closest to the slick, but arrives after the release ---------
    slick_x, slick_y = geo.lonlat_to_xy(det["centroid"][1], det["centroid"][0], lat0, lon0)
    course_d = (slick_axis + 78.0) % 360
    spd_d = 15.5
    vd = spd_d * 0.514444
    t_from_d = 2.5
    dxd = -vd * math.sin(math.radians(course_d)) * 2.5 * 3600.0
    dyd = -vd * math.cos(math.radians(course_d)) * 2.5 * 3600.0
    vessels.append(
        {
            "mmsi": "477553900",
            "name": "MV HAI FENG 09",
            "type": "Container ship",
            "flag": "Hong Kong",
            "length_m": 232,
            "pings": _straight_track(lat0, lon0, (slick_x + dxd, slick_y + dyd), course_d, spd_d, t_from_d, 0.0, rng=rng),
            "truth": "innocent (arrived after release window)",
        }
    )

    # --- dark-period vessel, wrong side of the envelope --------------------
    course_k = (slick_axis + 150.0) % 360
    spd_k = 10.8
    vessels.append(
        {
            "mmsi": "538007712",
            "name": "MT GULF SENTINEL",
            "type": "Chemical tanker",
            "flag": "Marshall Islands",
            "length_m": 176,
            "pings": _straight_track(
                lat0, lon0, (ox + 34000, oy - 27000), course_k, spd_k, h_rel + 6.0, 0.0,
                dark=(h_rel + 1.5, h_rel - 1.5), rng=rng,
            ),
            "truth": "innocent (dark period, but 30+ km away)",
        }
    )

    # --- ordinary lane traffic ---------------------------------------------
    for i, (nm, mm, tp, fl, ln) in enumerate(
        [
            ("MV SAGAR MOTI", "419001234", "Bulk carrier", "India", 199),
            ("MV NORD SPICA", "219871200", "General cargo", "Denmark", 158),
        ]
    ):
        vessels.append(
            {
                "mmsi": mm,
                "name": nm,
                "type": tp,
                "flag": fl,
                "length_m": ln,
                "pings": _straight_track(
                    lat0, lon0,
                    (ox + rng.uniform(-70000, -45000), oy + rng.uniform(35000, 60000)),
                    rng.uniform(0, 360), rng.uniform(9, 14), h_rel + 8.0, 0.0, rng=rng,
                ),
                "truth": "innocent (lane traffic)",
            }
        )

    return {"vessels": vessels, "release_hour_estimate": h_rel}


def load_tracks(det, drift_result, lat0, lon0, source="offline", seed=0, truth=None):
    """Single entry point. `source='gfw'` would query Global Fishing Watch."""
    if source != "offline":
        raise NotImplementedError("live AIS feed not wired in the offline prototype")
    if truth is not None:
        return build_truth_scenario(truth, det, lat0, lon0, seed=seed)
    return build_scenario(det, drift_result, lat0, lon0, seed=seed)
