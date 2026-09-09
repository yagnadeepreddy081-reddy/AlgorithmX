"""
Forward incident simulator -- ground truth for the end-to-end recovery test.

This is the piece that makes the demo an honest test rather than a circular one.

An earlier version of this prototype placed the "guilty" vessel using the output
of the backward drift model, which guarantees a hit and proves nothing.  Here
the causality runs the correct way:

    1. A vessel steams a course and opens its slop valve for a period ending
       several hours before the satellite pass.
    2. Oil parcels released along that track are advected FORWARD with the true
       wind and current field to the moment of acquisition.
    3. The drifted parcel cloud -- not a drawn ribbon -- is what gets rendered
       into the SAR scene as a damping footprint.

The detection, backward-drift and AIS stages then run with no knowledge of any
of this.  The vessel is recovered, or it is not, and either result is real.

The truth generator and the hindcast are deliberately *not* the same physics:
the truth uses one fixed wind-drift factor and one current realisation; the
hindcast runs a 12-member ensemble over a plausible range and does not know
which member is right.  That asymmetry is the whole point of the ensemble.
"""

import math
import numpy as np

import geo
from sar_sim import PIXEL_METRES

TRUE_ALPHA = 0.031      # the wind-drift factor reality used, unknown to the model
TRUE_DEFLECT = 14.0     # degrees, Coriolis deflection to the right
DT_FWD = 600.0          # 10 minute forward step
K_H_TRUE = 8.0          # true horizontal eddy diffusivity, m^2/s


def simulate_release(provider, size, seed=0,
                     release_start_h=9.0, release_end_h=6.0,
                     speed_kn=11.5, course_deg=None, n_parcels=2600):
    """
    Returns
    -------
    cov      float32 (size, size) oil coverage in [0, 1] at acquisition time
    truth    dict describing the release: vessel track, valve open/close times,
             true release centroid, true slick axis
    """
    rng = np.random.default_rng(seed)
    if course_deg is None:
        course_deg = rng.uniform(0, 360)
    v = speed_kn * 0.514444
    th = math.radians(course_deg)
    ux, uy = math.sin(th), math.cos(th)

    # Place the vessel so the drifted slick lands near the middle of the tile.
    # Position at t hours before acquisition, in local metres:
    #     P(t) = P_ref - (v * t * 3600) * heading
    mid_h = 0.5 * (release_start_h + release_end_h)
    p_ref = np.array([0.0, 0.0])  # vessel position at valve mid-point

    def vessel_xy(t_h):
        dt = (mid_h - t_h) * 3600.0
        return p_ref + np.array([ux, uy]) * v * dt

    # --- release parcels along the track while the valve is open -----------
    rel_times = np.linspace(release_start_h, release_end_h, 90)
    per = max(1, n_parcels // len(rel_times))
    px, py, pt = [], [], []
    for t_h in rel_times:
        base = vessel_xy(t_h)
        # initial cross-track spread: the width of the ship's wake
        for _ in range(per):
            off = rng.normal(0.0, 60.0)
            px.append(base[0] - uy * off + rng.normal(0, 25))
            py.append(base[1] + ux * off + rng.normal(0, 25))
            pt.append(t_h)
    px = np.array(px)
    py = np.array(py)
    pt = np.array(pt)

    # --- forward advection to acquisition time ----------------------------
    ca, sa = math.cos(math.radians(TRUE_DEFLECT)), math.sin(math.radians(TRUE_DEFLECT))
    diff_sigma = math.sqrt(2.0 * K_H_TRUE * DT_FWD)
    t_now = -release_start_h * 3600.0
    end = 0.0
    alive = pt <= release_start_h
    while t_now < end - 1e-6:
        step = min(DT_FWD, end - t_now)
        hours_before = -t_now / 3600.0
        active = pt >= hours_before  # parcel already released at this moment
        idx = np.nonzero(active)[0]
        if len(idx):
            for i in idx:
                cu, cv = provider.current_at(px[i], py[i], t_now)
                wu, wv = provider.wind_at(px[i], py[i], t_now)
                dwu = TRUE_ALPHA * (wu * ca + wv * sa)
                dwv = TRUE_ALPHA * (-wu * sa + wv * ca)
                px[i] += (cu + dwu) * step
                py[i] += (cv + dwv) * step
            px[idx] += rng.normal(0, diff_sigma, len(idx))
            py[idx] += rng.normal(0, diff_sigma, len(idx))
        t_now += step
    del alive

    # --- rasterise the parcel cloud into a coverage map -------------------
    cov = np.zeros((size, size), dtype=np.float32)
    cx_px = px / PIXEL_METRES + size / 2.0
    cy_px = size / 2.0 - py / PIXEL_METRES
    ok = (cx_px > 1) & (cx_px < size - 2) & (cy_px > 1) & (cy_px < size - 2)
    ii = np.clip(cy_px[ok].astype(int), 0, size - 1)
    jj = np.clip(cx_px[ok].astype(int), 0, size - 1)
    np.add.at(cov, (ii, jj), 1.0)

    # each parcel represents a patch of film, not a point
    from scipy import ndimage
    cov = ndimage.gaussian_filter(cov, sigma=1.8)
    if cov.max() > 0:
        cov = cov / np.percentile(cov[cov > 0], 68)
    cov = np.clip(cov, 0.0, 1.0)
    # thickness patchiness -- real slicks break into windrows
    from sar_sim import _smooth_noise
    cov *= np.clip(0.72 + 0.42 * _smooth_noise(rng, cov.shape, 7.0), 0.0, 1.25)
    cov = np.clip(cov, 0.0, 1.0)

    # --- truth record ------------------------------------------------------
    track_hours = np.arange(0.0, release_start_h + 8.01, 0.25)
    track = []
    for t_h in track_hours:
        p = vessel_xy(t_h)
        lon, lat = geo.xy_to_lonlat(p[0], p[1], provider_lat0, provider_lon0)
        track.append({"t_h": float(t_h), "lat": round(lat, 6), "lon": round(lon, 6)})

    rel_mid = vessel_xy(mid_h)
    lon_r, lat_r = geo.xy_to_lonlat(rel_mid[0], rel_mid[1], provider_lat0, provider_lon0)
    truth = {
        "course_deg": round(float(course_deg), 1),
        "speed_kn": speed_kn,
        "valve_open_h_before": release_start_h,
        "valve_close_h_before": release_end_h,
        "release_duration_h": round(release_start_h - release_end_h, 2),
        "true_release_lat": round(lat_r, 6),
        "true_release_lon": round(lon_r, 6),
        "true_alpha": TRUE_ALPHA,
        "true_deflection_deg": TRUE_DEFLECT,
        "vessel_track": track,
        "track_length_km": round(
            (release_start_h - release_end_h) * speed_kn * 1.852, 1
        ),
    }
    return cov, truth


# The projection origin is injected by the caller before use; keeping it module
# level avoids threading it through every helper in this small module.
provider_lat0 = 18.95
provider_lon0 = 71.85


def set_origin(lat0, lon0):
    global provider_lat0, provider_lon0
    provider_lat0, provider_lon0 = lat0, lon0
