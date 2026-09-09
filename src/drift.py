"""
Stage 4 -- reverse (backward) Lagrangian drift.

The single most important idea in this system: the oil is not where it was
released.  At a typical 0.3-0.5 m/s total drift, oil seen at 06:00 was released
10-18 km away from where the satellite sees it.  Searching AIS around the slick
looks for ships in the wrong place.  So we run the ocean backwards.

Physics
-------
Surface oil advects with

    U_oil = U_current + alpha * U_wind

where alpha is the wind-drift factor.  The operational value for surface oil is
3 % of the 10 m wind (Lehr et al.; used by NOAA GNOME and OpenDrift), with a
small Coriolis-driven deflection to the right in the northern hemisphere.
Sub-grid turbulence is added as a random walk with horizontal eddy diffusivity
K_h, giving a spreading standard deviation of sqrt(2 K_h t).

Integration is second-order Runge-Kutta (midpoint) at a 15-minute step, run
with a negative time increment.

Ensemble
--------
A single backward trajectory is a lie -- it hides the fact that alpha, the
current field and the wind are all uncertain.  We run N members with

    alpha        ~ U(0.020, 0.045)
    deflection   ~ U(-10, 35) deg
    current gain ~ N(1.0, 0.30)   magnitude
    current turn ~ N(0, 20) deg   direction
    wind gain    ~ N(1.0, 0.22)

and report the *envelope* of where the particles came from, not a track.

Honesty gate
------------
The envelope grows without limit as you go further back.  Beyond a radius where
the 90th-percentile spread exceeds `MAX_USEFUL_RADIUS_KM`, an AIS match is not
evidence, it is coincidence.  Those hours are computed, reported, and then
explicitly excluded from scoring.
"""

import math
import numpy as np

import geo
from sar_sim import PIXEL_METRES

WIND_DRIFT_ALPHA = 0.030      # 3 % of 10 m wind, standard for surface oil
K_H = 25.0                    # horizontal eddy diffusivity, m^2/s
DT = 900.0                    # 15 minute integration step
MAX_USEFUL_RADIUS_KM = 12.0   # beyond this the envelope stops being evidence


def seed_particles(det, lat0, lon0, size, n_particles=900, rng=None):
    """Seed particles uniformly over the detected slick footprint."""
    rng = rng or np.random.default_rng(0)
    ys, xs = det["_pixels"]
    idx = rng.integers(0, len(ys), size=n_particles)
    # jitter within the pixel so particles are not lattice-locked
    py = ys[idx] + rng.uniform(-0.5, 0.5, n_particles)
    px = xs[idx] + rng.uniform(-0.5, 0.5, n_particles)
    x_m = (px - size / 2.0) * PIXEL_METRES
    y_m = (size / 2.0 - py) * PIXEL_METRES
    return x_m, y_m


def backward_drift(det, provider, lat0, lon0, size, hours_back=18,
                   n_particles=900, n_members=12, seed=0):
    """
    Run the ensemble backwards.

    Returns a dict with, for every whole hour back:
        centroid lon/lat, spread radii (km) at the 50th/90th percentile,
        an ellipse polygon for the map, and a usable flag.
    """
    rng = np.random.default_rng(seed)
    x0, y0 = seed_particles(det, lat0, lon0, size, n_particles, rng)

    members = []
    for m in range(n_members):
        members.append(
            {
                # Spread chosen so the ensemble's position error grows at
                # roughly 1 km/h, which is what operational surface-drift
                # hindcasts actually achieve.  A tighter ensemble would look
                # better on screen and would be a lie: it would claim the
                # release point to a precision no drift model has.
                "alpha": rng.uniform(0.020, 0.045),
                "deflect": math.radians(rng.uniform(-10.0, 35.0)),
                "cgain": max(0.25, rng.normal(1.0, 0.30)),
                "wgain": max(0.25, rng.normal(1.0, 0.22)),
                "crot": math.radians(rng.normal(0.0, 20.0)),
            }
        )

    per_member = n_particles // n_members
    r90_zero = 0.0
    x = np.concatenate([x0[i * per_member : (i + 1) * per_member] for i in range(n_members)])
    y = np.concatenate([y0[i * per_member : (i + 1) * per_member] for i in range(n_members)])
    mem_id = np.concatenate([np.full(per_member, i) for i in range(n_members)])

    alpha = np.array([members[i]["alpha"] for i in mem_id])
    defl = np.array([members[i]["deflect"] for i in mem_id])
    cg = np.array([members[i]["cgain"] for i in mem_id])
    crot = np.array([members[i]["crot"] for i in mem_id])
    wg = np.array([members[i]["wgain"] for i in mem_id])

    def velocity(xa, ya, t):
        ux = np.empty_like(xa)
        uy = np.empty_like(ya)
        for i in range(len(xa)):
            cu, cv = provider.current_at(xa[i], ya[i], t)
            wu, wv = provider.wind_at(xa[i], ya[i], t)
            # wind-driven component, rotated right by the deflection angle
            ca, sa = math.cos(defl[i]), math.sin(defl[i])
            dwu = alpha[i] * wg[i] * (wu * ca + wv * sa)
            dwv = alpha[i] * wg[i] * (-wu * sa + wv * ca)
            # the current field itself is uncertain in direction, not
            # only in magnitude
            cr, sr = math.cos(crot[i]), math.sin(crot[i])
            cur_u = cg[i] * (cu * cr - cv * sr)
            cur_v = cg[i] * (cu * sr + cv * cr)
            ux[i] = cur_u + dwu
            uy[i] = cur_v + dwv
        return ux, uy

    steps_per_hour = int(3600 / DT)
    diff_sigma = math.sqrt(2.0 * K_H * DT)

    frames = []
    t = 0.0
    for h in range(hours_back + 1):
        if h > 0:
            for _ in range(steps_per_hour):
                ux, uy = velocity(x, y, t)
                xm = x - ux * DT / 2.0
                ym = y - uy * DT / 2.0
                ux2, uy2 = velocity(xm, ym, t - DT / 2.0)
                x = x - ux2 * DT + rng.normal(0.0, diff_sigma, len(x))
                y = y - uy2 * DT + rng.normal(0.0, diff_sigma, len(y))
                t -= DT

        cx, cy = float(np.mean(x)), float(np.mean(y))
        d = np.hypot(x - cx, y - cy)
        r50 = float(np.percentile(d, 50)) / 1000.0
        r90 = float(np.percentile(d, 90)) / 1000.0
        lon_c, lat_c = geo.xy_to_lonlat(cx, cy, lat0, lon0)
        # The cloud is already several km across at hour zero, because that is
        # the physical size of the slick.  Drift *uncertainty* is the spread
        # ADDED by running the model backwards -- subtract the slick's own
        # extent, or the honesty gate ends up measuring the slick, not the model.
        if h == 0:
            r90_zero = r90
        added_km = max(0.0, r90 - r90_zero)

        # covariance ellipse of the particle cloud, at the 90 % level
        cov = np.cov(np.vstack([x - cx, y - cy]))
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1.0)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        k = 2.146  # chi-square 2 dof, 90 %
        ell = []
        for th in np.linspace(0, 2 * math.pi, 40):
            p = vecs @ (k * np.sqrt(vals) * np.array([math.cos(th), math.sin(th)]))
            lo, la = geo.xy_to_lonlat(cx + p[0], cy + p[1], lat0, lon0)
            ell.append([round(la, 6), round(lo, 6)])

        frames.append(
            {
                "hours_back": h,
                "lat": round(lat_c, 6),
                "lon": round(lon_c, 6),
                "r50_km": round(r50, 2),
                "r90_km": round(r90, 2),
                "uncertainty_km": round(added_km, 2),
                "usable": bool(added_km <= MAX_USEFUL_RADIUS_KM),
                "ellipse": ell,
                # full parcel cloud in local metres, kept for the corridor test
                # and stripped before serialisation
                "_xy": (x.copy(), y.copy()),
                "particles": [
                    [round(geo.xy_to_lonlat(x[i], y[i], lat0, lon0)[1], 5),
                     round(geo.xy_to_lonlat(x[i], y[i], lat0, lon0)[0], 5)]
                    for i in range(0, len(x), max(1, len(x) // 120))
                ],
            }
        )

    usable = [f for f in frames if f["usable"]]
    return {
        "frames": frames,
        "hours_computed": hours_back,
        "hours_usable": len(usable) - 1 if usable else 0,
        "max_useful_radius_km": MAX_USEFUL_RADIUS_KM,
        "wind_drift_alpha_range": [0.020, 0.045],
        "eddy_diffusivity_m2s": K_H,
        "ensemble_members": n_members,
        "particles": n_particles,
        "field_source": provider.describe(),
    }
