"""
SAMUDRA NETRA -- Sentinel-1 SAR scene synthesiser.

Purpose
-------
Generates physically-grounded synthetic Sentinel-1 IW/GRD-like C-band VV scenes
containing oil slicks, three classes of oil look-alike, ships and land, together
with pixel-level ground-truth masks.

Why this exists
---------------
The Krestenitis et al. (2019) labelled benchmark is the correct training set for
this task, but it is distributed behind a request form.  Rather than block the
whole pipeline on a download, we synthesise scenes from published C-band ocean
radar physics.  `datasets.py` exposes the same interface for both sources, so
the real benchmark drops in without touching the model or training code.

Physics implemented
-------------------
1. Ocean normalised radar cross-section (NRCS) as a function of local wind
   speed, following the CMOD-family power-law behaviour at VV polarisation and
   ~35 deg incidence.  Wind is spatially varying (streaks + gusts), not constant.
2. Multiplicative speckle drawn from a Gamma distribution with shape equal to
   the equivalent number of looks (Sentinel-1 GRD ~ 4.4 looks).
3. Oil films damp short gravity-capillary waves (Marangoni damping), reducing
   NRCS by a published 3-10 dB.  Slicks are rendered as elongated ribbons along
   a vessel course with a broadening, feathering tail -- the signature of a
   discharge released continuously while underway.
4. Look-alikes reproduce the three operational false-positive families:
      - low-wind / wind-shadow zones : weak damping, very diffuse edges, round
      - biogenic / algal films       : weak damping, fractal edges, no course
      - rain cells                   : dark core with a bright convective rim
   Each is given a *weaker and blurrier* damping edge than oil, which is the
   real discriminative cue an analyst uses.
5. Ships appear as bright point targets with a cross-shaped sidelobe response
   and an occasional turbulent wake.

Class codes (identical to the Krestenitis benchmark ordering)
    0 sea   1 oil spill   2 look-alike   3 ship   4 land
"""

import numpy as np

SEA, OIL, LOOKALIKE, SHIP, LAND = 0, 1, 2, 3, 4
N_CLASSES = 5

# Ground sampling distance of the tiles we segment on.  Sentinel-1 GRD is 10 m
# for IW; slick-scale segmentation is done on 8x multilooked tiles.
PIXEL_METRES = 80.0


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _smooth_noise(rng, shape, scale):
    """Low-frequency random field, used for wind streaks and texture."""
    small = max(2, int(min(shape) / scale))
    coarse = rng.normal(0.0, 1.0, size=(small, small))
    yi = np.linspace(0, small - 1, shape[0])
    xi = np.linspace(0, small - 1, shape[1])
    y0 = np.clip(np.floor(yi).astype(int), 0, small - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, small - 1)
    y1 = np.clip(y0 + 1, 0, small - 1)
    x1 = np.clip(x0 + 1, 0, small - 1)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    top = coarse[np.ix_(y0, x0)] * (1 - fx) + coarse[np.ix_(y0, x1)] * fx
    bot = coarse[np.ix_(y1, x0)] * (1 - fx) + coarse[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


def _box_blur(a, radius):
    """Separable box blur via cumulative sums; cheap and dependency-free."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    pad = np.pad(a, radius, mode="edge")
    cs = np.cumsum(pad, axis=0)
    a1 = np.empty_like(pad)
    a1[:k] = cs[:k]
    a1[k:] = cs[k:] - cs[:-k]
    a1 = a1[: a.shape[0]] / k
    cs = np.cumsum(a1, axis=1)
    a2 = np.empty_like(a1)
    a2[:, :k] = cs[:, :k]
    a2[:, k:] = cs[:, k:] - cs[:, :-k]
    return a2[:, : a.shape[1]] / k


def _disc(shape, cy, cx, ry, rx, angle):
    """Elliptical distance field, values < 1 are inside the ellipse."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    ca, sa = np.cos(angle), np.sin(angle)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    return np.sqrt((u / max(rx, 1e-3)) ** 2 + (v / max(ry, 1e-3)) ** 2)


# ----------------------------------------------------------------------------
# ocean background
# ----------------------------------------------------------------------------

def ocean_nrcs_db(wind_ms):
    """
    C-band VV NRCS at ~35 deg incidence as a function of 10 m wind speed.

    Power-law fit to CMOD5.N behaviour over the 2-15 m/s range: sigma0 rises
    roughly as the 1.6th power of wind speed.  Returns dB.
    """
    wind = np.clip(wind_ms, 0.4, 25.0)
    return 10.0 * 1.60 * np.log10(wind) - 26.5


def _wind_field(rng, shape, mean_wind):
    """Spatially varying wind: large-scale gradient + streaks + gusts."""
    streaks = _smooth_noise(rng, shape, scale=18.0) * 0.9
    gusts = _smooth_noise(rng, shape, scale=6.0) * 0.45
    field = mean_wind + streaks + gusts
    return np.clip(field, 0.6, 24.0)


# ----------------------------------------------------------------------------
# feature painters -- each returns a soft coverage map in [0, 1]
# ----------------------------------------------------------------------------

def _paint_slick(rng, shape, cy, cx, course_rad, length_px, width_px):
    """
    Oil discharge ribbon: long, thin, aligned to the vessel course, broadening
    and feathering toward the older (tail) end, with a slight course drift.
    Edges are sharp -- this is the cue that separates oil from look-alikes.
    """
    h, w = shape
    cov = np.zeros(shape, dtype=np.float32)
    n = max(24, int(length_px))
    # slight curvature so the ribbon is not a perfect straight line
    curve = rng.uniform(-0.35, 0.35)
    t = np.linspace(0.0, 1.0, n)
    ang = course_rad + curve * (t - 0.5)
    step = length_px / n
    ys = cy + np.cumsum(np.sin(ang) * step) - np.sin(course_rad) * length_px / 2
    xs = cx + np.cumsum(np.cos(ang) * step) - np.cos(course_rad) * length_px / 2
    # tail broadens: the oldest oil has spread the most
    widths = width_px * (0.55 + 1.05 * t ** 1.4)
    # patchiness along the ribbon -- real slicks are not uniform
    patch = 0.72 + 0.28 * np.abs(_smooth_noise(rng, (1, n), 3.0)[0])
    patch = np.clip(patch, 0.35, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        r = widths[i]
        if r < 0.5:
            continue
        y0, y1 = int(max(0, ys[i] - 3 * r)), int(min(h, ys[i] + 3 * r + 1))
        x0, x1 = int(max(0, xs[i] - 3 * r)), int(min(w, xs[i] + 3 * r + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        d2 = (yy[y0:y1, x0:x1] - ys[i]) ** 2 + (xx[y0:y1, x0:x1] - xs[i]) ** 2
        blob = np.exp(-d2 / (2.0 * (r * 0.62) ** 2)) * patch[i]
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], blob)
    # sharpen: oil/water boundary is a step in damping, not a gradient
    cov = np.clip((cov - 0.20) / 0.30, 0.0, 1.0)
    # ragged edge from surface turbulence
    cov *= np.clip(0.80 + 0.30 * _smooth_noise(rng, shape, 5.0), 0.0, 1.2)
    return np.clip(cov, 0.0, 1.0)


def _paint_drifted_slick(rng, shape, cy, cx, course_rad, length_px, width_px):
    """
    Oil ribbon rendered the way a real one forms: parcels released along the
    vessel's track while the valve is open, then displaced by however long each
    parcel has been on the water.

    The difference from `_paint_slick` matters.  A drawn ribbon is uniform and
    straight; a drifted one is skewed, because the oil released first has been
    drifting longest.  The head and the tail of a real slick sit on *different*
    sides of the vessel's track, and the whole ribbon is sheared away from the
    course line.  Training on drawn ribbons alone produces a model that misses
    real ones, which is exactly what happened the first time we tried it.
    """
    h, w = shape
    n_rel = 70
    per = 26
    # drift over the age of the slick, in pixels
    drift_dir = rng.uniform(0, 2 * np.pi)
    drift_px = rng.uniform(0.10, 0.55) * length_px
    shear = rng.uniform(-0.35, 0.35)

    t = rng.random(n_rel * per)          # 0 = youngest parcel, 1 = oldest
    along = (np.repeat(np.linspace(-0.5, 0.5, n_rel), per) + rng.normal(0, 0.006, n_rel * per))
    px_ = cx + np.cos(course_rad) * along * length_px
    py_ = cy + np.sin(course_rad) * along * length_px
    age = (along + 0.5)                  # oldest at the tail end
    dd = drift_dir + shear * (age - 0.5)
    px_ = px_ + np.cos(dd) * drift_px * age + rng.normal(0, width_px * 0.55, len(px_))
    py_ = py_ + np.sin(dd) * drift_px * age + rng.normal(0, width_px * 0.55, len(py_))

    cov = np.zeros(shape, dtype=np.float32)
    ii = np.clip(py_.astype(int), 0, h - 1)
    jj = np.clip(px_.astype(int), 0, w - 1)
    ok = (py_ > 0) & (py_ < h - 1) & (px_ > 0) & (px_ < w - 1)
    np.add.at(cov, (ii[ok], jj[ok]), 1.0)
    from scipy import ndimage as _nd
    cov = _nd.gaussian_filter(cov, sigma=max(1.0, width_px * 0.45))
    if cov.max() > 0:
        cov = cov / max(np.percentile(cov[cov > 0], 62), 1e-6)
    cov = np.clip(cov, 0.0, 1.0)
    cov *= np.clip(0.70 + 0.45 * _smooth_noise(rng, shape, 7.0), 0.0, 1.3)
    del t
    return np.clip(cov, 0.0, 1.0)


def _paint_low_wind(rng, shape, cy, cx):
    """Wind-shadow / low-wind zone: large, round, very diffuse edges."""
    ry = rng.uniform(16, 40)
    rx = ry * rng.uniform(0.85, 1.6)
    d = _disc(shape, cy, cx, ry, rx, rng.uniform(0, np.pi))
    cov = np.clip(1.35 - d, 0.0, 1.0)
    cov *= np.clip(0.75 + 0.45 * _smooth_noise(rng, shape, 10.0), 0.0, 1.3)
    return np.clip(_box_blur(cov, 4), 0.0, 1.0)


def _paint_biogenic(rng, shape, cy, cx):
    """Algal / biogenic film: irregular fractal edge, no preferred course."""
    ry = rng.uniform(12, 30)
    rx = ry * rng.uniform(0.7, 1.5)
    d = _disc(shape, cy, cx, ry, rx, rng.uniform(0, np.pi))
    lobes = 0.55 * _smooth_noise(rng, shape, 4.0)
    cov = np.clip(1.25 - d + lobes, 0.0, 1.0)
    return np.clip(_box_blur(cov, 2), 0.0, 1.0)


def _paint_rain_cell(rng, shape, cy, cx):
    """Rain cell: damped core, bright convective rim. Returns (core, rim)."""
    r = rng.uniform(14, 30)
    d = _disc(shape, cy, cx, r, r * rng.uniform(0.85, 1.2), 0.0)
    core = np.clip(1.15 - d, 0.0, 1.0)
    rim = np.clip(1.0 - np.abs(d - 1.05) / 0.28, 0.0, 1.0) * 0.9
    return _box_blur(core, 3), _box_blur(rim, 2)


# ----------------------------------------------------------------------------
# scene generation
# ----------------------------------------------------------------------------

def generate_scene(seed, size=256, force_oil=None, oil_cov=None, mean_wind=None):
    """
    Returns
    -------
    img   : float32 (size, size) in [0, 1], dB-scaled and stretched like a
            displayed Sentinel-1 GRD quicklook
    mask  : uint8  (size, size) with class codes
    meta  : dict of scene truth (wind, slick course, slick length, ...)
    """
    rng = np.random.default_rng(seed)
    shape = (size, size)

    mean_wind = float(rng.uniform(2.2, 13.5) if mean_wind is None else mean_wind)
    if not np.isfinite(mean_wind) or mean_wind <= 0:
        raise ValueError("mean_wind must be finite and positive")
    wind = _wind_field(rng, shape, mean_wind)
    nrcs_db = ocean_nrcs_db(wind)

    mask = np.zeros(shape, dtype=np.uint8)
    meta = {
        "wind_ms": float(mean_wind),
        "has_oil": False,
        "slick_course_deg": None,
        "slick_length_km": None,
        "n_lookalikes": 0,
        "n_ships": 0,
    }

    # --- land (coastline intruding from one edge) --------------------------
    if rng.random() < 0.22:
        edge = rng.integers(0, 4)
        depth = rng.integers(size // 9, size // 4)
        coast = depth + (_smooth_noise(rng, (1, size), 5.0)[0] * depth * 0.45)
        land = np.zeros(shape, dtype=bool)
        idx = np.arange(size)
        for j in range(size):
            c = int(np.clip(coast[j], 4, size - 4))
            if edge == 0:
                land[:c, j] = True
            elif edge == 1:
                land[size - c :, j] = True
            elif edge == 2:
                land[j, :c] = True
            else:
                land[j, size - c :] = True
        # land backscatter is brighter and much rougher than the sea
        nrcs_db = np.where(land, rng.uniform(-9.0, -5.0) + 2.4 * _smooth_noise(rng, shape, 3.0), nrcs_db)
        mask[land] = LAND
        del idx

    sea = mask == SEA

    # --- oil slick ---------------------------------------------------------
    want_oil = (rng.random() < 0.55 if force_oil is None else force_oil) or oil_cov is not None
    if want_oil:
        if oil_cov is not None:
            # Footprint supplied by the forward incident simulator: a real
            # drifted parcel cloud, not a drawn shape.
            cov = oil_cov.astype(np.float32)
            course = 0.0
            length = 0.0
        else:
            course = rng.uniform(0, 2 * np.pi)
            # Ribbons are allowed to run off the edge of the tile.  Real slicks
            # are 10-60 km long and a 20 km tile only ever shows part of one, so
            # a training set where every slick fits inside the frame teaches the
            # model something false about what a slick looks like.
            length = rng.uniform(0.35, 1.70) * size
            width = rng.uniform(0.010, 0.040) * size
            cy = rng.uniform(0.20, 0.80) * size
            cx = rng.uniform(0.20, 0.80) * size
            if rng.random() < 0.55:
                cov = _paint_drifted_slick(rng, shape, cy, cx, course, length, width)
            else:
                cov = _paint_slick(rng, shape, cy, cx, course, length, width)
        # Marangoni damping: 3-10 dB reduction, published range for mineral oil
        damping = rng.uniform(3.2, 9.5)
        # damping collapses in very low wind (no capillary waves to damp) and
        # in very high wind (the film is broken up) -- the physical validity band
        eff = np.clip((wind - 2.0) / 2.0, 0.0, 1.0) * np.clip((15.0 - wind) / 4.0, 0.0, 1.0)
        nrcs_db = nrcs_db - cov * damping * eff
        # An analyst can only label what the radar could actually record.  In
        # winds outside the validity band the film leaves no contrast, so the
        # scene is labelled sea -- this teaches the model the physical limit
        # rather than teaching it to hallucinate invisible oil.
        detectable = (cov * damping * eff) > 1.6
        oil_px = (cov > 0.42) & sea & detectable
        mask[oil_px] = OIL
        if oil_px.sum() > 40:
            meta["has_oil"] = True
            if oil_cov is None:
                meta["slick_course_deg"] = float((np.degrees(course)) % 180.0)
                meta["slick_length_km"] = float(length * PIXEL_METRES / 1000.0)

    # --- look-alikes -------------------------------------------------------
    for _ in range(rng.integers(0, 3)):
        kind = rng.integers(0, 3)
        cy = rng.uniform(0.15, 0.85) * size
        cx = rng.uniform(0.15, 0.85) * size
        if kind == 0:
            cov = _paint_low_wind(rng, shape, cy, cx)
            nrcs_db = nrcs_db - cov * rng.uniform(1.6, 4.0)
        elif kind == 1:
            cov = _paint_biogenic(rng, shape, cy, cx)
            nrcs_db = nrcs_db - cov * rng.uniform(1.8, 4.6)
        else:
            cov, rim = _paint_rain_cell(rng, shape, cy, cx)
            nrcs_db = nrcs_db - cov * rng.uniform(1.5, 3.8) + rim * rng.uniform(1.0, 2.4)
        la = (cov > 0.45) & (mask == SEA)
        if la.sum() > 30:
            mask[la] = LOOKALIKE
            meta["n_lookalikes"] += 1

    # --- ships -------------------------------------------------------------
    for _ in range(rng.integers(0, 5)):
        sy = int(rng.uniform(0.06, 0.94) * size)
        sx = int(rng.uniform(0.06, 0.94) * size)
        if mask[sy, sx] == LAND:
            continue
        ln = int(rng.integers(1, 4))
        nrcs_db[max(0, sy - ln) : sy + ln + 1, max(0, sx - ln) : sx + ln + 1] += rng.uniform(14.0, 22.0)
        # azimuth/range sidelobe cross, a real SAR artefact around bright targets
        nrcs_db[max(0, sy - 3 * ln) : sy + 3 * ln + 1, sx] += 4.0
        nrcs_db[sy, max(0, sx - 3 * ln) : sx + 3 * ln + 1] += 4.0
        mask[max(0, sy - ln) : sy + ln + 1, max(0, sx - ln) : sx + ln + 1] = SHIP
        meta["n_ships"] += 1

    # --- speckle -----------------------------------------------------------
    sigma0 = 10.0 ** (nrcs_db / 10.0)
    looks = 4.4  # Sentinel-1 GRD equivalent number of looks
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=shape)
    intensity = sigma0 * speckle

    # --- dB stretch to an 8-bit-like display product -----------------------
    db = 10.0 * np.log10(np.maximum(intensity, 1e-7))
    lo, hi = -30.0, -3.0
    img = np.clip((db - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return img, mask, meta


def generate_batch(n, seed0=0, size=256):
    imgs = np.zeros((n, size, size), dtype=np.float32)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    metas = []
    for i in range(n):
        im, mk, mt = generate_scene(seed0 + i, size=size)
        imgs[i], masks[i] = im, mk
        metas.append(mt)
    return imgs, masks, metas


if __name__ == "__main__":
    im, mk, mt = generate_scene(7)
    print("image", im.shape, im.dtype, round(float(im.min()), 3), round(float(im.max()), 3))
    print("mask classes", np.unique(mk), "meta", mt)
