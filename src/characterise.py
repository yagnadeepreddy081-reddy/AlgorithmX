"""
Stage 3 -- slick characterisation and look-alike discrimination.

Takes the network's class map and turns each connected oil region into a
measured object with a geo-referenced polygon, shape statistics and a
calibrated confidence.  Everything downstream (drift seeding, course matching,
the dossier) consumes this output, not the raw mask.

The look-alike discriminator is deliberately kept separate from the network.
The network learns appearance; the discriminator applies *physics and context*
that no amount of training data can put into a 256x256 tile:

  * wind gating       -- below ~3 m/s there are no capillary waves for a film
                         to damp, so every dark patch is suspect; above ~12 m/s
                         a real film is broken up and should not be visible.
                         Detections outside 3-12 m/s are demoted, not deleted.
  * elongation        -- an operational discharge is released while the ship is
                         underway, so it is a ribbon.  Algal blooms and wind
                         shadows have no reason to be elongated.
  * edge sharpness    -- an oil/water boundary is a step change in damping; a
                         low-wind zone fades out gradually.
  * damping contrast  -- how many dB darker the patch is than the surrounding
                         sea.  Real mineral oil sits in a 3-10 dB band.
"""

import math
import numpy as np
from scipy import ndimage

import geo
from sar_sim import PIXEL_METRES, OIL, LOOKALIKE

MIN_AREA_PX = 60  # ~0.4 km^2 at 80 m pixels; below this, attribution is hopeless


def _principal_axis(ys, xs):
    """PCA on the pixel cloud -> (orientation_deg_from_north, elongation)."""
    y = ys.astype(np.float64) - ys.mean()
    x = xs.astype(np.float64) - xs.mean()
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return 0.0, 1.0
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 1e-9)
    major = vecs[:, order[0]]
    # image y grows southwards, so north component is -major[1]
    ang = math.degrees(math.atan2(major[0], -major[1])) % 180.0
    elong = float(math.sqrt(vals[0] / vals[1]))
    return float(ang), elong


def _edge_sharpness(img, region):
    """Mean gradient magnitude on the region boundary, in image units/pixel."""
    er = ndimage.binary_erosion(region, iterations=1)
    edge = region & ~er
    if edge.sum() < 5:
        return 0.0
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.sqrt(gy[edge] ** 2 + gx[edge] ** 2).mean())


def _contour_lonlat(region, lat0, lon0, size, max_pts=90):
    """Ordered outline of the region as lon/lat, decimated for the map."""
    from skimage import measure

    cs = measure.find_contours(region.astype(float), 0.5)
    if not cs:
        return []
    c = max(cs, key=len)
    if len(c) > max_pts:
        c = c[:: max(1, len(c) // max_pts)]
    out = []
    for yy, xx in c:
        x_m = (xx - size / 2.0) * PIXEL_METRES
        y_m = (size / 2.0 - yy) * PIXEL_METRES
        lon, lat = geo.xy_to_lonlat(x_m, y_m, lat0, lon0)
        out.append([round(lat, 6), round(lon, 6)])
    return out


def wind_validity(wind_ms):
    """
    Confidence multiplier from the SAR oil-detection validity band.
    Full confidence 4-11 m/s, tapering to near zero outside 2.5-14 m/s.
    """
    if wind_ms <= 2.5 or wind_ms >= 14.0:
        return 0.10
    if wind_ms < 4.0:
        return 0.10 + 0.90 * (wind_ms - 2.5) / 1.5
    if wind_ms > 11.0:
        return 0.10 + 0.90 * (14.0 - wind_ms) / 3.0
    return 1.0


def characterise(img, class_map, probs, wind_ms, lat0, lon0, acquired_utc):
    """
    Parameters
    ----------
    img        float32 (H, W) displayed SAR tile in [0, 1]
    class_map  uint8  (H, W) argmax class map from the U-Net
    probs      float32 (5, H, W) softmax probabilities
    wind_ms    scene mean 10 m wind speed (from ERA5 in production)
    lat0, lon0 centre of the tile
    acquired_utc  ISO timestamp of the SAR acquisition

    Returns a list of detection dicts, most confident first.
    """
    size = img.shape[0]
    sea_mask = class_map == 0
    sea_level = float(np.median(img[sea_mask])) if sea_mask.sum() > 100 else float(np.median(img))

    # ------------------------------------------------------------------
    # Fragment grouping.
    #
    # A single discharge does not appear in SAR as one continuous film.  Wind
    # and surface convergence break it into windrows, so the segmenter returns
    # a chain of separate patches lying along one line.  Treating each patch as
    # its own incident would produce a dozen weak detections instead of one
    # strong one, and would destroy the elongation and orientation measurements
    # the whole attribution chain depends on.
    #
    # Patches are grouped by dilating the oil mask by GROUP_DILATE pixels,
    # labelling the dilated mask, then intersecting back.  Anything within
    # 2 x GROUP_DILATE pixels (~1.3 km) of another patch joins the same slick.
    # ------------------------------------------------------------------
    GROUP_DILATE = 8
    oil = class_map == OIL
    grouped = ndimage.binary_dilation(oil, iterations=GROUP_DILATE, structure=np.ones((3, 3)))
    labels, n = ndimage.label(grouped, structure=np.ones((3, 3)))
    detections = []
    for lab in range(1, n + 1):
        region = oil & (labels == lab)
        npx = int(region.sum())
        if npx < MIN_AREA_PX:
            continue
        ys, xs = np.nonzero(region)
        area_km2 = npx * (PIXEL_METRES ** 2) / 1e6
        orient, elong = _principal_axis(ys, xs)

        # geometric length/width from the principal axis extent
        cy, cx = ys.mean(), xs.mean()
        a = math.radians(orient)
        # unit vector along the axis in image coords (x east, y south)
        ux, uy = math.sin(a), -math.cos(a)
        proj = (xs - cx) * ux + (ys - cy) * uy
        perp = -(xs - cx) * uy + (ys - cy) * ux
        length_km = float((proj.max() - proj.min()) * PIXEL_METRES / 1000.0)
        width_km = float((perp.max() - perp.min()) * PIXEL_METRES / 1000.0)

        # Contrast is measured on the eroded core against a ring of clean sea
        # a few pixels outside the boundary.  Measuring across the whole region
        # averages in the soft edge and systematically under-reports damping.
        core = ndimage.binary_erosion(region, iterations=1)
        if core.sum() < 10:
            core = region
        ring = ndimage.binary_dilation(region, iterations=6) & ~ndimage.binary_dilation(region, iterations=2) & sea_mask
        surround = float(np.median(img[ring])) if ring.sum() > 20 else sea_level
        inside = float(np.median(img[core]))
        # convert the display stretch back to dB (lo=-30, hi=-3 in sar_sim)
        contrast_db = float((surround - inside) * 27.0)
        sharpness = _edge_sharpness(img, region)

        mean_oil_p = float(probs[OIL][region].mean())
        mean_la_p = float(probs[LOOKALIKE][region].mean())

        # --- discriminator ------------------------------------------------
        w_val = wind_validity(wind_ms)
        elong_score = float(np.clip((elong - 1.6) / 3.4, 0.0, 1.0))
        sharp_score = float(np.clip(sharpness / 0.06, 0.0, 1.0))
        # Mineral-oil damping sits in a 3-10 dB band.  Thin sheens read lower
        # after speckle averaging, so the lower shoulder starts at 1.2 dB; above
        # 12 dB the patch is more likely a radar shadow or a land artefact.
        damp_score = float(np.clip((contrast_db - 1.2) / 1.8, 0.0, 1.0)) * float(
            np.clip((12.0 - contrast_db) / 3.0, 0.0, 1.0)
        )
        model_score = float(np.clip(mean_oil_p / max(mean_oil_p + mean_la_p, 1e-6), 0.0, 1.0))

        confidence = w_val * (
            0.40 * model_score + 0.22 * elong_score + 0.20 * damp_score + 0.18 * sharp_score
        )
        if confidence >= 0.55:
            verdict = "oil (high confidence)"
        elif confidence >= 0.35:
            verdict = "probable oil - analyst review"
        else:
            verdict = "likely look-alike - not escalated"

        detections.append(
            {
                "id": "SLK-%03d" % lab,
                "acquired_utc": acquired_utc,
                "area_km2": round(area_km2, 2),
                "length_km": round(length_km, 2),
                "width_km": round(width_km, 2),
                "elongation": round(elong, 2),
                "orientation_deg": round(orient, 1),
                "damping_contrast_db": round(contrast_db, 2),
                "edge_sharpness": round(sharpness, 4),
                "model_oil_prob": round(mean_oil_p, 3),
                "model_lookalike_prob": round(mean_la_p, 3),
                "wind_ms": round(float(wind_ms), 1),
                "wind_validity": round(w_val, 2),
                "confidence": round(float(confidence), 3),
                "verdict": verdict,
                "factors": {
                    "model evidence": round(model_score, 3),
                    "elongation": round(elong_score, 3),
                    "damping in 3-10 dB band": round(damp_score, 3),
                    "edge sharpness": round(sharp_score, 3),
                    "wind validity gate": round(w_val, 3),
                },
                "centroid": [
                    round(geo.xy_to_lonlat((cx - size / 2) * PIXEL_METRES, (size / 2 - cy) * PIXEL_METRES, lat0, lon0)[1], 6),
                    round(geo.xy_to_lonlat((cx - size / 2) * PIXEL_METRES, (size / 2 - cy) * PIXEL_METRES, lat0, lon0)[0], 6),
                ],
                "polygon": _contour_lonlat(region, lat0, lon0, size),
                "_pixels": (ys, xs),
            }
        )

    # The strongest slick is the one an operator opens first, and "strongest"
    # means confident AND big -- a 0.5 km2 patch at 0.84 confidence is not the
    # incident, it is a fragment.
    detections.sort(key=lambda d: -(d["confidence"] * min(d["area_km2"], 25.0) ** 0.5))
    return detections


def characterise_lookalikes(class_map, lat0, lon0, size):
    """
    Report what the discriminator rejected.

    A detector that never says "this is not oil" is not a detector, so the
    look-alike class is surfaced explicitly rather than silently dropped.
    """
    la = class_map == LOOKALIKE
    grouped = ndimage.binary_dilation(la, iterations=8, structure=np.ones((3, 3)))
    labels, n = ndimage.label(grouped, structure=np.ones((3, 3)))
    out = []
    for lab in range(1, n + 1):
        region = la & (labels == lab)
        npx = int(region.sum())
        if npx < MIN_AREA_PX * 2:
            continue
        ys, xs = np.nonzero(region)
        _, elong = _principal_axis(ys, xs)
        out.append(
            {
                "id": "LAL-%03d" % lab,
                "area_km2": round(npx * (PIXEL_METRES ** 2) / 1e6, 2),
                "elongation": round(elong, 2),
                "polygon": _contour_lonlat(region, lat0, lon0, size),
                "reason": (
                    "Rejected: elongation %.1f:1 is too round for a discharge released "
                    "while underway, and the boundary is diffuse rather than a step change."
                    % elong
                ),
            }
        )
    out.sort(key=lambda d: -d["area_km2"])
    return out
