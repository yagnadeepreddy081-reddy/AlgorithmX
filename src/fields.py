"""Synthetic winds and surface currents for the pilot region.

No measured ERA5/CMEMS backend is connected. The interface can support one,
but current fields are generated from simplified mean flow, tides and eddies.
"""

import math
import numpy as np

# Pilot box: Mumbai High / Mumbai port approaches, Arabian Sea
PILOT = {
    "name": "Mumbai approaches / Mumbai High, Arabian Sea",
    "lat0": 18.95,
    "lon0": 71.85,
}

M_TIDE = 12.42 * 3600.0  # principal lunar semidiurnal period, seconds


class FieldProvider:
    """Generated representative fields for the pilot box.

    Parameters
    ----------
    seed        controls the mesoscale eddy realisation
    mean_wind   scene-mean 10 m wind speed in m/s (from the SAR scene / ERA5)
    wind_dir    meteorological direction the wind blows *from*, degrees
    """

    source = "synthetic representative field (pilot box)"

    def __init__(self, seed=0, mean_wind=7.0, wind_dir=290.0):
        rng = np.random.default_rng(seed)
        self.mean_wind = float(mean_wind)
        self.wind_dir = float(wind_dir)
        # mesoscale current variability: two random eddy centres
        self.eddies = [
            {
                "x": rng.uniform(-25000, 25000),
                "y": rng.uniform(-25000, 25000),
                "r": rng.uniform(9000, 20000),
                "amp": rng.uniform(-0.22, 0.22),
            }
            for _ in range(2)
        ]
        self.tide_phase = rng.uniform(0, 2 * math.pi)
        # West India Coastal Current, north-westward in the post-monsoon period
        self.mean_u = -0.13 + rng.uniform(-0.04, 0.04)
        self.mean_v = 0.22 + rng.uniform(-0.05, 0.05)

    # -- winds --------------------------------------------------------------
    def wind_at(self, x_m, y_m, t_s):
        """Return (u, v) 10 m wind in m/s. Meteorological 'from' convention in,
        oceanographic 'towards' vector out."""
        theta = math.radians(self.wind_dir)
        # blowing *towards* the opposite bearing
        u = -self.mean_wind * math.sin(theta)
        v = -self.mean_wind * math.cos(theta)
        # slow synoptic veer over the hindcast window
        veer = math.radians(6.0) * (t_s / 86400.0)
        cu = u * math.cos(veer) - v * math.sin(veer)
        cv = u * math.sin(veer) + v * math.cos(veer)
        return cu, cv

    # -- currents -----------------------------------------------------------
    def current_at(self, x_m, y_m, t_s):
        """Return (u, v) surface current in m/s at local metres and epoch-offset
        seconds.  Mean coastal current + semidiurnal tidal ellipse + eddies."""
        u, v = self.mean_u, self.mean_v
        ph = 2 * math.pi * t_s / M_TIDE + self.tide_phase
        u += 0.16 * math.cos(ph)
        v += 0.09 * math.sin(ph)
        for e in self.eddies:
            dx, dy = x_m - e["x"], y_m - e["y"]
            d = math.hypot(dx, dy)
            if d < 1.0:
                continue
            g = math.exp(-(d ** 2) / (2 * e["r"] ** 2))
            u += e["amp"] * g * (-dy / d)
            v += e["amp"] * g * (dx / d)
        return u, v

    def describe(self):
        cu, cv = self.current_at(0.0, 0.0, 0.0)
        return {
            "source": self.source,
            "region": PILOT["name"],
            "mean_wind_ms": round(self.mean_wind, 1),
            "wind_from_deg": round(self.wind_dir, 0),
            "surface_current_ms": round(math.hypot(cu, cv), 3),
            "current_towards_deg": round((math.degrees(math.atan2(cu, cv)) + 360) % 360, 0),
        }
