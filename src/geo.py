"""Small geodesy helpers -- local flat-earth projection about a scene origin.

Over a 20 km tile the error from treating the surface as locally flat is a few
metres, far below the kilometre-scale uncertainty of the drift model, so a full
projected CRS would be false precision here.
"""

import math

R_EARTH = 6371008.8


def m_per_deg(lat_deg):
    """Metres per degree of latitude and of longitude at this latitude."""
    lat = math.radians(lat_deg)
    mlat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    mlon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return mlat, mlon


def xy_to_lonlat(x_m, y_m, lat0, lon0):
    """Local east/north metres -> lon/lat degrees."""
    mlat, mlon = m_per_deg(lat0)
    return lon0 + x_m / mlon, lat0 + y_m / mlat


def lonlat_to_xy(lon, lat, lat0, lon0):
    mlat, mlon = m_per_deg(lat0)
    return (lon - lon0) * mlon, (lat - lat0) * mlat


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2
        + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    )
    return 2 * R_EARTH * math.asin(math.sqrt(a)) / 1000.0


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing, degrees clockwise from north."""
    p = math.pi / 180.0
    dl = (lon2 - lon1) * p
    y = math.sin(dl) * math.cos(lat2 * p)
    x = math.cos(lat1 * p) * math.sin(lat2 * p) - math.sin(lat1 * p) * math.cos(lat2 * p) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def axis_difference_deg(a, b):
    """
    Smallest angle between two *undirected axes* (0-90 deg).

    A slick's principal axis has no head or tail, so a vessel steaming 070 and
    one steaming 250 are equally aligned with it.  Using a directed heading
    difference here would wrongly reject half of all true matches.
    """
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)
