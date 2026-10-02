"""AEQD geometry for native IMD products resampled to 0.877 km/pixel."""
import math

KM_PER_PX = 0.877
EARTH_RADIUS_KM = 6371.0


def geometry(lat, lon, range_km):
    radius = range_km / KM_PER_PX
    center = math.ceil(radius)
    width = 2 * center + 1
    clat, clon = math.radians(lat), math.radians(lon)

    def forward(la, lo):
        la, dl = math.radians(la), math.radians(lo) - clon
        c = math.acos(max(-1.0, min(1.0, math.sin(clat) * math.sin(la)
                         + math.cos(clat) * math.cos(la) * math.cos(dl))))
        bearing = math.atan2(math.sin(dl) * math.cos(la),
                            math.cos(clat) * math.sin(la)
                            - math.sin(clat) * math.cos(la) * math.cos(dl))
        distance = c * EARTH_RADIUS_KM / KM_PER_PX
        return center + distance * math.sin(bearing), center - distance * math.cos(bearing)

    def latlon_to_pixel(la, lo):
        x, y = forward(la, lo)
        return int(round(x)), int(round(y))

    def pixel_to_latlon(x, y):
        dx, dy = x - center, center - y
        delta = math.hypot(dx, dy) * KM_PER_PX / EARTH_RADIUS_KM
        bearing = math.atan2(dx, dy)
        la = math.asin(math.sin(clat) * math.cos(delta)
                       + math.cos(clat) * math.sin(delta) * math.cos(bearing))
        lo = clon + math.atan2(math.sin(bearing) * math.sin(delta) * math.cos(clat),
                              math.cos(delta) - math.sin(clat) * math.sin(la))
        return math.degrees(la), math.degrees(lo)

    def is_within_radar(la, lo):
        x, y = forward(la, lo)
        return math.hypot(x - center, y - center) <= radius

    return width, center, latlon_to_pixel, pixel_to_latlon, is_within_radar
