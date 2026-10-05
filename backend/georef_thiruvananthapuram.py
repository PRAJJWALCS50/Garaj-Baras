"""Thiruvananthapuram IMD CDWRTERLS, 240 km MAX(Z) coverage.

Site 8.5374 N, 76.8657 E and 240 km range are printed in the source.
Source crosshair (300,438); outer ring radius 257.7 px, measured from
the western arc (0.33 source-pixel RMS). Normalize to 0.877 km/px.
"""
from georef_native import geometry, KM_PER_PX

CENTER_LAT, CENTER_LON, RANGE_KM = 8.5374, 76.8657, 240.0
IMAGE_WIDTH, CENTER_PX, latlon_to_pixel, pixel_to_latlon, is_within_radar = geometry(
    CENTER_LAT, CENTER_LON, RANGE_KM)
IMAGE_HEIGHT = IMAGE_WIDTH
CENTER_PY = CENTER_PX
PX_PER_KM = 1 / KM_PER_PX
