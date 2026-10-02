"""Sohra: header site 25.2680N, 91.7332E; 240-km outer ring.

Native 1078x770 MAX(Z): map (43,193)-(598,748), crosshair (320,470),
radius 277px. AEQD range/bearing graticule checked against Guwahati,
Shillong, Tezpur and Agartala. The adapter resamples to 0.877 km/pixel.
"""
from georef_native import geometry, KM_PER_PX

CENTER_LAT, CENTER_LON, RANGE_KM = 25.2680, 91.7332, 240.0
IMAGE_WIDTH, CENTER_PX, latlon_to_pixel, pixel_to_latlon, is_within_radar = geometry(
    CENTER_LAT, CENTER_LON, RANGE_KM)
IMAGE_HEIGHT = IMAGE_WIDTH
CENTER_PY = CENTER_PX
PX_PER_KM = 1 / KM_PER_PX
