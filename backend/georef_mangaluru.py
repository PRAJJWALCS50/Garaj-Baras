"""Mangaluru MAXDISPLAY(Z), 250-km radar at Shakthi Nagar.

Source 1310x1080: map (0,200)-(880,1080), ring center (440,640),
50/100/150/200/250-km radii 88/176/264/352/440 px. Site rendering
position (12.9037N,74.8620E) fitted to the source's IXE/CNN/CLT/MYS
airport markers; residuals <=1.4 source pixels. Adapter normalizes to
0.877 km/pixel, with exact spherical AEQD inverse and circular bounds.
"""
from georef_native import geometry, KM_PER_PX

CENTER_LAT, CENTER_LON, RANGE_KM = 12.9037, 74.8620, 250.0
IMAGE_WIDTH, CENTER_PX, latlon_to_pixel, pixel_to_latlon, is_within_radar = geometry(
    CENTER_LAT, CENTER_LON, RANGE_KM)
IMAGE_HEIGHT = IMAGE_WIDTH
CENTER_PY = CENTER_PX
PX_PER_KM = 1 / KM_PER_PX
