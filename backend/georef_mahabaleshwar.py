"""Mahabaleshwar: IMD network site 17.9217N, 73.6556E; 170-km range.

720x720 IRIS MAX_Z: square map ends at y=640; its width changes with
the vertical-panel height (453px on the inspected reflectivity image).
Detect that border per frame; center is the midpoint of the square.
The 100/150-km rings and Pune/Satara/Ratnagiri/Kolhapur markers check
the AEQD geometry. Resampled to 0.877 km/pixel before processing.
"""
from georef_native import geometry, KM_PER_PX

CENTER_LAT, CENTER_LON, RANGE_KM = 17.9217, 73.6556, 170.0
IMAGE_WIDTH, CENTER_PX, latlon_to_pixel, pixel_to_latlon, is_within_radar = geometry(
    CENTER_LAT, CENTER_LON, RANGE_KM)
IMAGE_HEIGHT = IMAGE_WIDTH
CENTER_PY = CENTER_PX
PX_PER_KM = 1 / KM_PER_PX
