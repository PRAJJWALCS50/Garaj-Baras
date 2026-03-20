# Garaj Baras - georef.py

# --- Calibration constants (manually verified) ---
CENTER_PIXEL_X     = 196
CENTER_PIXEL_Y     = 246
CENTER_LAT         = 26.7606
CENTER_LON         = 80.8893
PIXELS_PER_DEG_LAT = 86.73
PIXELS_PER_DEG_LON = 77.44

# Radar image dimensions
IMAGE_WIDTH        = 423
IMAGE_HEIGHT       = 442


def latlon_to_pixel(lat, lon):
    """Convert real-world lat/lon to radar image pixel (x, y)."""
    px = CENTER_PIXEL_X + (lon - CENTER_LON) * PIXELS_PER_DEG_LON
    py = CENTER_PIXEL_Y - (lat - CENTER_LAT) * PIXELS_PER_DEG_LAT
    return (int(round(px)), int(round(py)))


def pixel_to_latlon(px, py):
    """Convert radar image pixel (x, y) to real-world lat/lon."""
    lat = CENTER_LAT + (CENTER_PIXEL_Y - py) / PIXELS_PER_DEG_LAT
    lon = CENTER_LON + (px - CENTER_PIXEL_X) / PIXELS_PER_DEG_LON
    return (round(lat, 4), round(lon, 4))


def is_within_radar(lat, lon):
    """Check if a lat/lon coordinate falls within the radar image bounds."""
    px, py = latlon_to_pixel(lat, lon)
    return (0 <= px <= 423 and 0 <= py <= 442)


if __name__ == "__main__":
    # Test known locations
    test_locations = [
        ("Lucknow",   26.7606, 80.8893),
        ("Fatehpur",  25.9298, 80.8133),
        ("Kanpur",    26.4499, 80.3319),
        ("Varanasi",  25.3176, 82.9739),
        ("Sitapur",   27.5706, 80.6829),
        ("Raebareli", 26.2309, 81.2399),
    ]

    print("Testing geo-referencing:")
    for name, lat, lon in test_locations:
        px, py = latlon_to_pixel(lat, lon)
        lat2, lon2 = pixel_to_latlon(px, py)
        print(f"{name}: ({lat},{lon}) -> pixel({px},{py}) -> ({lat2},{lon2})")
