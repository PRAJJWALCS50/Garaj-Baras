# Garaj Baras - georef.py

# --- Calibration constants (manually verified) ---
CENTER_PIXEL_X     = 264
CENTER_PIXEL_Y     = 262
CENTER_LAT         = 28.5562
CENTER_LON         = 77.1000
PIXELS_PER_DEG_LAT = 117.87
PIXELS_PER_DEG_LON = 103.53

# Radar image dimensions
IMAGE_WIDTH        = 527
IMAGE_HEIGHT       = 525


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
    return (0 <= px <= 527 and 0 <= py <= 525)


if __name__ == "__main__":
    # Test known locations
    test_locations = [
        ("IGI Airport", 28.5562, 77.1000),
        ("Noida",       28.5355, 77.3910),
        ("Gurugram",    28.4595, 77.0266),
        ("Ghaziabad",   28.6692, 77.4538),
        ("Faridabad",   28.4089, 77.3178),
    ]

    print("Testing geo-referencing:")
    for name, lat, lon in test_locations:
        px, py = latlon_to_pixel(lat, lon)
        lat2, lon2 = pixel_to_latlon(px, py)
        print(f"{name}: ({lat},{lon}) -> pixel({px},{py}) -> ({lat2},{lon2})")
