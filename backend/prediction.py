import math
import numpy as np
import cv2
from PIL import Image
from optical_flow import isolate_rain


def debug_route_on_image(frames, waypoints_pixels, clutter_mask):
    """
    Temporary debug: Prints the raw pixel color at each waypoint 
    to see what's actually under the hood.
    """
    img = Image.open(frames[-1]).convert('RGB')
    arr = np.array(img)
    
    print("\nDEBUG: Pixel colors at each waypoint:")
    for i, (px, py, eta) in enumerate(waypoints_pixels):
        # Clip coordinates to image bounds
        h, w = arr.shape[:2]
        safe_y = max(0, min(py, h - 1))
        safe_x = max(0, min(px, w - 1))
        
        r, g, b = arr[safe_y, safe_x]
        
        # Simple classification heuristics
        is_green = (70 < r < 180) and (100 < g < 200) and (40 < b < 100)
        is_blue  = (b > 150) and (r < 100)
        is_clutter = clutter_mask[safe_y, safe_x] > 0
        
        label = "GREEN(background)" if is_green else \
                "BLUE(rain)" if is_blue else \
                "OTHER"
        clutter_status = "CLUTTER" if is_clutter else "clean"
        
        print(f"  WP{i+1:02d} pixel({px:3d},{py:3d}) RGB({r:3d},{g:3d},{b:3d}) -> {label} | {clutter_status}")


DRIVING_SPEED_KMH = 40.0  # city driving speed for Delhi NCR


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Returns great-circle distance in km between two lat/lon points.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def generate_waypoints(start_latlon, end_latlon, spacing_km=2.0):
    """
    Generate evenly spaced waypoints between start and end coordinates.

    Args:
        start_latlon : (lat, lon) tuple
        end_latlon   : (lat, lon) tuple
        spacing_km   : distance between consecutive waypoints in km (default 2.0)

    Returns:
        list of (lat, lon, eta_mins) tuples
        ETA calculated assuming DRIVING_SPEED_KMH (40 km/h city speed).
        Always includes start (eta=0) and end point.
    """
    lat1, lon1 = start_latlon
    lat2, lon2 = end_latlon
    total_km = haversine_km(lat1, lon1, lat2, lon2)

    if total_km == 0:
        return [(lat1, lon1, 0)]

    num_segments = max(1, int(math.ceil(total_km / spacing_km)))
    waypoints = []
    for i in range(num_segments + 1):
        t = i / num_segments          # 0.0 … 1.0 along the route
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        dist_from_start = t * total_km
        eta_mins = (dist_from_start / DRIVING_SPEED_KMH) * 60.0
        waypoints.append((lat, lon, eta_mins))

    return waypoints


def predict_rain_position(latest_frame_path, dx, dy, minutes_ahead, clutter_mask=None):
    """
    Predicts where rain will be after minutes_ahead minutes.

    Shifts the current rain mask forward using the optical flow vector.
    Each 10-minute frame corresponds to one (dx, dy) movement unit.
    Clutter pixels are removed before shifting if clutter_mask is provided.

    Returns: predicted rain mask (numpy array, uint8, 0=no rain, 255=rain)
    """
    rain_mask = isolate_rain(latest_frame_path, clutter_mask=clutter_mask).astype(np.float32)

    frames_ahead = minutes_ahead / 10.0
    shift_x = dx * frames_ahead
    shift_y = dy * frames_ahead

    h, w = rain_mask.shape
    M = np.float32([[1, 0, shift_x],
                    [0, 1, shift_y]])
    shifted = cv2.warpAffine(rain_mask, M, (w, h), flags=cv2.INTER_LINEAR)

    return (shifted > 127).astype(np.uint8) * 255


def is_rain_at_pixel(predicted_mask, px, py, radius=1):
    """
    Checks if predicted rain exists at or near pixel (px, py).
    Uses a radius of 1 pixel (3x3 grid) to account for minor error.
    Returns True if rain detected, False otherwise.
    """
    h, w = predicted_mask.shape
    # Ensure coordinates are within bounds
    py = max(0, min(py, h - 1))
    px = max(0, min(px, w - 1))

    # BUG 4: Check center pixel first
    if predicted_mask[py, px] > 0:
        return True

    # Check radius only if center is not rain
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)

    region = predicted_mask[y0:y1, x0:x1]
    return bool(np.any(region > 0))


def check_route_rain(waypoints_pixels, dx, dy, latest_frame_path,
                     eta_minutes, clutter_mask=None, lag_mins=25.0):
    """
    Checks each waypoint on a route for expected rain at its ETA,
    adjusted for radar lag.

    Args:
        waypoints_pixels   : list of (px, py, eta_mins) tuples
        dx, dy             : rain movement vector (pixels per 10-min frame)
        latest_frame_path  : path to the most recent radar frame
        eta_minutes        : total trip duration (kept for API clarity)
        clutter_mask       : optional clutter mask to clean predictions
        lag_mins           : minutes elapsed since the radar frame was last updated

    For each waypoint, checks rain at (effective_eta), (effective_eta+5),
    and (effective_eta+10) minutes.
    """
    mask_cache = {}

    def get_mask(t_mins):
        # Round to nearest integer to help caching
        t = int(max(0, round(t_mins)))
        if t not in mask_cache:
            mask_cache[t] = predict_rain_position(
                latest_frame_path, dx, dy, t, clutter_mask=clutter_mask
            )
        return mask_cache[t]

    # Load latest frame to check if waypoints start as green/background
    try:
        img_arr = np.array(Image.open(latest_frame_path).convert('RGB'))
    except Exception:
        img_arr = None

    results = []
    for i, (px, py, eta) in enumerate(waypoints_pixels):
        effective_eta = eta + lag_mins
        
        # Buffer: check effective_eta, +5, +10
        check_times = [effective_eta, effective_eta + 5, effective_eta + 10]
        hits = 0

        # Check if center pixel of the ORIGINAL frame is green (background)
        is_center_green = False
        if img_arr is not None:
            # Clamp coordinates for safety
            h, w = img_arr.shape[:2]
            spy = max(0, min(py, h - 1))
            spx = max(0, min(px, w - 1))
            r, g, b = img_arr[spy, spx]
            is_center_green = (70 < r < 180) and (100 < g < 200) and (40 < b < 100)

        for t in check_times:
            mask = get_mask(t)
            # If starting on green, be strict.
            if mask[py, px] > 0:
                hits += 1
            elif not is_center_green and is_rain_at_pixel(mask, px, py, radius=1):
                hits += 1

        if hits == 3:
            confidence = "high"
        elif hits == 2:
            confidence = "medium"
        elif hits == 1:
            confidence = "low"
        else:
            confidence = "none"

        # Long range confidence nerf
        if eta > 60 and confidence != "none":
            confidence = "low"

        results.append({
            "px": px,
            "py": py,
            "eta_mins": eta,
            "effective_eta": effective_eta,
            "rain_expected": hits > 0,
            "confidence": confidence,
            "lag_applied": lag_mins
        })

    return results


if __name__ == "__main__":
    from radar import get_recent_frames, get_all_frames
    from optical_flow import get_movement_vector, build_clutter_mask
    from georef import latlon_to_pixel, is_within_radar

    # Setup - run once for both routes
    print("Setting up radar data...")
    all_frames = get_all_frames()
    recent_frames = get_recent_frames(n=6)
    clutter_mask = build_clutter_mask(all_frames)
    dx, dy, dir_from, dir_to, speed = get_movement_vector(
        recent_frames, clutter_mask=clutter_mask
    )
    latest_frame = recent_frames[-1]
    print(f"Rain moving FROM {dir_from} -> TO {dir_to} at {speed:.1f} km/h")
    print()

    # Define both routes
    routes = [
        {
            "name": "Ghaziabad -> Jind",
            "start": (28.6692, 77.4538),  # Ghaziabad
            "end":   (29.3162, 76.3148),  # Jind
        },
        {
            "name": "Etah -> Sambhal",
            "start": (27.5590, 78.6624),  # Etah
            "end":   (28.5836, 78.5685),  # Sambhal
        },
    ]

    # Run prediction for each route
    for route in routes:
        print("=" * 50)
        print(f"ROUTE: {route['name']}")
        print("=" * 50)

        # Check bounds first
        start_in = is_within_radar(route['start'][0], route['start'][1])
        end_in   = is_within_radar(route['end'][0],   route['end'][1])
        print(f"Start in radar bounds: {start_in}")
        print(f"End   in radar bounds: {end_in}")

        if not start_in or not end_in:
            print("WARNING: Part of route outside Delhi DWR coverage!")
            print("   Predictions for out-of-bounds waypoints unreliable!")
            print()

        # Generate waypoints
        waypoints_latlon = generate_waypoints(
            route['start'],
            route['end'],
            spacing_km=2.0
        )

        total_dist = haversine_km(
            route['start'][0], route['start'][1],
            route['end'][0],   route['end'][1]
        )
        print(f"Distance: {total_dist:.1f} km")
        print(f"Waypoints: {len(waypoints_latlon)}")
        print()

        # Convert to pixels
        waypoints_pixels = []
        for lat, lon, eta in waypoints_latlon:
            px, py = latlon_to_pixel(lat, lon)
            in_bounds = is_within_radar(lat, lon)
            waypoints_pixels.append((px, py, eta, in_bounds))

        # Run rain check
        max_eta = waypoints_latlon[-1][2]
        results = check_route_rain(
            [(px, py, eta) for px, py, eta, _ in waypoints_pixels],
            dx, dy,
            latest_frame,
            eta_minutes=max_eta,
            clutter_mask=clutter_mask
        )

        # Print results
        print(f"{'WP':<4} {'Lat':>8} {'Lon':>8} {'ETA':>6} {'Status':<6} {'Conf':<8} {'Note'}")
        print("-" * 65)
        for i, (r, (lat, lon, eta)) in enumerate(
            zip(results, waypoints_latlon)
        ):
            in_bounds = is_within_radar(lat, lon)
            status = "RAIN" if r["rain_expected"] else "CLEAR"
            conf   = r["confidence"]
            note   = ""
            if not in_bounds:
                note = "out of bounds"
            elif eta > 60:
                note = "long range"
            print(f"WP{i+1:02d} {lat:>8.4f} {lon:>8.4f} {eta:>5.0f}m {status:<6s} {conf:<8} {note}")

        print()
