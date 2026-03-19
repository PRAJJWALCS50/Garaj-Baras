COLOR_TABLE = [
    (0,   25,  176, 20),
    (0,   58,  200, 25),
    (0,   71,  255, 30),
    (26,  163, 255, 35),
    (135, 241, 255, 38),
    (0,   200, 0,   41),
    (255, 255, 0,   44),
    (255, 165, 0,   50),
    (255, 0,   0,   55),
    (255, 255, 255, 60),
]

import math

def rgb_to_dbz(r, g, b):
    min_dist = float('inf')
    best_dbz = 0
    for cr, cg, cb, dbz in COLOR_TABLE:
        dist = math.sqrt((r-cr)**2 + (g-cg)**2 + (b-cb)**2)
        if dist < min_dist:
            min_dist = dist
            best_dbz = dbz
    if min_dist > 80:
        return 0
    return best_dbz

def dbz_to_label(dbz):
    if dbz == 0:
        return "No Rain", "#000000"
    elif dbz < 25:
        return "Very Light Rain", "#4169E1"
    elif dbz < 35:
        return "Light Rain", "#00BFFF"
    elif dbz < 45:
        return "Moderate Rain", "#00FF00"
    elif dbz < 55:
        return "Heavy Rain", "#FFD700"
    else:
        return "Very Heavy Rain", "#FF4500"

def get_pixel_intensity(frame_path, px, py):
    from PIL import Image
    img = Image.open(frame_path).convert('RGB')
    r, g, b = img.getpixel((px, py))
    dbz = rgb_to_dbz(r, g, b)
    label, color = dbz_to_label(dbz)
    return {
        "dbz": dbz,
        "label": label,
        "color": color,
        "rgb": (r, g, b)
    }

def enrich_results(results, waypoints_latlon, 
                   latest_frame, dx, dy, lag_info=None):
    from PIL import Image
    import numpy as np

    enriched = []
    img = Image.open(latest_frame).convert('RGB')
    arr = np.array(img)

    lag_mins = lag_info['lag_mins'] if lag_info else 25.0

    for r, (lat, lon, eta) in zip(results, waypoints_latlon):
        from georef import latlon_to_pixel
        px, py = latlon_to_pixel(lat, lon)

        if r["rain_expected"]:
            effective_eta = eta + lag_mins
            # Get shifted pixel position at effective_eta
            frames_ahead = effective_eta / 10.0
            shifted_px = int(px - dx * frames_ahead)
            shifted_py = int(py - dy * frames_ahead)

            # Clamp to image bounds
            h, w = arr.shape[:2]
            shifted_px = max(0, min(w - 1, shifted_px))
            shifted_py = max(0, min(h - 1, shifted_py))

            rv, gv, bv = arr[shifted_py, shifted_px]
            dbz = rgb_to_dbz(rv, gv, bv)
            label, color = dbz_to_label(dbz)

            if dbz == 0:
                label, color = dbz_to_label(20)

            message = f"{label} expected in {eta:.0f} mins"
            message += f"\n(radar ~{lag_mins:.0f} mins old — actual arrival may vary)"
        else:
            label  = "No Rain"
            color  = "#000000"
            dbz    = 0
            message = "Clear"

        enriched.append({
            "lat": lat,
            "lon": lon,
            "eta_mins": eta,
            "rain_expected": r["rain_expected"],
            "confidence": r["confidence"],
            "label": label,
            "color": color,
            "dbz": dbz,
            "message": message
        })

    return enriched

if __name__ == "__main__":
    # Test 1: Color mapping
    test_pixels = [
        ("Very Light", 0,   25,  176),
        ("Light",      0,   71,  255),
        ("Moderate",   135, 241, 255),
        ("Heavy",      255, 255, 0  ),
        ("Very Heavy", 255, 0,   0  ),
        ("Background", 100, 173, 64 ),
    ]

    print("=" * 55)
    print("FUZZY INTENSITY TEST")
    print("=" * 55)
    for name, r, g, b in test_pixels:
        dbz = rgb_to_dbz(r, g, b)
        label, color = dbz_to_label(dbz)
        print(f"  {name:<12} RGB({r:3d},{g:3d},{b:3d})"
              f" -> {dbz:2.0f}dBZ -> {label:<15} {color}")

    print()

    # Test 2: Full pipeline Bijnor -> Muzaffarnagar
    print("=" * 55)
    print("ROUTE TEST: Badaun -> Bulandshahr (FIXED LAG)")
    print("=" * 55)

    from radar import get_recent_frames, get_all_frames, get_radar_lag_mins
    from optical_flow import get_movement_vector, build_clutter_mask
    from georef import latlon_to_pixel, is_within_radar
    from prediction import (generate_waypoints, 
                            check_route_rain, 
                            haversine_km)

    all_frames, _    = get_all_frames()
    recent_frames, _ = get_recent_frames(n=6)
    lag_info         = get_radar_lag_mins()
    
    clutter_mask = build_clutter_mask(all_frames)
    dx, dy, dir_from, dir_to, speed = get_movement_vector(
        recent_frames, clutter_mask=clutter_mask
    )
    latest_frame = recent_frames[-1]

    print(f"Lag Assumption : {lag_info['message']}")
    print(f"Rain moves     : FROM {dir_from} -> TO {dir_to} at {speed:.1f} km/h")
    print()

    start = (28.0315, 79.1247)  # Badaun
    end   = (28.4070, 77.8498)  # Bulandshahr

    start_in = is_within_radar(start[0], start[1])
    end_in   = is_within_radar(end[0],   end[1])
    print(f"Start in radar: {start_in}")
    print(f"End   in radar: {end_in}")
    print()

    waypoints_latlon = generate_waypoints(
        start, end, spacing_km=2.0
    )
    total_dist = haversine_km(
        start[0], start[1], end[0], end[1]
    )

    print(f"Distance : {total_dist:.1f} km")
    print(f"Waypoints: {len(waypoints_latlon)}")
    print()

    waypoints_pixels = [
        (latlon_to_pixel(lat, lon)[0],
         latlon_to_pixel(lat, lon)[1],
         eta)
        for lat, lon, eta in waypoints_latlon
    ]

    max_eta = waypoints_latlon[-1][2]
    results = check_route_rain(
        waypoints_pixels, dx, dy,
        latest_frame,
        eta_minutes=max_eta,
        clutter_mask=clutter_mask,
        lag_mins=lag_info['lag_mins']
    )

    enriched = enrich_results(
        results, waypoints_latlon,
        latest_frame, dx, dy,
        lag_info=lag_info
    )

    print(f"{'WP':<5} {'ETA':>5} {'Status':<12} "
          f"{'Intensity':<16} {'dBZ':>4}  Note")
    print("-" * 65)

    rain_count = 0
    for i, e in enumerate(enriched):
        in_bounds = is_within_radar(e["lat"], e["lon"])
        status = "RAIN" if e["rain_expected"] else "CLEAR"
        note = ""
        if not in_bounds:
            note = "WARNING: outside radar"
        elif e["eta_mins"] > 60:
            note = "WARNING: long range"
        if e["rain_expected"]:
            rain_count += 1
        
        print(f"WP{i+1:02d}  {e['eta_mins']:>4.0f}m  "
              f"{status:<12} {e['label']:<16} "
              f"{e['dbz']:>4.0f}  {note}")

    print()
    print(f"SUMMARY:")
    print(f"  Total waypoints : {len(enriched)}")
    print(f"  Rain waypoints  : {rain_count}")
    print(f"  Clear waypoints : {len(enriched) - rain_count}")
    if rain_count > 0:
        first = next(e for e in enriched if e["rain_expected"])
        print(f"  First rain at   : {first['eta_mins']:.0f} mins")
        print(f"  First intensity : {first['label']}")
        print(f"  Lag assumption  : {lag_info['lag_mins']} mins")
