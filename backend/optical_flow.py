# Garaj Baras - optical_flow.py

import numpy as np
import cv2
from PIL import Image


def isolate_rain(frame_path, clutter_mask=None):
    """
    Opens a radar frame and returns a mask of rain pixels.
    Background (terrain, border, sidebar) is removed.
    If clutter_mask is provided, permanent ground clutter pixels are also removed.
    Returns: numpy array (uint8), 0=no rain, 255=rain
    """
    img = np.array(Image.open(frame_path).convert('RGB'))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Background masks to REMOVE
    green = (r >= 70) & (r <= 180) & (g >= 100) & (g <= 200) & (b >= 40)  & (b <= 100)
    brown = (r >= 150) & (r <= 220) & (g >= 120) & (g <= 180) & (b >= 60)  & (b <= 120)
    black = (r <= 20)  & (g <= 20)  & (b <= 20)
    gray  = (r >= 150) & (r <= 220) & (g >= 150) & (g <= 220) & (b >= 150) & (b <= 220)

    background = green | brown | black | gray
    rain_mask = np.where(background, 0, 255).astype(np.uint8)

    # Remove permanent ground clutter pixels if mask provided
    if clutter_mask is not None:
        rain_mask = np.where(clutter_mask > 0, 0, rain_mask).astype(np.uint8)

    return rain_mask


def build_clutter_mask(frames_list, threshold=0.95):
    """
    Identifies permanent ground clutter pixels by analyzing all radar frames.

    A pixel is clutter if it appears as "rain" in MORE THAN threshold (80%) of frames.
    Real rain moves between frames; clutter stays fixed across all frames.

    Args:
        frames_list : list of all frame paths (use all 16 frames)
        threshold   : fraction of frames a pixel must be "rain" in to be clutter (default 0.80)

    Returns: clutter_mask (numpy array, uint8, 255=clutter, 0=not clutter)
    """
    if not frames_list:
        return None

    # Stack rain masks — without clutter correction (raw masks)
    masks = []
    for path in frames_list:
        mask = isolate_rain(path, clutter_mask=None)
        masks.append(mask > 0)

    stacked = np.stack(masks, axis=0)  # shape: (N, H, W)
    rain_fraction = stacked.mean(axis=0)  # 0.0–1.0 per pixel

    clutter_mask = np.where(rain_fraction > threshold, 255, 0).astype(np.uint8)
    return clutter_mask


def calculate_optical_flow(frame1_path, frame2_path, clutter_mask=None):
    """
    Calculates dense optical flow between two consecutive radar frames.
    Only considers pixels where rain exists in the first frame.
    Returns: (mean_dx, mean_dy) in pixels.
    """
    rain_mask = isolate_rain(frame1_path, clutter_mask=clutter_mask)

    img1 = np.array(Image.open(frame1_path).convert('L'))
    img2 = np.array(Image.open(frame2_path).convert('L'))

    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    rain_pixels = rain_mask > 0
    if rain_pixels.sum() == 0:
        return (0.0, 0.0)

    mean_dx = float(np.mean(flow[:, :, 0][rain_pixels]))
    mean_dy = float(np.mean(flow[:, :, 1][rain_pixels]))
    return (mean_dx, mean_dy)


def get_direction_string(dx, dy):
    """
    Converts (dx, dy) movement to 8-point compass direction.
    dx positive=East, dy positive=South.
    """
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    bearing = (angle + 90) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[int((bearing + 22.5) / 45) % 8]


def get_movement_vector(frame_data, clutter_mask=None):
    """
    Calculates average rain movement across all consecutive frame pairs.

    Builds a clutter mask from ALL provided frames first (if not supplied),
    then uses clutter-cleaned masks for optical flow.

    Args:
        frame_data: list of (frame_path, timestamp_or_None)

    Uses OCR timestamps to compute the real time gap between consecutive frames.
    - Skips pairs with gap > 20 mins
    - Uses assumed gap=10 mins if OCR timestamp missing

    Returns dx/dy normalized to a 10-minute movement unit, so that
    downstream shifting via minutes_ahead/10.0 remains correct.

    Discards pairs where movement > 30 px (optical flow noise).

    Returns: (mean_dx, mean_dy, direction_from, direction_to, speed_kmh)
    """
    if len(frame_data) < 2:
        return (0.0, 0.0, "N", "S", 0.0)

    valid_vectors = []  # (dx_10, dy_10, speed_kmh)

    for i in range(len(frame_data) - 1):
        path1, ts1 = frame_data[i]
        path2, ts2 = frame_data[i + 1]

        # Check time gap
        if ts1 is not None and ts2 is not None:
            gap_mins = (ts2 - ts1).total_seconds() / 60.0

            if gap_mins < 0:
                print(f"  SKIP (time went backward): frame_{i:02d}->{i+1:02d}")
                continue
            if gap_mins > 20:
                print(f"  SKIP (gap={gap_mins:.0f}mins too large): frame_{i:02d}->{i+1:02d}")
                continue

            print(f"  USE gap={gap_mins:.0f}mins: frame_{i:02d}->{i+1:02d}")
        else:
            # No timestamps - assume 10 min gap
            gap_mins = 10.0
            print(f"  USE gap=10mins(assumed): frame_{i:02d}->{i+1:02d}")

        # Calculate optical flow
        dx, dy = calculate_optical_flow(
            path1,
            path2,
            clutter_mask=clutter_mask
        )

        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        if magnitude < 0.5:
            print(f"  SKIP (no rain) mag={magnitude:.2f}px: frame_{i:02d}->{i+1:02d}")
            continue
        if magnitude > 30:
            print(f"  SKIP (noise)   mag={magnitude:.1f}px: frame_{i:02d}->{i+1:02d}")
            continue

        # Calculate REAL speed using real time gap
        km_moved = magnitude * 0.877
        speed_kmh = (km_moved / gap_mins) * 60.0

        # Normalize dx/dy to a 10-min unit for downstream predictions.
        dx_10 = dx * (10.0 / gap_mins)
        dy_10 = dy * (10.0 / gap_mins)

        print(
            f"  OK dx={dx:.2f} dy={dy:.2f} mag={magnitude:.1f}px "
            f"speed={speed_kmh:.1f}km/h (gap={gap_mins:.1f}m)"
        )
        valid_vectors.append((dx_10, dy_10, speed_kmh))

    if not valid_vectors:
        return (0.0, 0.0, "Unknown", "Unknown", 0.0)

    # Weight recent pairs more
    weights = list(range(1, len(valid_vectors) + 1))
    total_w = sum(weights)

    avg_dx = sum(v[0] * w for v, w in zip(valid_vectors, weights)) / total_w
    avg_dy = sum(v[1] * w for v, w in zip(valid_vectors, weights)) / total_w
    avg_speed = sum(v[2] * w for v, w in zip(valid_vectors, weights)) / total_w

    direction_to = get_direction_string(avg_dx, avg_dy)
    opposite = {"N":"S","NE":"SW","E":"W","SE":"NW","S":"N","SW":"NE","W":"E","NW":"SE"}
    direction_from = opposite[direction_to]

    return (avg_dx, avg_dy, direction_from, direction_to, avg_speed)


if __name__ == "__main__":
    from radar import get_recent_frames, get_all_frames
    all_frame_data = get_all_frames()
    recent_frame_data = get_recent_frames(n=6)

    all_paths = [p for (p, _ts) in all_frame_data]
    print(f"Building clutter mask from {len(all_paths)} frames...")
    clutter_mask  = build_clutter_mask(all_paths)
    clutter_pixels = int((clutter_mask > 0).sum())
    print(f"Clutter pixels identified: {clutter_pixels}")
    print()

    dx, dy, direction_from, direction_towards, speed = get_movement_vector(
        recent_frame_data, clutter_mask=clutter_mask
    )
    print(f"Rain coming FROM: {direction_from}")
    print(f"Rain moving TO:   {direction_towards}")
    print(f"Approx speed:     {speed:.1f} km/h")
