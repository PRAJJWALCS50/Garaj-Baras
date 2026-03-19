# Garaj Baras - radar.py

import os
import requests
from PIL import Image, ImageSequence
from datetime import datetime, timezone, timedelta

GIF_URL = "https://mausam.imd.gov.in/Radar/animation/Converted/DELHI_MAXZ.gif"
GIF_SAVE_PATH = os.path.join(os.path.dirname(__file__), "delhi_radar.gif")
FRAMES_FOLDER = os.path.join(os.path.dirname(__file__), "frames")

IST = timezone(timedelta(hours=5, minutes=30))


def download_gif(url, save_path):
    """
    Download the radar GIF from IMD.
    Returns (True, None) on success.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True, None
    except Exception as e:
        print(f"Error downloading GIF: {e}")
        return False, None


def extract_frames(gif_path, output_folder):
    """
    Extract all frames from animated GIF to output_folder as PNGs.
    Returns list of frame file paths.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_paths = []
    try:
        with Image.open(gif_path) as im:
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                frame_rgb = frame.convert('RGB')
                frame_filename = f"frame_{i:02d}.png"
                frame_path = os.path.join(output_folder, frame_filename)
                frame_rgb.save(frame_path)
                frame_paths.append(frame_path)
    except Exception as e:
        print(f"Error extracting frames: {e}")

    return frame_paths


def get_latest_frames():
    """Download fresh GIF and extract all frames. Returns (frames, None)."""
    success, _ = download_gif(GIF_URL, GIF_SAVE_PATH)
    if success:
        return extract_frames(GIF_SAVE_PATH, FRAMES_FOLDER), None
    else:
        print("Failed to download radar GIF.")
        return [], None


def get_all_frames():
    """Alias for get_latest_frames."""
    return get_latest_frames()


def get_recent_frames(n=6):
    """Download fresh GIF and return only the LAST n frames + None."""
    frames, _ = get_latest_frames()
    if len(frames) <= n:
        return frames, None
    return frames[-n:], None


def get_radar_lag_mins():
    """
    IMD radar reality:
    - Radar scans every 10 mins
    - IMD uploads with delay
    - GIF animation updates irregularly
    - We cannot reliably extract exact scan time
    
    Conservative safe assumption: 25 mins lag
    This is honest and safe for predictions.
    """
    return {
        "lag_mins": 25.0,
        "freshness": "stale",
        "message": "Radar data assumed ~25 mins old (conservative estimate)",
        "warning": "Exact scan time unavailable. Using safe 25 min estimate."
    }


if __name__ == "__main__":
    print("=" * 45)
    print("RADAR LAG AWARENESS (FIXED)")
    print("=" * 45)
    
    frames, _ = get_recent_frames(n=6)
    lag_info = get_radar_lag_mins()
    
    print(f"Radar lag assumption : {lag_info['lag_mins']} mins")
    print(f"Freshness            : {lag_info['freshness']}")
    print(f"Message              : {lag_info['message']}")
    print(f"Frames extracted     : {len(frames)}")
    print()
    
    print(f"WARNING: {lag_info['warning']}")
    print("Lag correction (25 min) applied to all predictions.")
