"""IMD Thiruvananthapuram animation and current MAX(Z), native decoding."""
from native_radar import NativeRadarFeed
from radar import RADAR_TTL_SEC, get_radar_lag_mins

_feed = NativeRadarFeed('thiruvananthapuram', 'TVM', 'https://mausam.imd.gov.in/Radar/caz_tvm.gif')
GIF_URL = _feed.gif_url
GIF_SAVE_PATH = str(_feed.current)
FRAMES_FOLDER = str(_feed.folder)
extract_frames = _feed.extract
refresh_frames_if_stale = _feed.refresh
augment_current = _feed.augment


def get_all_frames():
    return _feed.refresh(force=True)[0]
