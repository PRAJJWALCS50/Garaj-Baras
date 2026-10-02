"""Use actual MAX_Z; advertised MBL_MAXZ animation can contain MAX_V.

Validated Z animation frames are used if IMD fixes the feed. Otherwise collect
six real MAX_Z observations over requests/alert sweeps, persisting on disk.
"""
from native_radar import NativeRadarFeed
from radar import RADAR_TTL_SEC, get_radar_lag_mins

_feed = NativeRadarFeed('mahabaleshwar', 'MBL', 'https://mausam.imd.gov.in/Radar/MAX_Z_mbl.gif')
GIF_URL = _feed.gif_url
GIF_SAVE_PATH = str(_feed.current)
FRAMES_FOLDER = str(_feed.folder)
extract_frames = _feed.extract
refresh_frames_if_stale = _feed.refresh
augment_current = _feed.augment


def get_all_frames():
    return _feed.refresh(force=True)[0]
