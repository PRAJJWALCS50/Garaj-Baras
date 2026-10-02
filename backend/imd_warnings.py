# Garaj Baras — imd_warnings.py
#
# IMD district-level weather warnings along a route ("IMD warnings on your route").
#
# Source: the WFS layer behind IMD's district warning map
# (mausam.imd.gov.in/responsive/districtWiseWarningGIS.php), served by IMD's
# GeoServer at reactjs.imd.gov.in. One keyless request returns every district
# with 5 days of warnings: per day a comma-separated list of hazard codes
# (Day_N) and a severity colour (DayN_Color). The official
# mausam.imd.gov.in/api/warnings_district_api.php is IP-whitelisted, so we
# use the public map layer instead.
#
# Flow:
#   refresh()        fetch the attributes-only WFS response (~370 KB) and upsert
#                    the North India districts into `imd_district_warnings`
#                    (Postgres in prod via db.py, imd_warnings.db SQLite in dev).
#   route_warnings() map each route waypoint to a district (point-in-polygon on
#                    the bundled data/imd_north_districts.geojson, keyed by the
#                    same IMD district ID), de-duplicate in driving order, and
#                    return today's + tomorrow's hazards for the warned ones.
#
# Districts are not all re-issued on the same day, so "Day 1" is relative to
# each record's own issue Date; we shift by (target date - issue date).

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import requests

import db

DB_PATH = os.path.join(os.path.dirname(__file__), "imd_warnings.db")
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "data", "imd_north_districts.geojson")

WFS_URL = "https://reactjs.imd.gov.in/geoserver/wfs"
WFS_PARAMS = {
    "service": "WFS",
    "version": "1.1.0",
    "request": "GetFeature",
    "typename": "imd:district_warnings_india",
    "outputFormat": "application/json",
    "propertyName": ("ID,Date,UTC,District,Day_1,Day_2,Day_3,Day_4,Day_5,"
                     "Day1_Color,Day2_Color,Day3_Color,Day4_Color,Day5_Color,updated_at"),
}
FETCH_TIMEOUT_SEC = 30

# Re-fetch when the stored copy is older than this (IMD re-issues at 00/06/12 UTC).
STALE_AFTER = timedelta(hours=6)

IST = timezone(timedelta(hours=5, minutes=30))

# Hazard codes, from the getWarning() lookup in the IMD page's own JS.
HAZARDS = {
    1: "No warning",
    2: "Heavy rain",
    3: "Heavy snow",
    4: "Thunderstorm & lightning, squall",
    5: "Hailstorm",
    6: "Dust storm",
    7: "Dust raising winds",
    8: "Strong surface winds",
    9: "Heat wave",
    10: "Hot day",
    11: "Warm night",
    12: "Cold wave",
    13: "Cold day",
    14: "Ground frost",
    15: "Fog",
    16: "Very heavy rain",
    17: "Extremely heavy rain",
}

# DayN_Color values (GeoServer SLD "WarningsDistrictColor"); names follow the
# map legend: red = Warning, orange = Alert, yellow = Watch, green = none.
LEVELS = {
    1: ("warning", "Warning", "#F20505"),
    2: ("alert", "Alert", "#F28705"),
    3: ("watch", "Watch", "#F2E205"),
    4: ("none", "No warning", "#078C03"),
}

# LGD state codes for the default North India set (used for display names;
# the bundled GeoJSON already contains only these states).
STATE_NAMES = {
    1: "Jammu & Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    37: "Ladakh",
}

_db_lock = threading.Lock()
_db_ready = False
_refresh_lock = threading.Lock()
_districts = None          # lazily loaded polygons, see _load_districts()
_districts_lock = threading.Lock()


# ── Database ────────────────────────────────────────────────────────────────

def _conn():
    return db.connect(DB_PATH)


def init_db():
    global _db_ready
    if _db_ready:
        return
    with _db_lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS imd_district_warnings (
                district_id INTEGER PRIMARY KEY,
                district TEXT NOT NULL,
                issue_date TEXT NOT NULL,     -- YYYY-MM-DD; Day_1 refers to this date
                issue_utc INTEGER,            -- issue hour (0/6/12/18 UTC)
                updated_at TEXT,              -- IMD's own timestamp
                days_json TEXT NOT NULL,      -- [{"codes":[..],"color":n}] x 5
                fetched_at TEXT NOT NULL      -- when we stored it (UTC ISO)
            )
        """)
    _db_ready = True


def _last_fetched_at():
    init_db()
    with _conn() as c:
        row = c.execute("SELECT MAX(fetched_at) FROM imd_district_warnings").fetchone()
    if not row or not row[0]:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except ValueError:
        return None


# ── Fetch + store ───────────────────────────────────────────────────────────

def _parse_codes(val):
    codes = []
    for part in str(val or "").split(","):
        part = part.strip()
        if part.isdigit():
            codes.append(int(part))
    return codes


def _parse_feature(props):
    """WFS feature properties → DB row tuple, or None if unusable."""
    try:
        district_id = int(props.get("ID") or 0)
    except (TypeError, ValueError):
        return None
    issue_date = str(props.get("Date") or "")[:10]
    if district_id <= 0 or len(issue_date) != 10:
        return None
    days = []
    for n in range(1, 6):
        try:
            color = int(props.get(f"Day{n}_Color") or 0)
        except (TypeError, ValueError):
            color = 0
        days.append({"codes": _parse_codes(props.get(f"Day_{n}")), "color": color})
    try:
        issue_utc = int(props.get("UTC")) if props.get("UTC") is not None else None
    except (TypeError, ValueError):
        issue_utc = None
    return (district_id, str(props.get("District") or "").strip().title(), issue_date,
            issue_utc, props.get("updated_at"), json.dumps(days))


def refresh() -> dict:
    """Fetch IMD district warnings and upsert the North India districts.
    Returns a small summary; raises on network/parse failure."""
    init_db()
    with _refresh_lock:
        resp = requests.get(WFS_URL, params=WFS_PARAMS, timeout=FETCH_TIMEOUT_SEC)
        resp.raise_for_status()
        features = resp.json().get("features") or []

        wanted = set(_load_districts()["by_id"])
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for f in features:
            row = _parse_feature(f.get("properties") or {})
            if row and row[0] in wanted:
                rows.append(row + (now,))
        if not rows:
            raise RuntimeError(f"IMD returned {len(features)} features but none for North India.")

        with _db_lock, _conn() as c:
            c.executemany("""
                INSERT INTO imd_district_warnings
                    (district_id, district, issue_date, issue_utc, updated_at, days_json, fetched_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(district_id) DO UPDATE SET
                    district=excluded.district, issue_date=excluded.issue_date,
                    issue_utc=excluded.issue_utc, updated_at=excluded.updated_at,
                    days_json=excluded.days_json, fetched_at=excluded.fetched_at
            """, rows)
        return {"ok": True, "features": len(features), "stored": len(rows), "fetched_at": now}


def refresh_bg():
    """Background refresh for the scheduled task; never raises."""
    try:
        print(f"[imd_warnings] refreshed: {refresh()}")
    except Exception as e:  # noqa: BLE001
        print(f"[imd_warnings] refresh failed: {e}")


def ensure_fresh():
    """Make sure we have usable data: fetch synchronously when there is none
    (≈0.5 s), refresh in the background when it is merely stale."""
    last = _last_fetched_at()
    if last is None:
        try:
            refresh()
        except Exception as e:  # noqa: BLE001
            print(f"[imd_warnings] initial fetch failed: {e}")
        return
    if datetime.now(timezone.utc) - last > STALE_AFTER and not _refresh_lock.locked():
        threading.Thread(target=refresh_bg, daemon=True).start()


# ── Districts (point-in-polygon) ────────────────────────────────────────────

def _load_districts():
    """Load the bundled North India district polygons once (~few MB in RAM).
    Each district: id, name, state, bbox, list of polygons (outer + holes)."""
    global _districts
    if _districts is not None:
        return _districts
    with _districts_lock:
        if _districts is not None:
            return _districts
        with open(GEOJSON_PATH, encoding="utf-8") as f:
            gj = json.load(f)
        items = []
        for feat in gj.get("features") or []:
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            try:
                did = int(props.get("ID") or 0)
            except (TypeError, ValueError):
                continue
            if did <= 0 or not geom:
                continue
            if geom.get("type") == "Polygon":
                polys = [geom["coordinates"]]
            elif geom.get("type") == "MultiPolygon":
                polys = geom["coordinates"]
            else:
                continue
            # rings as tuples of (lon, lat)
            polys = [[[(float(p[0]), float(p[1])) for p in ring] for ring in poly] for poly in polys]
            xs = [p[0] for poly in polys for p in poly[0]]
            ys = [p[1] for poly in polys for p in poly[0]]
            if not xs:
                continue
            try:
                state = STATE_NAMES.get(int(props.get("state_lgd") or 0))
            except (TypeError, ValueError):
                state = None
            items.append({
                "id": did,
                "name": str(props.get("District") or "").strip().title(),
                "state": state,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "polys": polys,
            })
        _districts = {"items": items, "by_id": {d["id"]: d for d in items}}
        return _districts


def _in_ring(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _in_district(lon, lat, d):
    x0, y0, x1, y1 = d["bbox"]
    if not (x0 <= lon <= x1 and y0 <= lat <= y1):
        return False
    for poly in d["polys"]:
        if _in_ring(lon, lat, poly[0]) and not any(_in_ring(lon, lat, h) for h in poly[1:]):
            return True
    return False


def district_at(lat, lon, hint=None):
    """District dict containing (lat, lon), or None outside North India.
    `hint` (the previous waypoint's district) is checked first — consecutive
    waypoints are almost always in the same district."""
    if hint is not None and _in_district(lon, lat, hint):
        return hint
    for d in _load_districts()["items"]:
        if d is not hint and _in_district(lon, lat, d):
            return d
    return None


# ── Route lookup ────────────────────────────────────────────────────────────

def _day_info(row, target):
    """Hazards + level for one district on `target` date, or None if the
    stored issue doesn't cover that date."""
    if row is None:
        return None
    try:
        issued = datetime.strptime(row["issue_date"], "%Y-%m-%d").date()
    except ValueError:
        return None
    idx = (target - issued).days
    if not 0 <= idx < len(row["days"]):
        return None
    day = row["days"][idx]
    codes = [c for c in day["codes"] if c != 1 and c in HAZARDS]
    level_key, level_name, color = LEVELS.get(day["color"], ("unknown", "Not issued", "#9CA3AF"))
    return {
        "date": target.isoformat(),
        "level": level_key,
        "level_name": level_name,
        "color": color,
        "hazard_codes": codes,
        "hazards": [HAZARDS[c] for c in codes],
        "warned": bool(codes) or level_key in ("warning", "alert", "watch"),
        "issued": row["issue_date"],
    }


def route_warnings(waypoints) -> dict:
    """waypoints: iterable of (lat, lon, eta_mins).
    → today's + tomorrow's IMD warnings for each unique district on the route
      (driving order), keeping only districts with a warning on either day."""
    ensure_fresh()

    # 1) waypoint → district, unique, in driving order
    order, first_eta, hint, uncovered = [], {}, None, 0
    for lat, lon, eta in waypoints:
        d = district_at(float(lat), float(lon), hint)
        if d is None:
            uncovered += 1
            continue
        hint = d
        if d["id"] not in first_eta:
            first_eta[d["id"]] = float(eta or 0)
            order.append(d)

    # 2) stored warnings for those districts
    rows = {}
    last = None
    if order:
        ids = [d["id"] for d in order]
        marks = ",".join("?" * len(ids))
        with _conn() as c:
            for r in c.execute(
                f"SELECT district_id, issue_date, days_json, fetched_at "
                f"FROM imd_district_warnings WHERE district_id IN ({marks})", ids,
            ):
                rows[int(r[0])] = {"issue_date": r[1], "days": json.loads(r[2])}
                last = max(last or r[3], r[3])

    today = datetime.now(IST).date()
    tomorrow = today + timedelta(days=1)
    districts = []
    for d in order:
        row = rows.get(d["id"])
        t0, t1 = _day_info(row, today), _day_info(row, tomorrow)
        if not ((t0 and t0["warned"]) or (t1 and t1["warned"])):
            continue
        districts.append({
            "id": d["id"],
            "district": d["name"],
            "state": d["state"],
            "first_eta_mins": round(first_eta[d["id"]]),
            "today": t0,
            "tomorrow": t1,
        })

    return {
        "available": bool(rows),
        "source": "IMD district-wise warnings",
        "fetched_at": last,
        "today": today.isoformat(),
        "tomorrow": tomorrow.isoformat(),
        "districts_on_route": len(order),
        "uncovered_waypoints": uncovered,
        "districts": districts,
    }
