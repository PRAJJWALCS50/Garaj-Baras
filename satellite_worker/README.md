# Satellite fog worker

Turns Meteosat-9 night images into a small fog map of North India every
30 minutes, so the website can show fog along any route instantly.

```
Meteosat-9 (EUMETSAT, every 15 min)
   │  run.py, every 30 min at night (GitHub Actions: .github/workflows/fog_satellite.yml)
   ▼
download newest scan (.nat, whole Earth) → keep 3.9 + 10.8 µm over 72–89°E, 23–33°N (0.05° grid)
   ▼
fogmask.py  per pixel: fog / clear / unknown  →  km to nearest fog  →  fog-edge speed (vs 30 & 60 min ago)
   ▼
Supabase Storage bucket "fog-satellite" (public read)
   latest.json                 newest 3 scenes (time, file, fog share)
   scenes/YYYYMMDD_HHMM.npz    fog, dist_km, speed_kmh, btd, t108   (~100-300 KB)
   ▼
backend/fog_satellite.py  →  POST /fog/satellite  →  frontend "Satellite" fog source
```

## Fog rule (night only)

| Pixel test | Class |
|---|---|
| sun higher than −3° (day / twilight) | unknown |
| T10.8 < 255 K (cold high cloud on top) | unknown |
| T10.8 − T3.9 > 2.5 K | **fog / low stratus** |
| otherwise | clear |

Thresholds are first guesses in `config.py`; tune them against airport METAR.

## Per waypoint (backend)

3×3 pixel window (~15 km): ≥3 fog pixels → **fog**, 1–2 → **patchy**,
>4 unknown → **unknown**. Otherwise, if the nearest fog is ≤50 km away and its
edge is approaching (two 30-min speed estimates agree), fog ETA = distance ÷
speed; if fog arrives before the driver (+15 min) → **approaching**, else clear.
Arrivals more than 3 h ahead are flagged `outlook` (satellite extrapolation is
not trusted that far).

## Known limits

- Night only. Daytime needs a visible-channel method (not built yet).
- Cannot see under high cloud ("unknown", never "clear"). Fog hidden under
  cloud also makes distances/speeds near cloud edges unreliable.
- Sees cloud tops: low stratus over cities (Delhi) can look like ground fog.
- ~5 km pixels: very small patches (a canal, a river crossing) can be missed.
- Bare desert (west Rajasthan) can give a positive BTD on clear nights; the
  threshold must be checked there.

## Running

```bash
pip install -r satellite_worker/requirements.txt
# local test, no Supabase: one historical scene + a picture
EUMETSAT_KEY=... EUMETSAT_SECRET=... \
  python satellite_worker/run.py --local fog_test --time 2025-01-03T22:00 --quicklook fog_test/map.png
# offline tests (no keys needed)
python -m pytest satellite_worker/test_worker.py -q
```

Production needs 4 GitHub repo secrets: `EUMETSAT_KEY`, `EUMETSAT_SECRET`,
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and the bucket from
`supabase_setup.sql`. Without the secrets, scheduled runs exit with
"not configured" (green), so merging this is safe before setup.
