# AIFS fog worker

ECMWF's AI weather model (AIFS Single, free open data) → fog forecast along a
route at the time the driver gets there. The satellite says where fog **is
now**; AIFS says whether fog is **likely later** (up to 72 h).

```
ECMWF AIFS Single (4 runs/day 00/06/12/18 UTC, 6-hourly steps, 0.25°)
   │  run.py, hourly (GitHub Actions: .github/workflows/fog_aifs.yml), skips runs already done
   ▼
download 2 m temperature + dew point, 10 m wind, low cloud, steps 0..72 h (~60 MB global GRIB)
   ▼
crop to 72–89°E, 23–33°N → per step: dew-point depression, wind speed, low cloud, fog class
   ▼
Supabase Storage bucket "fog-satellite" (public read)
   aifs/latest.json             newest run, fog share per step, the fog rule (thresholds)
   aifs/run_YYYYMMDDHH.npz      dpd, wind, lcc, fog class per step (~50 KB)
   ▼
backend/fog_aifs.py  →  POST /fog/aifs  →  frontend "ECMWF" fog source
```

## Fog rule (first guess, not yet calibrated)

AIFS has no visibility output, so fog is diagnosed from its ingredients:

| Test | Class |
|---|---|
| dew-point depression ≤ 1 K and wind ≤ 3 m/s | **fog** |
| low cloud ≥ 80 % and dpd ≤ 1.5 K and wind ≤ 4 m/s | **fog** |
| dpd ≤ 2 K and wind ≤ 5 m/s | possible |
| otherwise | clear |

The thresholds live in `config.py` and are written into every manifest; the
backend reads them from there, so both sides always agree.

## Per waypoint (backend)

Time the driver reaches the point = now + ETA. The two 6-hourly steps around
that time are blended linearly, the 4 grid cells around the point bilinearly,
then the rule is applied. Statuses: fog · possible · clear · outside (not in the
box) · beyond (after +72 h) · no_data. A run older than 24 h is flagged stale.

## Known limits

- 6-hourly steps (05:30 / 11:30 / 17:30 / 23:30 IST): fog that forms and lifts
  between two steps can be smoothed away. Blending in time helps but cannot
  invent the dawn minimum.
- 0.25° cells (~27 km): no local fog pockets (river banks, low ground).
- The rule is not calibrated yet. Next step: score it against airport METAR
  (the answer key) over many nights and tune the thresholds.
- Only the last ~4 days of runs are on the open-data server.

## Running

```
pip install -r aifs_worker/requirements.txt
python aifs_worker/run.py --local aifs_test --quicklook aifs_test/map.png   # PC test, no Supabase
python -m pytest aifs_worker/test_aifs.py                                    # offline tests
```
