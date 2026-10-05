# Native IMD radar regression snapshots

Captured from official IMD sources on 2026-10-02; retained as PNG snapshots for
deterministic, offline regression tests (not runtime radar data).

- `sohra.png`: last distinct frame from
  https://mausam.imd.gov.in/Radar/animation/Converted/CPJ_MAXZ.gif,
  10:48:38 UTC, 1078×770; header gives 25.2680 N, 91.7332 E and 240 km range.
- `mahabaleshwar.png`: https://mausam.imd.gov.in/Radar/MAX_Z_mbl.gif,
  11:30:09 UTC, 720×720, MAX_Z / Reflectivity in dBZ / 170 km.
- `mahabaleshwar_velocity.png`: first frame from
  https://mausam.imd.gov.in/Radar/animation/Converted/MBL_MAXZ.gif,
  06:50:09 UTC. Despite the URL it is MAX_V / Mean Velocity in m/s.

Mahabaleshwar site coordinates are from IMD's radar network map at
https://mausam.imd.gov.in/responsive/radar.php?id=Mahabaleshwar.
Native ring centers/radii and square-panel borders calibrate the AEQD model;
town/airport markers provide a secondary visual cross-check (IMD label positions
are not precise ground-control points). Sohra source estimates: Guwahati
(320,358), Tezpur (442,292), Agartala (267,654); Mahabaleshwar source estimates:
Pune city (254,324), Satara (277,449), Ratnagiri (177,551), Kolhapur (309,593).
Processed pixels use 0.877 km/px; errors from integer sampling are <=0.62 km.

Native colour intervals come from each image's own legend. In particular Sohra
white represents 34.19–35.81 dBZ. It cannot use the old white=60 mapping. The
adapter preserves the existing intensity categories, with the engine's 20 dBZ
floor and ten-level quantization; no pixels from MAX_V or accumulated rainfall
are sent to rain predictions.

- `mangaluru.png`: selected original pixels from
  https://mausam.imd.gov.in/Radar/caz_mlr.gif captured 2026-10-05,
  23:30:01 UTC on **2026-10-04**. Retains header `(1080,0,1310,115)`,
  legend `(1160,345,1230,835)`, and rain sample `(380,580,560,760)`
  on a black 1310×1080 canvas. The 250 km ring is 440 source pixels;
  white is 41.3 dBZ. Runtime code reads the explicit printed date.
