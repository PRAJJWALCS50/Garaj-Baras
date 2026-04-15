# Garaj Baras

Route-based rain prediction for **Delhi-NCR and nearby locations**.

Users enter a **starting point**, **destination**, and **average speed**. The app draws the real road route on a map and **colors the route by predicted rain intensity** (Very Light → Very Heavy). Clicking the route shows a popup with the nearby location name, intensity, dBZ, and ETA.

## What it does
- **Delhi-NCR bounded place search**: typeahead suggestions (free) via OpenStreetMap Nominatim.
- **Road route**: fetches zig-zag driving route geometry from OpenRouteService (ORS).
- **Speed-aware ETAs**: samples waypoints every **5 minutes** based on user speed along the route polyline.
- **Radar-based rain nowcasting**:
  - downloads the IMD Delhi radar GIF
  - extracts frames and estimates rain movement from the **latest frames**
  - predicts if rain will intersect each waypoint at its ETA
- **Route visualization**: route segments are colored by intensity; out-of-radar segments show as **Unknown (gray)**.
- **Effects**: welcome popup (serving area notice) + thunder sound when rain is detected on the route.

## Tech stack

### Frontend
- **React** (Vite)
- **Leaflet + react-leaflet** for map rendering and interactive polylines
- **Axios** for API calls (backend + Nominatim + ORS)
- Static asset serving for thunder audio (`frontend/public/mixkit-thunder-deep-rumble-1296.wav`)

Key files:
- `frontend/src/App.jsx`
- `frontend/src/App.css`

### Backend
- **Python + FastAPI**
- **Pillow (PIL)** for GIF/frame extraction
- **OpenCV + NumPy** for optical flow / rain mask operations
- **Requests** for downloading radar GIF and calling routing services

Key files:
- `backend/main.py` (API)
- `backend/radar.py` (download GIF, extract frames, timestamps, freshness)
- `backend/prediction.py` (ETA/waypoint logic + prediction checks)
- `backend/georef.py` (lat/lon ↔ pixel mapping for radar crop)

### External services / data sources
- **IMD Delhi Radar GIF**: `DELHI_MAXZ.gif` animation feed
- **OpenRouteService**: driving route geometry (requires API key)
- **OpenStreetMap Nominatim**: place search + reverse geocoding (rate-limited)

## API overview
- `POST /predict_waypoints`
  - input: `{ waypoints: [{lat, lon, eta_mins}, ...] }`
  - output: waypoint predictions including `label`, `dbz`, `confidence`, `rain_expected`, `in_radar_bounds`
- `POST /predict`
  - legacy endpoint using backend-driven waypointing
- `GET /frames/latest`, `GET /radar/gif`
  - helper endpoints for radar debug/inspection

## Local development

### Backend
From repo root:
- `cd backend`
- create/activate venv (optional)
- run:
  - `uvicorn main:app --reload --port 8000`

### Frontend
From repo root:
- `cd frontend`
- `npm install`
- set env vars (example in `frontend/.env.example`)
- run:
  - `npm run dev`

### Frontend environment variables
Create `frontend/.env` (or set in Vercel):
- `VITE_ORS_API_KEY` = OpenRouteService key
- `VITE_API_BASE` = backend base URL
  - local: `http://127.0.0.1:8000`
  - render: `https://<your-service>.onrender.com`

## Deployment
- **Backend**: Render (FastAPI service)
- **Frontend**: Vercel (Vite build)
  - Root Directory: `frontend`
  - Build Command: `npm run build`
  - Output Directory: `dist`
  - Env vars: `VITE_API_BASE`, `VITE_ORS_API_KEY`

## Notes
- Nominatim is free but **rate-limited**; the frontend uses debouncing and request canceling for autocomplete.
- Radar OCR timestamps may fail on some environments; the backend falls back to conservative lag estimates when needed.

