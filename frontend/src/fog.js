import axios from 'axios'

// ── Fog / visibility along a route ────────────────────────────────────────────
// IMD radar only sees precipitation, so fog comes from Open-Meteo's hourly
// visibility forecast (free, keyless, CORS-enabled → fetched straight from the
// browser; nothing touches the 512 MB backend). Each waypoint gets the
// visibility forecast for the hour the driver will actually be there.

const OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
const MAX_POINTS = 80 // keep the multi-location URL short

// Visibility (m) → fog level. null = no fog worth flagging.
export function fogLevel(visM) {
  if (!Number.isFinite(visM)) return null
  if (visM < 200) return 'dense'
  if (visM < 500) return 'thick'
  if (visM < 1000) return 'fog'
  return null
}

export const FOG_COLORS = {
  dense: '#E5E7EB',
  thick: '#CBD5E1',
  fog: '#94A3B8',
}

// ── Satellite fog (Meteosat-9 night fog maps via the backend) ─────────────────
// Backend: POST /fog/satellite → per-waypoint status from the newest precomputed
// fog map (satellite_worker/ → Supabase Storage). The satellite sees fog, not
// visibility, so visM stays NaN and the level encodes how sure we are.
const SAT_LEVEL = { fog: 'thick', patchy: 'fog', approaching: 'fog', forming: 'fog' }

export async function fetchSatelliteFog(apiBase, waypoints, signal) {
  const wps = (waypoints || []).filter(
    (w) => Number.isFinite(Number(w?.lat)) && Number.isFinite(Number(w?.lon)) && Number.isFinite(w?._cumKm),
  )
  if (!wps.length) return { scene: null, points: [] }
  const { data } = await axios.post(
    `${apiBase}/fog/satellite`,
    { waypoints: wps.map((w) => ({ lat: Number(w.lat), lon: Number(w.lon), eta_mins: Number(w.eta_mins) || 0 })) },
    { signal, timeout: 20000 },
  )
  const pts = Array.isArray(data?.points) ? data.points : []
  const points = wps.map((w, i) => {
    const p = pts[i] || {}
    return {
      lat: Number(w.lat),
      lon: Number(w.lon),
      cumKm: w._cumKm,
      etaMins: Number(w.eta_mins) || 0,
      visM: NaN,
      status: p.status || 'no_data',
      level: SAT_LEVEL[p.status] || null,
      distKm: p.dist_km ?? null,
      fogEtaMins: p.fog_eta_mins ?? null,
    }
  }).sort((a, b) => a.cumKm - b.cumKm)
  return { scene: data?.scene || null, points }
}

/** One-line state of the satellite source, or null when it is working normally. */
export function satelliteNote(scene, points, t) {
  const state = scene?.state
  if (!state) return t('Satellite fog unavailable', 'सैटेलाइट कोहरा उपलब्ध नहीं')
  if (state === 'daytime') return t('No fog', 'कोहरा नहीं')   // night-only method; product choice: show "No fog" by day
  if (state === 'stale') return t('Satellite data is delayed', 'सैटेलाइट डेटा देर से है')
  if (state !== 'ok') return t('Satellite fog unavailable', 'सैटेलाइट कोहरा उपलब्ध नहीं')
  const unknown = (points || []).filter((p) => p.status === 'unknown').length
  if (unknown) return t(`Cloud hides ${unknown} point(s) — fog unknown there`, `${unknown} बिंदु बादल से छिपे — वहाँ कोहरा अज्ञात`)
  const forming = (points || []).filter((p) => p.status === 'forming').length
  if (forming) return t(`New fog forming near ${forming} point(s)`, `${forming} बिंदुओं के पास नया कोहरा बन रहा है`)
  return null
}

export function fmtVisibility(visM) {
  if (!Number.isFinite(visM)) return '—'
  return visM >= 1000 ? `${(visM / 1000).toFixed(1)} km` : `${Math.round(visM / 10) * 10} m`
}

/**
 * waypoints: [{ lat, lon, eta_mins, _cumKm }]
 * → [{ lat, lon, cumKm, etaMins, visM, level }] sorted by cumKm (subsampled
 *   to MAX_POINTS). Throws on network failure — callers treat fog as optional.
 */
export async function fetchRouteFog(waypoints, signal) {
  let wps = (waypoints || []).filter(
    (w) => Number.isFinite(Number(w?.lat)) && Number.isFinite(Number(w?.lon)) && Number.isFinite(w?._cumKm),
  )
  if (!wps.length) return []
  if (wps.length > MAX_POINTS) {
    const stride = wps.length / MAX_POINTS
    const picked = []
    for (let i = 0; i < MAX_POINTS; i++) picked.push(wps[Math.floor(i * stride)])
    picked.push(wps[wps.length - 1])
    wps = picked
  }

  const maxEta = Math.max(0, ...wps.map((w) => Number(w.eta_mins) || 0))
  const params = {
    latitude: wps.map((w) => Number(w.lat).toFixed(3)).join(','),
    longitude: wps.map((w) => Number(w.lon).toFixed(3)).join(','),
    hourly: 'visibility',
    forecast_hours: Math.min(48, Math.ceil(maxEta / 60) + 2),
    timezone: 'GMT',
  }
  const { data } = await axios.get(OPEN_METEO_URL, { params, signal, timeout: 20000 })
  const rows = Array.isArray(data) ? data : [data]

  const now = Date.now()
  return wps.map((w, i) => {
    const hourly = rows[i]?.hourly
    const times = hourly?.time || []
    const vis = hourly?.visibility || []
    const target = now + (Number(w.eta_mins) || 0) * 60000
    let best = null
    let bestDiff = Infinity
    for (let k = 0; k < times.length; k++) {
      const diff = Math.abs(Date.parse(`${times[k]}Z`) - target)
      if (diff < bestDiff && Number.isFinite(vis[k])) { bestDiff = diff; best = vis[k] }
    }
    return {
      lat: Number(w.lat),
      lon: Number(w.lon),
      cumKm: w._cumKm,
      etaMins: Number(w.eta_mins) || 0,
      visM: best,
      level: fogLevel(best),
    }
  }).sort((a, b) => a.cumKm - b.cumKm)
}

const LEVEL_RANK = { fog: 1, thick: 2, dense: 3 }

/** Contiguous foggy stretches → [{ startKm, endKm, minVis, level }] */
export function fogZones(fog) {
  const pts = fog || []
  const zones = []
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]
    if (!p.level) continue
    // each point owns the stretch halfway to its neighbours
    const startKm = i > 0 ? (pts[i - 1].cumKm + p.cumKm) / 2 : p.cumKm
    const endKm = i < pts.length - 1 ? (p.cumKm + pts[i + 1].cumKm) / 2 : p.cumKm
    const last = zones[zones.length - 1]
    if (last && Math.abs(last.endKm - startKm) < 1e-6) {
      last.endKm = endKm
      if (p.visM < last.minVis) { last.minVis = p.visM; last.level = p.level }
      else if ((LEVEL_RANK[p.level] || 0) > (LEVEL_RANK[last.level] || 0)) last.level = p.level
    } else {
      zones.push({ startKm, endKm, minVis: p.visM, level: p.level })
    }
  }
  return zones
}
