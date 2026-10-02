import axios from 'axios'

// ── IMD district warnings along a route ───────────────────────────────────────
// The backend maps each waypoint to its district and returns today's +
// tomorrow's IMD district warnings for the unique districts on the route
// (only those with a warning). Optional like fog: failures are silent.

const MAX_POINTS = 300 // waypoints are ~5 driving minutes apart; plenty for district lookup

export async function fetchRouteImdWarnings(apiBase, waypoints, signal) {
  let wps = (waypoints || []).filter(
    (w) => Number.isFinite(Number(w?.lat)) && Number.isFinite(Number(w?.lon)),
  )
  if (!wps.length) return null
  if (wps.length > MAX_POINTS) {
    const stride = wps.length / MAX_POINTS
    const picked = []
    for (let i = 0; i < MAX_POINTS; i++) picked.push(wps[Math.floor(i * stride)])
    picked.push(wps[wps.length - 1])
    wps = picked
  }
  const { data } = await axios.post(
    `${apiBase}/imd_warnings/route`,
    { waypoints: wps.map((w) => ({ lat: Number(w.lat), lon: Number(w.lon), eta_mins: Number(w.eta_mins) || 0 })) },
    { signal, timeout: 30000 },
  )
  return data
}
