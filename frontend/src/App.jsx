import { useMemo, useState } from 'react'
import axios from 'axios'
import L from 'leaflet'
import {
  CircleMarker,
  MapContainer,
  Popup,
  Polyline,
  TileLayer,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import './App.css'

const API_BASE = 'https://garaj-baras-api.onrender.com'
const PREDICT_URL = `${API_BASE}/predict`

const NOMINATIM_SEARCH_URL = 'https://nominatim.openstreetmap.org/search'
const NOMINATIM_REVERSE_URL = 'https://nominatim.openstreetmap.org/reverse'
const NOMINATIM_USER_AGENT = 'GarajBaras/1.0'

const ORS_KEY = import.meta.env.VITE_ORS_API_KEY

function getRainColor(label) {
  const l = String(label || '')
  if (l === 'No Rain') return '#22D3EE'
  if (l.includes('Very Light')) return '#7DD3FC'
  if (l.includes('Light')) return '#38BDF8'
  if (l.includes('Moderate')) return '#0EA5E9'
  if (l.includes('Heavy') && !l.includes('Very Heavy')) return '#F59E0B'
  if (l.includes('Very Heavy')) return '#EF4444'
  return '#22D3EE'
}

function getRainGroupLabel(label) {
  const l = String(label || '')
  if (l === 'No Rain') return 'Clear'
  if (l.includes('Very Light') || l.includes('Light')) return 'Light'
  if (l.includes('Moderate')) return 'Moderate'
  if (l.includes('Heavy') || l.includes('Very Heavy')) return 'Heavy'
  return 'Clear'
}

function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2
  return R * 2 * Math.asin(Math.sqrt(a))
}

function toCityRouteName(a, b) {
  const left = (a ?? '').trim() || 'Source'
  const right = (b ?? '').trim() || 'Destination'
  return `${left} → ${right}`
}

function extractCityAndName(addr) {
  const name =
    addr?.suburb ||
    addr?.village ||
    addr?.town ||
    addr?.city_district ||
    addr?.county ||
    'Unknown'
  const city = addr?.city || addr?.state_district || addr?.town || ''
  const display = city ? `${name}, ${city}` : name
  return { name, city, display }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function geocode(place) {
  const q = String(place ?? '').trim()
  if (!q) throw new Error('Please enter both Source and Destination.')

  const url = `${NOMINATIM_SEARCH_URL}?q=${encodeURIComponent(
    q
  )}&format=json&limit=1&countrycodes=in`

  try {
    const res = await axios.get(url, {
      timeout: 15000,
      headers: { 'User-Agent': NOMINATIM_USER_AGENT, Accept: 'application/json' },
    })
    const data = Array.isArray(res.data) ? res.data[0] : null
    if (!data?.lat || !data?.lon) throw new Error('No geocoding results.')
    return { lat: parseFloat(data.lat), lon: parseFloat(data.lon), display_name: data.display_name }
  } catch (e) {
    const res2 = await axios.get(url, {
      timeout: 15000,
      headers: { Accept: 'application/json' },
    })
    const data2 = Array.isArray(res2.data) ? res2.data[0] : null
    if (!data2?.lat || !data2?.lon) throw new Error('No geocoding results.')
    return { lat: parseFloat(data2.lat), lon: parseFloat(data2.lon), display_name: data2.display_name }
  }
}

async function reverseGeocode(lat, lon) {
  const url = `${NOMINATIM_REVERSE_URL}?lat=${lat}&lon=${lon}&format=json`

  try {
    const res = await axios.get(url, {
      timeout: 15000,
      headers: { 'User-Agent': NOMINATIM_USER_AGENT, Accept: 'application/json' },
    })
    const addr = res.data?.address || null
    return extractCityAndName(addr)
  } catch (e) {
    const res2 = await axios.get(url, {
      timeout: 15000,
      headers: { Accept: 'application/json' },
    })
    const addr2 = res2.data?.address || null
    return extractCityAndName(addr2)
  }
}

function sampleRouteEvery2km(routeCoords, stepKm = 2.0) {
  // routeCoords: Array of [lon, lat]
  if (!Array.isArray(routeCoords) || routeCoords.length < 2) return []

  const sampled = []
  sampled.push(routeCoords[0])

  let distAcc = 0
  for (let i = 1; i < routeCoords.length; i++) {
    const [lon1, lat1] = routeCoords[i - 1]
    const [lon2, lat2] = routeCoords[i]
    distAcc += haversine(lat1, lon1, lat2, lon2)
    if (distAcc >= stepKm) {
      sampled.push(routeCoords[i])
      distAcc = 0
    }
  }

  const last = routeCoords[routeCoords.length - 1]
  const lastKey = `${last[0]}|${last[1]}`
  const lastSampled = sampled[sampled.length - 1]
  const lastSampledKey = `${lastSampled[0]}|${lastSampled[1]}`
  if (lastKey !== lastSampledKey) sampled.push(last)

  // Convert to {lat,lon} objects
  return sampled.map(([lon, lat]) => ({ lat, lon }))
}

export default function App() {
  const [source, setSource] = useState('')
  const [destination, setDestination] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [routeCoords, setRouteCoords] = useState([]) // [lat, lon] full road polyline
  const [waypointLocations, setWaypointLocations] = useState([]) // {name, city, display} aligned to sampled points
  const [error, setError] = useState(null)

  const routeName = useMemo(
    () => toCityRouteName(source, destination),
    [source, destination]
  )

  async function handlePredict() {
    const startCity = source.trim()
    const endCity = destination.trim()

    if (!startCity || !endCity) {
      setError('Please enter both Source and Destination city names.')
      return
    }
    if (!ORS_KEY) {
      setError('Missing ORS API key. Set `VITE_ORS_API_KEY` in frontend/.env.')
      return
    }

    setError(null)
    setLoading(true)
    setResult(null)
    setRouteCoords([])
    setWaypointLocations([])

    try {
      // 1) Geocode
      const start = await geocode(startCity)
      const end = await geocode(endCity)

      // 2) Get road route from ORS
      let routeLonLat = null
      try {
        const orsRes = await axios.post(
          'https://api.openrouteservice.org/v2/directions/driving-car/geojson',
          {
            coordinates: [
              [start.lon, start.lat],
              [end.lon, end.lat],
            ],
            // Helps when geocoded points are slightly away from roads.
            radiuses: [5000, 5000],
          },
          {
            timeout: 60000,
            headers: {
              Authorization: ORS_KEY,
              'Content-Type': 'application/json',
            },
          }
        )
        routeLonLat = orsRes.data?.features?.[0]?.geometry?.coordinates || null
      } catch (orsError) {
        // ORS can return 404 "Could not find routable point..." for city centers.
        // Fallback to a straight segment so prediction can still proceed.
        routeLonLat = [
          [start.lon, start.lat],
          [end.lon, end.lat],
        ]
      }

      if (!Array.isArray(routeLonLat) || routeLonLat.length < 2) {
        throw new Error('Route planning failed (ORS returned empty geometry).')
      }

      // 3) Sample points every ~2km
      const sampled = sampleRouteEvery2km(routeLonLat, 2.0) // [{lat,lon}, ...]

      // 4) Reverse geocode sampled points (max 15) with 200ms spacing
      const MAX_REVERSE = 15
      const sampledForReverse = sampled.slice(0, MAX_REVERSE)
      const locations = []
      for (let i = 0; i < sampledForReverse.length; i++) {
        const p = sampledForReverse[i]
        // Rate limit friendly delay
        // eslint-disable-next-line no-await-in-loop
        await sleep(200)
        // eslint-disable-next-line no-await-in-loop
        const addr = await reverseGeocode(p.lat, p.lon)
        locations.push(addr)
      }

      // 5) Call predict API
      const payload = {
        start_lat: sampled[0].lat,
        start_lon: sampled[0].lon,
        end_lat: sampled[sampled.length - 1].lat,
        end_lon: sampled[sampled.length - 1].lon,
        spacing_km: 2.0,
      }
      const predictRes = await axios.post(PREDICT_URL, payload, {
        timeout: 90000,
      })

      const predictWaypoints = Array.isArray(predictRes.data?.waypoints)
        ? predictRes.data.waypoints
        : []

      // 6) Prepare map polyline (full road)
      const polylineLatLon = routeLonLat.map(([lon, lat]) => [lat, lon])
      setRouteCoords(polylineLatLon)

      // 7) Merge results with limited reverse-geocoded names
      // We align by waypoint index (both are intended to be ~2km spaced).
      const mergedWaypoints = predictWaypoints.map((wp, idx) => {
        const loc = locations[idx] || null
        const locationName = loc?.name || 'Unknown'
        const city = loc?.city || ''
        const locationDisplay = loc?.display || 'Unknown'

        return {
          ...wp,
          locationName,
          city,
          locationDisplay,
          rainGroup: getRainGroupLabel(wp.label),
          rainColor: getRainColor(wp.label),
        }
      })

      setWaypointLocations(locations)
      setResult({
        ...predictRes.data,
        waypoints: mergedWaypoints.slice(0, MAX_REVERSE),
      })
    } catch (e) {
      const status = e?.response?.status
      const detail =
        e?.response?.data?.detail ||
        e?.response?.data?.error?.message ||
        e?.response?.data?.message
      const msg = status
        ? `Request failed (${status}): ${detail || e?.message || 'Unknown error'}`
        : e?.message || 'Something went wrong while scanning the radar.'
      setError(typeof msg === 'string' ? msg : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const hasRain = !!result && (result.rain_waypoints ?? 0) > 0
  const midpoint = routeCoords.length
    ? routeCoords[Math.floor(routeCoords.length / 2)]
    : [26.7606, 80.8893]

  const routeBounds = useMemo(() => {
    if (!routeCoords || routeCoords.length < 2) return null
    return L.latLngBounds(routeCoords)
  }, [routeCoords])

  return (
    <div className="app">
      <header className="topbar">
        <div className="menuButton" aria-label="Menu" role="button" tabIndex={0}>
          ☰
        </div>
        <div className="brandTitle">GARAJ BARAS</div>
        <div style={{ width: 36 }} />
      </header>

      <div className="screen">
        {/* Hero + Inputs */}
        {!loading && !result && (
          <>
            <section className="heroBlock">
              <h1 className="heroTitle">Where to next?</h1>
              <div className="heroSubtitle">Real-time rain prediction on your route</div>
            </section>

            <section className="card plannerCard">
              <div className="fieldGrid">
                <div className="field">
                  <div className="fieldLabel">Source</div>
                  <div className="inputShell">
                    <span className="glowDot glowDot--source" aria-hidden="true" />
                    <input
                      className="routeInput"
                      placeholder="e.g., Lucknow"
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                    />
                  </div>
                </div>

                <button
                  className="swapButton"
                  type="button"
                  onClick={() => {
                    const tmp = source
                    setSource(destination)
                    setDestination(tmp)
                    setError(null)
                    setResult(null)
                  }}
                >
                  SWAP
                </button>

                <div className="field">
                  <div className="fieldLabel">Destination</div>
                  <div className="inputShell">
                    <span className="glowDot glowDot--destination" aria-hidden="true" />
                    <input
                      className="routeInput"
                      placeholder="e.g., Fatehpur"
                      value={destination}
                      onChange={(e) => setDestination(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              <button
                className="planButton"
                type="button"
                onClick={handlePredict}
                disabled={!source.trim() || !destination.trim() || loading}
                aria-label="Plan My Route"
              >
                Plan My Route →
              </button>

              <div className="dividerMini" />

              <div className="radarStatus">
                <div className="radarLeft">
                  <span className="radarDot" aria-hidden="true" />
                  <div>
                    <div className="radarText">Lucknow DWR • Active</div>
                    <div className="radarSub">Realtime scanning feed</div>
                  </div>
                </div>
              </div>
            </section>

            {error && (
              <section className="errorCard" role="alert" aria-live="polite">
                <div className="errorTitle">Scan failed</div>
                <div className="errorBody">{error}</div>
              </section>
            )}
          </>
        )}

        {/* Loading */}
        {loading && (
          <section className="card loadingCard" aria-live="polite">
            <div className="spinner" aria-hidden="true" />
            <div className="loadingText">SCANNING RADAR...</div>
          </section>
        )}

        {/* Results */}
        {result && (
          <section className="resultsWrap">
            <section className="card resultsCard">
              <div className="summaryTop">
                <div>
                  <div className="routeName">{routeName}</div>
                </div>

                <div
                  className={`rainBadge ${
                    hasRain ? 'rainBadge--rain' : 'rainBadge--clear'
                  }`}
                >
                  {hasRain ? 'Rain' : 'Clear'}
                </div>
              </div>

              <div className="statsRow">
                <div className="statBox">
                  <div className="statLabel">Distance</div>
                  <div className="statValue">
                    {Number(result.route_distance_km).toFixed(1)} km
                  </div>
                </div>
                <div className="statBox">
                  <div className="statLabel">First Rain ETA</div>
                  <div className="statValue">
                    {result.first_rain_eta == null
                      ? 'N/A'
                      : `${Math.round(result.first_rain_eta)} mins`}
                  </div>
                </div>
                <div className="statBox">
                  <div className="statLabel">Rain Speed</div>
                  <div className="statValue">
                    {Number(result.rain_speed_kmh).toFixed(1)} km/h
                  </div>
                </div>
              </div>

              <div className="directionRow">
                <div>
                  <div className="directionText">
                    Rain direction: {result.rain_direction_from} -&gt;{' '}
                    {result.rain_direction_to}
                  </div>
                  <div className="directionSub">{result.radar_freshness}</div>
                </div>
              </div>

              {/* Map */}
              <div className="map-container">
                <MapContainer
                  center={midpoint}
                  zoom={9}
                  style={{ height: '320px', width: '100%' }}
                  zoomControl={false}
                  scrollWheelZoom={false}
                  whenCreated={(map) => {
                    if (routeBounds && routeBounds.isValid()) {
                      map.fitBounds(routeBounds, { padding: [20, 20] })
                    }
                  }}
                >
                  <TileLayer
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    attribution="CartoDB"
                  />

                  {routeCoords.length > 0 && (
                    <Polyline
                      positions={routeCoords}
                      color="#0EA5E9"
                      weight={4}
                      opacity={0.8}
                    />
                  )}

                  {Array.isArray(result.waypoints) &&
                    result.waypoints.map((wp, idx) => (
                      <CircleMarker
                        key={`${idx}-${wp.lat}-${wp.lon}`}
                        center={[wp.lat, wp.lon]}
                        radius={8}
                        fillColor={wp.rainColor || getRainColor(wp.label)}
                        color="#FFFFFF"
                        weight={2}
                        fillOpacity={0.9}
                      >
                        <Popup>
                          <div
                            style={{
                              background: '#061529',
                              color: 'white',
                              padding: '8px',
                              borderRadius: '8px',
                              minWidth: '170px',
                            }}
                          >
                            <div style={{ fontWeight: 800, fontSize: 13 }}>
                              {wp.locationName || 'Unknown'}
                              {wp.city ? `, ${wp.city}` : ''}
                            </div>
                            <div
                              style={{
                                color: wp.rainColor || getRainColor(wp.label),
                                fontSize: '13px',
                                marginTop: '4px',
                                fontWeight: 700,
                              }}
                            >
                              {wp.label || 'No Rain'}
                            </div>
                            <div style={{ marginTop: 6, fontSize: 12, opacity: 0.9 }}>
                              ETA {Math.round(Number(wp.eta_mins))} mins
                            </div>
                          </div>
                        </Popup>
                      </CircleMarker>
                    ))}
                </MapContainer>
              </div>

              {/* Waypoint list */}
              <div className="waypointList">
                {Array.isArray(result.waypoints) && result.waypoints.length > 0 ? (
                  result.waypoints.map((wp, idx) => (
                    <div
                      key={`${idx}-${wp.lat}-${wp.lon}`}
                      className={`waypointItem ${wp.rain_expected ? 'has-rain' : ''}`}
                    >
                      <div
                        className="waypointDot"
                        style={{
                          background: wp.rainColor || getRainColor(wp.label),
                          boxShadow: wp.rain_expected
                            ? `0 0 24px ${wp.rainColor || getRainColor(wp.label)}`
                            : 'none',
                        }}
                        aria-hidden="true"
                      />
                      <div className="waypointMeta">
                        <div className="waypointLabel">
                          {wp.locationName || 'Unknown'}
                          {wp.city ? `, ${wp.city}` : ''}
                        </div>
                        <div className="waypointEta">{wp.rainGroup || 'Clear'}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="errorBody">
                    No waypoints returned for this route. Try another city pair.
                  </div>
                )}
              </div>
            </section>
          </section>
        )}
      </div>
    </div>
  )
}
