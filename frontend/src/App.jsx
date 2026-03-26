import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import L from 'leaflet'
import {
  MapContainer,
  Popup,
  Polyline,
  TileLayer,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import './App.css'

const API_BASE =
  import.meta.env.VITE_API_BASE || 'https://garaj-baras-api.onrender.com'
const PREDICT_WAYPOINTS_URL = `${API_BASE}/predict_waypoints`

const NOMINATIM_SEARCH_URL = 'https://nominatim.openstreetmap.org/search'
const NOMINATIM_REVERSE_URL = 'https://nominatim.openstreetmap.org/reverse'
const NOMINATIM_USER_AGENT = 'GarajBaras/1.0'

const ORS_KEY = import.meta.env.VITE_ORS_API_KEY

// Rough Delhi NCR bounding box (tune anytime): left,top,right,bottom (lon,lat,lon,lat)
const NCR_VIEWBOX = '76.5,29.7,78.9,28.0'

function getRainColor(label) {
  const l = String(label || '')
  if (l === 'No Rain') return '#FFFFFF'
  if (l.includes('Very Light')) return '#7DD3FC'
  if (l.includes('Light')) return '#38BDF8'
  if (l.includes('Moderate')) return '#0EA5E9'
  if (l.includes('Heavy') && !l.includes('Very Heavy')) return '#F59E0B'
  if (l.includes('Very Heavy')) return '#EF4444'
  return '#FFFFFF'
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

async function searchPlaces(query, signal) {
  const q = String(query ?? '').trim()
  if (!q) return []

  const res = await axios.get(NOMINATIM_SEARCH_URL, {
    timeout: 15000,
    signal,
    params: {
      q,
      format: 'jsonv2',
      limit: 6,
      addressdetails: 1,
      bounded: 1,
      viewbox: NCR_VIEWBOX,
      countrycodes: 'in',
    },
    headers: { 'User-Agent': NOMINATIM_USER_AGENT, Accept: 'application/json' },
  })

  const arr = Array.isArray(res.data) ? res.data : []
  return arr
    .filter((x) => x?.lat && x?.lon && x?.display_name)
    .map((x) => ({
      id: String(x.place_id ?? x.osm_id ?? x.display_name),
      display_name: String(x.display_name),
      lat: Number(x.lat),
      lon: Number(x.lon),
      type: x.type ? String(x.type) : '',
    }))
    .filter((x) => Number.isFinite(x.lat) && Number.isFinite(x.lon))
}

async function reversePlaceName(lat, lon, signal) {
  const res = await axios.get(NOMINATIM_REVERSE_URL, {
    timeout: 15000,
    signal,
    params: {
      lat,
      lon,
      format: 'jsonv2',
      zoom: 16,
      addressdetails: 1,
    },
    headers: { 'User-Agent': NOMINATIM_USER_AGENT, Accept: 'application/json' },
  })
  const name = res.data?.display_name
  return typeof name === 'string' && name.trim() ? name.trim() : null
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

function sampleRouteEvery5Min(routeCoords, speedKmh, intervalMin = 5) {
  // routeCoords: Array of [lon, lat]
  if (!Array.isArray(routeCoords) || routeCoords.length < 2) return []
  const v = Number(speedKmh)
  if (!Number.isFinite(v) || v <= 0) return []

  const stepKm = v * (intervalMin / 60)
  const sampled = []

  let cumKm = 0
  let distAcc = 0

  // always start
  const [lon0, lat0] = routeCoords[0]
  sampled.push({ lat: lat0, lon: lon0, eta_mins: 0, cumKm: 0 })

  for (let i = 1; i < routeCoords.length; i++) {
    const [lon1, lat1] = routeCoords[i - 1]
    const [lon2, lat2] = routeCoords[i]
    const dKm = haversine(lat1, lon1, lat2, lon2)
    cumKm += dKm
    distAcc += dKm

    if (distAcc >= stepKm) {
      const eta = (cumKm / v) * 60
      sampled.push({ lat: lat2, lon: lon2, eta_mins: eta, cumKm })
      distAcc = 0
    }
  }

  const [lonLast, latLast] = routeCoords[routeCoords.length - 1]
  const last = sampled[sampled.length - 1]
  if (!last || Math.abs(last.lat - latLast) > 1e-9 || Math.abs(last.lon - lonLast) > 1e-9) {
    const eta = (cumKm / v) * 60
    sampled.push({ lat: latLast, lon: lonLast, eta_mins: eta, cumKm })
  }

  return sampled
}

function binarySearchNearestIndex(sortedNums, target) {
  if (!Array.isArray(sortedNums) || !sortedNums.length) return -1
  let lo = 0
  let hi = sortedNums.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const v = sortedNums[mid]
    if (v === target) return mid
    if (v < target) lo = mid + 1
    else hi = mid - 1
  }
  if (lo <= 0) return 0
  if (lo >= sortedNums.length) return sortedNums.length - 1
  const a = sortedNums[lo - 1]
  const b = sortedNums[lo]
  return Math.abs(a - target) <= Math.abs(b - target) ? lo - 1 : lo
}

function buildColoredSegments(routeLonLat, predictedWaypoints) {
  // routeLonLat: Array<[lon,lat]>
  if (!Array.isArray(routeLonLat) || routeLonLat.length < 2) return []
  if (!Array.isArray(predictedWaypoints) || !predictedWaypoints.length) return []

  const cumKm = [0]
  let acc = 0
  for (let i = 1; i < routeLonLat.length; i++) {
    const [lon1, lat1] = routeLonLat[i - 1]
    const [lon2, lat2] = routeLonLat[i]
    acc += haversine(lat1, lon1, lat2, lon2)
    cumKm.push(acc)
  }

  const wpEta = predictedWaypoints.map((w) => Number(w?.eta_mins || 0))
  const maxEta = Math.max(...wpEta, 0)
  const totalKm = cumKm[cumKm.length - 1] || 1e-6
  const wpKm = predictedWaypoints.map((w) => {
    const eta = Number(w?.eta_mins || 0)
    const frac = maxEta > 0 ? Math.max(0, Math.min(1, eta / maxEta)) : 0
    return frac * totalKm
  })

  const segments = []
  for (let i = 1; i < routeLonLat.length; i++) {
    const midKm = (cumKm[i - 1] + cumKm[i]) / 2
    const idx = binarySearchNearestIndex(wpKm, midKm)
    const wp = idx >= 0 ? predictedWaypoints[idx] : null
    const inBounds = !!wp?.in_radar_bounds
    const label = wp?.label || 'Unknown'
    const color = inBounds ? getRainColor(label) : '#64748B'
    const midLat = (routeLonLat[i - 1][1] + routeLonLat[i][1]) / 2
    const midLon = (routeLonLat[i - 1][0] + routeLonLat[i][0]) / 2
    segments.push({
      positions: [
        [routeLonLat[i - 1][1], routeLonLat[i - 1][0]],
        [routeLonLat[i][1], routeLonLat[i][0]],
      ],
      color,
      inBounds,
      label,
      eta_mins: wp?.eta_mins ?? null,
      dbz: wp?.dbz ?? null,
      rain_expected: !!wp?.rain_expected,
      mid: { lat: midLat, lon: midLon },
    })
  }
  return segments
}

function joinApiUrl(maybePath) {
  const p = String(maybePath || '')
  if (!p) return ''
  if (p.startsWith('http://') || p.startsWith('https://')) return p
  if (p.startsWith('/')) return `${API_BASE}${p}`
  return `${API_BASE}/${p}`
}

export default function App() {
  const [showWelcomePopup, setShowWelcomePopup] = useState(true)
  const [source, setSource] = useState('')
  const [destination, setDestination] = useState('')
  const [avgSpeedKmh, setAvgSpeedKmh] = useState('')
  const [sourcePlace, setSourcePlace] = useState(null) // {lat,lon,display_name}
  const [destPlace, setDestPlace] = useState(null) // {lat,lon,display_name}

  const [sourceSug, setSourceSug] = useState([])
  const [destSug, setDestSug] = useState([])
  const [sourceOpen, setSourceOpen] = useState(false)
  const [destOpen, setDestOpen] = useState(false)
  const sourceDebounceRef = useRef(null)
  const destDebounceRef = useRef(null)
  const sourceAbortRef = useRef(null)
  const destAbortRef = useRef(null)

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [routeCoords, setRouteCoords] = useState([]) // [lat, lon] full road polyline
  const [routeSegments, setRouteSegments] = useState([]) // colored line segments
  const [activeSeg, setActiveSeg] = useState(null) // {lat,lon,label,dbz,eta_mins,rain_expected,inBounds,locationName}
  const [routeDistanceKm, setRouteDistanceKm] = useState(null)
  const [error, setError] = useState(null)
  const [mapRef, setMapRef] = useState(null)

  const reverseAbortRef = useRef(null)
  const reverseCacheRef = useRef(new Map())

  const routeName = useMemo(
    () => toCityRouteName(source, destination),
    [source, destination]
  )

  // Autocomplete: Source
  useEffect(() => {
    const q = String(source || '').trim()
    if (sourceAbortRef.current) sourceAbortRef.current.abort()
    if (sourceDebounceRef.current) clearTimeout(sourceDebounceRef.current)

    if (q.length < 3) {
      setSourceSug([])
      return
    }

    sourceDebounceRef.current = setTimeout(async () => {
      const ac = new AbortController()
      sourceAbortRef.current = ac
      try {
        const items = await searchPlaces(q, ac.signal)
        setSourceSug(items)
      } catch (e) {
        if (e?.name === 'CanceledError' || e?.name === 'AbortError') return
        setSourceSug([])
      }
    }, 350)

    return () => {
      if (sourceDebounceRef.current) clearTimeout(sourceDebounceRef.current)
    }
  }, [source])

  // Autocomplete: Destination
  useEffect(() => {
    const q = String(destination || '').trim()
    if (destAbortRef.current) destAbortRef.current.abort()
    if (destDebounceRef.current) clearTimeout(destDebounceRef.current)

    if (q.length < 3) {
      setDestSug([])
      return
    }

    destDebounceRef.current = setTimeout(async () => {
      const ac = new AbortController()
      destAbortRef.current = ac
      try {
        const items = await searchPlaces(q, ac.signal)
        setDestSug(items)
      } catch (e) {
        if (e?.name === 'CanceledError' || e?.name === 'AbortError') return
        setDestSug([])
      }
    }, 350)

    return () => {
      if (destDebounceRef.current) clearTimeout(destDebounceRef.current)
    }
  }, [destination])

  async function handlePredict() {
    const startCity = source.trim()
    const endCity = destination.trim()
    const speedNum = Number(avgSpeedKmh)

    if (!startCity || !endCity) {
      setError('Please enter both Source and Destination city names.')
      return
    }
    if (!Number.isFinite(speedNum) || speedNum <= 0) {
      setError('Please enter a valid average speed (km/h).')
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
    setRouteSegments([])
    setRouteDistanceKm(null)

    try {
      // If user typed but didn't pick from dropdown, fall back to text-geocode.
      // (We don't hard-require dropdown selection.)
      // 1) Geocode
      const start = sourcePlace || (await geocode(startCity))
      const end = destPlace || (await geocode(endCity))

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

      // Total distance from the actual road geometry (used for UI stats)
      let totalKm = 0
      for (let i = 1; i < routeLonLat.length; i++) {
        const [lon1, lat1] = routeLonLat[i - 1]
        const [lon2, lat2] = routeLonLat[i]
        totalKm += haversine(lat1, lon1, lat2, lon2)
      }
      setRouteDistanceKm(totalKm)

      // 3) Sample waypoints every 5 minutes by constant avg speed
      const sampled = sampleRouteEvery5Min(routeLonLat, speedNum, 5) // [{lat,lon,eta_mins,cumKm}, ...]
      if (!sampled.length) {
        throw new Error('Could not sample route into waypoints. Try a different route or speed.')
      }

      // 4) Prepare map polyline (full road)
      const polylineLatLon = routeLonLat.map(([lon, lat]) => [lat, lon])
      setRouteCoords(polylineLatLon)

      // 5) Call predict-waypoints API (ETA driven by frontend; for now approximate from distance/speed)
      const cumWaypoints = sampled.map(({ lat, lon, eta_mins }) => ({
        lat,
        lon,
        eta_mins,
      }))

      const predictRes = await axios.post(
        PREDICT_WAYPOINTS_URL,
        { waypoints: cumWaypoints },
        { timeout: 90000 }
      )

      const predictWaypoints = Array.isArray(predictRes.data?.waypoints)
        ? predictRes.data.waypoints
        : []

      const mergedWaypoints = predictWaypoints.map((wp) => ({
        ...wp,
        rainGroup: getRainGroupLabel(wp.label),
        rainColor: getRainColor(wp.label),
      }))

      const segments = buildColoredSegments(routeLonLat, mergedWaypoints)
      setRouteSegments(segments)

      setResult({
        ...predictRes.data,
        route_distance_km: totalKm,
        waypoints: mergedWaypoints,
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

  async function openSegmentPopup(seg) {
    if (!seg?.mid) return
    const lat = Number(seg.mid.lat)
    const lon = Number(seg.mid.lon)
    const key = `${lat.toFixed(4)},${lon.toFixed(4)}`

    setActiveSeg({
      ...seg,
      locationName: reverseCacheRef.current.get(key) || null,
    })

    if (reverseCacheRef.current.has(key)) return

    if (reverseAbortRef.current) reverseAbortRef.current.abort()
    const ac = new AbortController()
    reverseAbortRef.current = ac
    try {
      const name = await reversePlaceName(lat, lon, ac.signal)
      if (name) reverseCacheRef.current.set(key, name)
      setActiveSeg((prev) => {
        if (!prev) return prev
        return { ...prev, locationName: name }
      })
    } catch (e) {
      // ignore abort/network errors; popup will still show lat/lon + intensity
    }
  }

  const hasRain = !!result && (result.rain_waypoints ?? 0) > 0
  const shownDistanceKm = Number.isFinite(Number(routeDistanceKm))
    ? Number(routeDistanceKm)
    : Number.isFinite(Number(result?.route_distance_km))
      ? Number(result.route_distance_km)
      : null
  const midpoint = routeCoords.length
    ? routeCoords[Math.floor(routeCoords.length / 2)]
    : [26.7606, 80.8893]

  const routeBounds = useMemo(() => {
    if (!routeCoords || routeCoords.length < 2) return null
    return L.latLngBounds(routeCoords)
  }, [routeCoords])

  useEffect(() => {
    if (!mapRef || !routeBounds || !routeBounds.isValid()) return
    // Leaflet sometimes computes a too-wide zoom on first render because the container
    // hasn't finalized layout yet. Invalidate + refit fixes the initial view.
    const doFit = () => {
      mapRef.invalidateSize()
      mapRef.fitBounds(routeBounds, {
        padding: [10, 10],
        maxZoom: 17,
        animate: true,
      })
    }
    doFit()
    const t = setTimeout(doFit, 120)
    return () => clearTimeout(t)
  }, [mapRef, routeBounds])

  function handleBackToPlanner() {
    setResult(null)
    setError(null)
    setActiveSeg(null)
    setRouteCoords([])
    setRouteSegments([])
    setRouteDistanceKm(null)
  }

  return (
    <div className="app">
      {showWelcomePopup && (
        <div className="welcomeOverlay" role="dialog" aria-modal="true" aria-label="Service area notice">
          <div className="welcomeCard">
            <div className="welcomeTitle">Garaj Baras</div>
            <div className="welcomeText">
              Currently serving in Delhi-NCR and nearby locations.
            </div>
            <button
              type="button"
              className="welcomeBtn"
              onClick={() => setShowWelcomePopup(false)}
            >
              Continue
            </button>
          </div>
        </div>
      )}
      <header className="topbar">
        <div style={{ width: 36 }} />
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
                  <div className="typeaheadWrap">
                    <div className="inputShell">
                      <span className="glowDot glowDot--source" aria-hidden="true" />
                      <input
                        className="routeInput"
                        placeholder="Starting point"
                        value={source}
                        onChange={(e) => {
                          setSource(e.target.value)
                          setSourcePlace(null)
                          setSourceOpen(true)
                        }}
                        onFocus={() => setSourceOpen(true)}
                        onBlur={() => {
                          // allow click selection to register
                          setTimeout(() => setSourceOpen(false), 140)
                        }}
                      />
                    </div>

                    {sourceOpen && sourceSug.length > 0 && (
                      <div className="typeaheadMenu" role="listbox">
                        {sourceSug.map((it) => (
                          <button
                            key={it.id}
                            type="button"
                            className="typeaheadItem"
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => {
                              setSource(it.display_name)
                              setSourcePlace(it)
                              setSourceSug([])
                              setSourceOpen(false)
                            }}
                          >
                            <div className="typeaheadPrimary">{it.display_name}</div>
                            {it.type ? <div className="typeaheadSecondary">{it.type}</div> : null}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="arrowBetween" aria-hidden="true">
                  →
                </div>

                <div className="field">
                  <div className="fieldLabel">Destination</div>
                  <div className="typeaheadWrap">
                    <div className="inputShell">
                      <span className="glowDot glowDot--destination" aria-hidden="true" />
                      <input
                        className="routeInput"
                        placeholder="Destination"
                        value={destination}
                        onChange={(e) => {
                          setDestination(e.target.value)
                          setDestPlace(null)
                          setDestOpen(true)
                        }}
                        onFocus={() => setDestOpen(true)}
                        onBlur={() => {
                          setTimeout(() => setDestOpen(false), 140)
                        }}
                      />
                    </div>

                    {destOpen && destSug.length > 0 && (
                      <div className="typeaheadMenu" role="listbox">
                        {destSug.map((it) => (
                          <button
                            key={it.id}
                            type="button"
                            className="typeaheadItem"
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => {
                              setDestination(it.display_name)
                              setDestPlace(it)
                              setDestSug([])
                              setDestOpen(false)
                            }}
                          >
                            <div className="typeaheadPrimary">{it.display_name}</div>
                            {it.type ? <div className="typeaheadSecondary">{it.type}</div> : null}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="speedRow">
                <div className="field">
                  <div className="fieldLabel">Avg speed (km/h)</div>
                  <div className="inputShell">
                    <input
                      className="routeInput"
                      inputMode="decimal"
                      placeholder="e.g., 55"
                      value={avgSpeedKmh}
                      onChange={(e) => setAvgSpeedKmh(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              <button
                className="planButton"
                type="button"
                onClick={handlePredict}
                disabled={!source.trim() || !destination.trim() || !String(avgSpeedKmh).trim() || loading}
                aria-label="Plan My Route"
              >
                Plan My Route →
              </button>

              <div className="dividerMini" />
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
                <div
                  className={`rainBadge ${
                    hasRain ? 'rainBadge--rain' : 'rainBadge--clear'
                  }`}
                >
                  {hasRain ? 'Rain' : 'Clear'}
                </div>

                <button
                  type="button"
                  className="backBtn"
                  onClick={handleBackToPlanner}
                >
                  ← Back
                </button>
              </div>

              <div className="statsRow">
                <div className="statBox">
                  <div className="statLabel">Distance</div>
                  <div className="statValue">
                    {shownDistanceKm == null ? '—' : `${shownDistanceKm.toFixed(1)} km`}
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
                  zoomControl
                  scrollWheelZoom
                  whenCreated={(map) => {
                    setMapRef(map)
                    if (routeBounds && routeBounds.isValid()) {
                      map.fitBounds(routeBounds, {
                        padding: [10, 10],
                        maxZoom: 17,
                        animate: true,
                      })
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
                      weight={5}
                      opacity={0.35}
                    />
                  )}

                  {Array.isArray(routeSegments) &&
                    routeSegments.map((seg, idx) => (
                      <Polyline
                        key={`seg-${idx}`}
                        positions={seg.positions}
                        color={seg.color}
                        weight={7}
                        opacity={0.92}
                        eventHandlers={{
                          click: () => openSegmentPopup(seg),
                        }}
                      />
                    ))}

                  {activeSeg?.mid && (
                    <Popup
                      position={[activeSeg.mid.lat, activeSeg.mid.lon]}
                      closeButton
                      autoClose
                      closeOnEscapeKey
                      eventHandlers={{
                        remove: () => setActiveSeg(null),
                      }}
                    >
                      <div style={{ minWidth: 220 }}>
                        <div style={{ fontWeight: 900, marginBottom: 6 }}>
                          {activeSeg.locationName || 'Selected location'}
                        </div>
                        <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 8 }}>
                          {activeSeg.mid.lat.toFixed(4)}, {activeSeg.mid.lon.toFixed(4)}
                        </div>
                        <div style={{ fontWeight: 800 }}>
                          {activeSeg.inBounds ? activeSeg.label : 'Unknown (out of radar)'}
                        </div>
                        <div style={{ fontSize: 12, marginTop: 6 }}>
                          Rain: {activeSeg.rain_expected ? 'Yes' : 'No'}
                          {activeSeg.eta_mins != null ? ` • ETA ~${Math.round(Number(activeSeg.eta_mins))} min` : ''}
                          {activeSeg.dbz != null ? ` • dBZ ${Math.round(Number(activeSeg.dbz))}` : ''}
                        </div>
                      </div>
                    </Popup>
                  )}
                </MapContainer>
                <button
                  type="button"
                  className="zoomRouteBtn"
                  onClick={() => {
                    if (mapRef && routeBounds && routeBounds.isValid()) {
                      mapRef.fitBounds(routeBounds, {
                        padding: [10, 10],
                        maxZoom: 17,
                        animate: true,
                      })
                    }
                  }}
                >
                  Zoom to route
                </button>
              </div>

              <div className="legendWrap">
                <div className="legendTitle">Reflectivity (dBZ)</div>
                <div className="legendGrid">
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--veryheavy" />
                    <span className="legendText">Very Heavy</span>
                    <span className="legendDbz">&gt; 60</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--heavy" />
                    <span className="legendText">Heavy</span>
                    <span className="legendDbz">49–60</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--moderate" />
                    <span className="legendText">Moderate</span>
                    <span className="legendDbz">36–49</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--light" />
                    <span className="legendText">Light</span>
                    <span className="legendDbz">25–36</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--verylight" />
                    <span className="legendText">Very Light</span>
                    <span className="legendDbz">20–25</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--clear" />
                    <span className="legendText">Clear</span>
                    <span className="legendDbz">&lt; 20</span>
                  </div>
                  <div className="legendRow">
                    <span className="legendSwatch legendSwatch--unknown" />
                    <span className="legendText">Unknown</span>
                    <span className="legendDbz">Out of radar</span>
                  </div>
                </div>
              </div>
            </section>
          </section>
        )}
      </div>
    </div>
  )
}
