import { useEffect, useMemo, useState } from 'react'
import L from 'leaflet'
import { MapContainer, Popup, Polyline, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

export default function RouteMap({
  routeCoords,
  routeSegments,
  activeSeg,
  setActiveSeg,
  openSegmentPopup,
}) {
  const [mapRef, setMapRef] = useState(null)

  const midpoint = routeCoords?.length
    ? routeCoords[Math.floor(routeCoords.length / 2)]
    : [26.7606, 80.8893]

  const routeBounds = useMemo(() => {
    if (!Array.isArray(routeCoords) || routeCoords.length < 2) return null
    return L.latLngBounds(routeCoords)
  }, [routeCoords])

  useEffect(() => {
    if (!mapRef || !routeBounds || !routeBounds.isValid()) return
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

  return (
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

        {Array.isArray(routeCoords) && routeCoords.length > 0 && (
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
                {activeSeg.eta_mins != null
                  ? ` • ETA ~${Math.round(Number(activeSeg.eta_mins))} min`
                  : ''}
                {activeSeg.dbz != null
                  ? ` • dBZ ${Math.round(Number(activeSeg.dbz))}`
                  : ''}
              </div>
              {activeSeg.cloud_cover_pct != null && (
                <div style={{ fontSize: 12, marginTop: 4, opacity: 0.85 }}>
                  Cloud cover: {Math.round(Number(activeSeg.cloud_cover_pct))}%
                  {activeSeg.cloud_override
                    ? ' (IMD rain flag dropped — skies clear)'
                    : ''}
                </div>
              )}
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
  )
}

