import { useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { baseTiles } from './mapTiles'

function FramePlaces({ points }) {
  const map = useMap()
  useEffect(() => {
    if (points.length === 2) map.fitBounds(points, { padding: [70, 70], maxZoom: 12 })
    else if (points.length) map.setView(points[0], 10)
  }, [map, points])
  useEffect(() => {
    const observer = new ResizeObserver(() => map.invalidateSize())
    observer.observe(map.getContainer())
    return () => observer.disconnect()
  }, [map])
  return null
}

export default function PlannerMap({ source, destination }) {
  const places = [source, destination].filter(p => p && Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon)))
  const points = places.map(p => [Number(p.lat), Number(p.lon)])
  return <MapContainer center={[28.0, 77.7]} zoom={8} zoomControl={false} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
    <TileLayer {...baseTiles('route')} />
    <FramePlaces points={points} />
    {places.map((place, i) => <CircleMarker key={`${place.lat}-${place.lon}`} center={points[i]} radius={8} pathOptions={{ color: '#fff', fillColor: place === source ? '#308bff' : '#f6ad55', fillOpacity: 1, weight: 3 }}><Popup>{place.display_name}</Popup></CircleMarker>)}
  </MapContainer>
}
