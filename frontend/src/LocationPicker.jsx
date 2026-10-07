import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import L from './leafletSetup'
import 'leaflet/dist/leaflet.css'
import { baseTiles } from './mapTiles'
import { useT } from './i18n'
import './LocationPicker.css'

const inIndiaBounds = ({ lat, lon }) => lat >= 6 && lat <= 38 && lon >= 68 && lon <= 98

export default function LocationPicker({ field, initialPlace, userLoc, onConfirm, onClose, returnFocusTo }) {
  const t = useT()
  const [initial] = useState(() => {
    const place = [userLoc, initialPlace].find(p => p && inIndiaBounds(p))
    return place ? { lat: place.lat, lon: place.lon, zoom: 14 } : { lat: 22.5, lon: 79, zoom: 5 }
  })
  const [point, setPoint] = useState(initial)
  const [moving, setMoving] = useState(false)
  const [tileError, setTileError] = useState(false)
  const [currentLocation, setCurrentLocation] = useState(userLoc)
  const [locationStatus, setLocationStatus] = useState(() => 'geolocation' in navigator ? 'locating' : 'unavailable')
  const dialogRef = useRef(null)
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const closeRef = useRef(onClose)
  useEffect(() => { closeRef.current = onClose }, [onClose])

  useEffect(() => {
    const previousFocus = returnFocusTo || document.activeElement
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialogRef.current.querySelector('button').focus()
    function onKeyDown(event) {
      if (event.key === 'Escape') closeRef.current()
      if (event.key !== 'Tab') return
      const items = [...dialogRef.current.querySelectorAll('button:not(:disabled), [tabindex="0"], a[href]')]
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    const dialog = dialogRef.current
    dialog.addEventListener('keydown', onKeyDown)
    return () => {
      dialog.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = overflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [returnFocusTo])

  useEffect(() => {
    const map = L.map(containerRef.current, { center: [initial.lat, initial.lon], zoom: initial.zoom, rotateControl: false })
    mapRef.current = map
    const { url, ...options } = baseTiles('light')
    L.tileLayer(url, options).on('tileerror', () => setTileError(true)).addTo(map)
    map.on('movestart', () => setMoving(true))
    map.on('moveend', () => {
      const center = map.getCenter().wrap()
      setPoint({ lat: center.lat, lon: center.lng })
      setMoving(false)
    })
    map.on('click', event => map.panTo(event.latlng))
    const observer = new ResizeObserver(() => map.invalidateSize())
    observer.observe(containerRef.current)

    // Refresh location each time the picker opens, without interrupting a user
    // who has already started choosing a point while the GPS request is pending.
    let alive = true
    let interacted = false
    const markInteracted = () => { interacted = true }
    const container = containerRef.current
    const inputEvents = ['pointerdown', 'keydown', 'wheel']
    inputEvents.forEach(event => container.addEventListener(event, markInteracted, { passive: true }))
    const unavailable = () => { if (alive) setLocationStatus('unavailable') }
    if ('geolocation' in navigator) {
      try {
        navigator.geolocation.getCurrentPosition(position => {
          if (!alive) return
          const location = { lat: Number(position.coords.latitude), lon: Number(position.coords.longitude) }
          if (!inIndiaBounds(location)) { unavailable(); return }
          setCurrentLocation(location)
          setLocationStatus('ready')
          if (!interacted) map.setView([location.lat, location.lon], 16)
        }, unavailable, { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 })
      } catch { unavailable() }
    }
    return () => {
      alive = false
      inputEvents.forEach(event => container.removeEventListener(event, markInteracted))
      observer.disconnect(); map.remove(); mapRef.current = null
    }
  }, [initial])

  const valid = inIndiaBounds(point)
  return createPortal(
    <div className="location-picker-backdrop">
      <section className="location-picker" role="dialog" aria-modal="true" aria-labelledby="location-picker-title" aria-describedby="location-picker-help" ref={dialogRef}>
        <header className="location-picker__header">
          <div>
            <h2 id="location-picker-title">{field === 'source' ? t('Choose starting point', 'शुरुआत का स्थान चुनें') : t('Choose destination', 'मंज़िल चुनें')}</h2>
            <p id="location-picker-help">{t('Move the map or tap a place to position the pin.', 'पिन लगाने के लिए नक्शा खिसकाएँ या किसी जगह पर टैप करें।')}</p>
          </div>
          <button type="button" className="location-picker__close" onClick={onClose} aria-label={t('Cancel map selection', 'नक्शे से चयन रद्द करें')}>×</button>
        </header>
        <div className="location-picker__map-wrap">
          <div className="location-picker__map" ref={containerRef} aria-label={t('Location map. Use arrow keys to move and plus or minus to zoom.', 'स्थान का नक्शा। खिसकाने के लिए तीर और ज़ूम के लिए प्लस या माइनस दबाएँ।')} />
          <svg className={`location-picker__pin${moving ? ' is-moving' : ''}`} width="40" height="52" viewBox="0 0 40 52" aria-hidden="true">
            <path d="M20 50S2 29 2 20a18 18 0 1 1 36 0c0 9-18 30-18 30Z" fill={field === 'source' ? '#863bff' : '#f59e0b'} stroke="white" strokeWidth="3" />
            <circle cx="20" cy="20" r="6" fill="white" />
          </svg>
          {currentLocation && inIndiaBounds(currentLocation) && <button type="button" className="location-picker__locate" onClick={() => mapRef.current?.setView([currentLocation.lat, currentLocation.lon], 16)}>{t('My location', 'मेरा स्थान')}</button>}
        </div>
        <footer className="location-picker__footer">
          <p className="location-picker__coordinates">{point.lat.toFixed(5)}, {point.lon.toFixed(5)}</p>
          <p role="status">{!valid ? t('Please choose a location within India.', 'कृपया भारत के भीतर कोई स्थान चुनें।') : tileError ? t('Map tiles could not load. Check your connection.', 'नक्शा लोड नहीं हुआ। अपना कनेक्शन जाँचें।') : locationStatus === 'locating' ? t('Finding your current location…', 'आपका वर्तमान स्थान खोज रहे हैं…') : locationStatus === 'unavailable' && !currentLocation ? t('Current location unavailable. Move the map to choose a place.', 'वर्तमान स्थान उपलब्ध नहीं है। नक्शा खिसकाकर कोई स्थान चुनें।') : t('The pin marks your exact route location.', 'पिन आपके रास्ते का सटीक स्थान दिखाता है।')}</p>
          <div className="location-picker__actions">
            <button type="button" onClick={onClose}>{t('Cancel', 'रद्द करें')}</button>
            <button type="button" className="location-picker__confirm" disabled={!valid || moving} onClick={() => onConfirm({ ...point, display_name: `${t('Pinned location', 'चुना हुआ स्थान')} (${point.lat.toFixed(5)}, ${point.lon.toFixed(5)})` })}>{t('Use this location', 'यह स्थान चुनें')}</button>
          </div>
        </footer>
      </section>
    </div>, document.body
  )
}
