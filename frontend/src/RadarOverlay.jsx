import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Circle, ImageOverlay, Pane, useMap } from 'react-leaflet'
import { useT } from './i18n'

export default function RadarOverlay({ apiBase, point }) {
  const map = useMap()
  const t = useT()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [index, setIndex] = useState(0)
  const [opacity, setOpacity] = useState(0.65)
  const [playing, setPlaying] = useState(false)
  const [retry, setRetry] = useState(0)
  const [clock, setClock] = useState(() => Date.now())
  const lat = point?.[0], lon = point?.[1]
  useEffect(() => {
    let active = true
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 90000)
    fetch(`${apiBase}/radar/overlay?lat=${lat}&lon=${lon}`, { signal: controller.signal })
      .then(async response => {
        const body = await response.json()
        if (!response.ok) throw new Error(body.detail || 'Radar unavailable')
        if (!body.frames?.length) throw new Error('No observed radar frames available')
        return body
      })
      .then(body => { if (active) { setData(body); setIndex(body.frames.length - 1); setClock(Date.now()) } })
      .catch(err => { if (!active) return; if (!controller.signal.aborted) setError(err.message); else setError('Radar request timed out. Please retry.') })
      .finally(() => clearTimeout(timeout))
    return () => { active = false; clearTimeout(timeout); controller.abort() }
  }, [apiBase, lat, lon, retry])
  useEffect(() => {
    const timer = setInterval(() => setClock(Date.now()), 30000)
    return () => clearInterval(timer)
  }, [])
  useEffect(() => {
    if (!playing || !data) return
    const timer = setInterval(() => setIndex(i => (i + 1) % data.frames.length), 850)
    return () => clearInterval(timer)
  }, [playing, data])
  const frame = data?.frames[index]
  const age = data ? Math.max(0, Math.floor((clock - Date.parse(data.latest_timestamp)) / 60000)) : 0
  const stale = age > 90
  const controls = <div className="radar-controls" onPointerDown={e => e.stopPropagation()} onClick={e => e.stopPropagation()} onWheel={e => e.stopPropagation()}>
    <div className="radar-controls__top"><strong>{t('Observed radar', 'देखा गया रडार')} {data && `· ${data.station}`}</strong><button type="button" onClick={() => { setData(null); setError(''); setPlaying(false); setRetry(v => v + 1) }}>{t('Refresh', 'ताज़ा करें')}</button></div>
    {!data && !error && <p role="status">{t('Loading measured scans… First load may take a minute.', 'रडार स्कैन लोड हो रहे हैं… पहली बार एक मिनट लग सकता है।')}</p>}
    {error && <p role="alert">{error}</p>}
    {data && <>
      <p role="status">{stale ? t('Scan is stale — overlay hidden. Refresh to retry.', 'स्कैन पुराना है — परत छिपाई गई है। ताज़ा करें।') : `${new Date(frame.timestamp).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false })} IST · ${t('Latest scan', 'नवीनतम स्कैन')} ${age} ${t('min ago', 'मिनट पहले')}`}</p>
      <label><button type="button" disabled={data.frames.length < 2 || stale} onClick={() => setPlaying(v => !v)} aria-label={playing ? t('Pause radar', 'रडार रोकें') : t('Play radar history', 'रडार इतिहास चलाएँ')}>{playing ? 'Ⅱ' : '▶'}</button><input type="range" aria-label={t('Radar observation time', 'रडार अवलोकन समय')} min="0" max={data.frames.length - 1} value={index} onChange={e => { setPlaying(false); setIndex(Number(e.target.value)) }} /></label>
      <label>{t('Opacity', 'अपारदर्शिता')}<input type="range" aria-label={t('Radar opacity', 'रडार अपारदर्शिता')} min="0.15" max="0.9" step="0.05" value={opacity} onChange={e => setOpacity(Number(e.target.value))} /></label>
      <div className="radar-controls__legend"><span><i style={{ background: '#0019b0' }} />20</span><span><i style={{ background: '#1aa3ff' }} />35</span><span><i style={{ background: '#ffff00' }} />44</span><span><i style={{ background: '#ff0000' }} />55+ dBZ</span></div>
      <p>{t('IMD reflectivity · One station’s coverage · Past scans, not a forecast.', 'IMD परावर्तन · एक स्टेशन का कवरेज · पिछले स्कैन, पूर्वानुमान नहीं।')}</p>
    </>}
  </div>
  return <>
    {frame && !stale && <Pane name="observed-radar" style={{ zIndex: 350, pointerEvents: 'none' }}><ImageOverlay url={frame.image} bounds={data.bounds} opacity={opacity} interactive={false} /><Circle center={data.center} radius={data.range_km * 1000} pathOptions={{ color: '#80b9ec', weight: 1, dashArray: '4 8', fill: false }} interactive={false} /></Pane>}
    {createPortal(controls, map.getContainer().parentElement)}
  </>
}
