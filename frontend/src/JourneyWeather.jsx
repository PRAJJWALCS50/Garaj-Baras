import { useState } from 'react'
import { useT } from './i18n'

export default function JourneyWeather({ waypoints, onSelect, onPreview }) {
  const t = useT()
  const [index, setIndex] = useState(0)
  const points = waypoints.filter(p => Number.isFinite(Number(p.eta_mins)))
  if (!points.length) return null
  const selected = Math.min(index, points.length - 1)
  const point = points[selected]
  const covered = point.in_radar_bounds === true
  const label = !covered ? t('Outside radar coverage', 'रडार कवरेज के बाहर') : point.rain_expected ? point.label : t('No rain predicted', 'बारिश का अनुमान नहीं')
  return <section className="journey-weather" aria-label={t('Weather along your journey', 'यात्रा के दौरान मौसम')}>
    <div className="journey-weather__heading"><h2>{t('Weather along your journey', 'यात्रा के दौरान मौसम')}</h2><span>{t('FORECAST', 'पूर्वानुमान')}</span></div>
    <div className="journey-weather__selected"><strong>{label}</strong><span>+{Math.round(point.eta_mins)} {t('min into your trip', 'मिनट बाद')}</span></div>
    <div className="journey-weather__strip" aria-hidden="true">{points.map((p, i) => <span key={i} style={{ background: p.in_radar_bounds !== true ? '#64748b' : p.rain_expected ? '#f6ad55' : '#308bff', opacity: i === selected ? 1 : 0.55 }} />)}</div>
    <input aria-label={t('Explore journey time', 'यात्रा समय देखें')} type="range" min="0" max={points.length - 1} value={selected} onChange={e => { const next = Number(e.target.value); setIndex(next); onPreview?.(points[next]) }} />
    <div className="journey-weather__labels"><span>{t('Departure', 'प्रस्थान')}</span><span>+{Math.round(points.at(-1).eta_mins)} {t('min', 'मिनट')}</span></div>
    <button type="button" className="journey-weather__detail" onClick={() => onSelect(point)} disabled={!covered}>{t('Explore weather at this point', 'इस बिंदु का मौसम देखें')}</button>
    <p>{t('Blue: no rain predicted · Amber: rain · Gray: no coverage', 'नीला: बारिश का अनुमान नहीं · पीला: बारिश · ग्रे: कवरेज नहीं')}</p>
  </section>
}
