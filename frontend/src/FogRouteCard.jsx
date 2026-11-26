// "Fog on your route" card: one place for every fog signal on the drive.
//   Satellite (Meteosat-9)  -> fog on the road NOW
//   ECMWF AIFS              -> fog expected at the time you pass each point
//   IMD district warnings   -> official fog watch/alert for districts on the route
// Shows a headline, an A -> B strip with one lane per source, and a row per source.
import { FOG_COLORS, fogZones } from './fog'

function clock(etaMins) {
  if (!Number.isFinite(etaMins)) return ''
  return new Date(Date.now() + etaMins * 60000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

function kmRange(z) {
  const a = Math.max(0, Math.round(z.startKm))
  const b = Math.max(a + 1, Math.round(z.endKm))
  return `km ${a}–${b}`
}

function zoneTime(z) {
  if (!Number.isFinite(z.etaFrom)) return ''
  const a = clock(z.etaFrom)
  const b = Number.isFinite(z.etaTo) && z.etaTo - z.etaFrom >= 10 ? clock(z.etaTo) : ''
  return b && b !== a ? `${a}–${b}` : a
}

/** IMD districts on the route with a fog warning (hazard code 15) today or tomorrow. */
export function imdFogDistricts(imd) {
  const out = []
  for (const d of imd?.districts || []) {
    const today = d.today?.warned && (d.today.hazard_codes || []).includes(15)
    const tomorrow = d.tomorrow?.warned && (d.tomorrow.hazard_codes || []).includes(15)
    if (today || tomorrow) out.push({ name: d.district, today, tomorrow, level: (today ? d.today : d.tomorrow).level_name })
  }
  return out
}

/** Everything the card, the status pill and the route chip need, in one place. */
export function fogSummary({ satPoints, satScene, aifsPoints, aifsRun, imd }) {
  const satZones = fogZones(satPoints || [])   // non-fog points carry level null
  const aifsZones = fogZones(aifsPoints || [])
  const aifsFog = aifsZones.filter((z) => z.level === 'thick')
  const imdFog = imdFogDistricts(imd)
  const checking = satPoints == null || aifsPoints == null
  let tone = 'clear'
  if (satZones.length || aifsFog.length) tone = 'fog'
  else if (aifsZones.length || imdFog.length) tone = 'watch'
  return { satZones, aifsZones, aifsFog, imdFog, checking, tone, risk: tone !== 'clear',
           satState: satScene?.state || null, aifsState: aifsRun?.state || null }
}

function Lane({ label, zones, totalKm, kind }) {
  return (
    <div className="fogcard__lane">
      <span className="fogcard__lane-label">{label}</span>
      <div className={`fogcard__track fogcard__track--${kind}`}>
        {zones.map((z, i) => {
          const left = Math.max(0, Math.min(100, (z.startKm / totalKm) * 100))
          const width = Math.max(2.5, Math.min(100 - left, ((z.endKm - z.startKm) / totalKm) * 100))
          return (
            <span key={i} className="fogcard__seg"
              style={{ left: `${left}%`, width: `${width}%`, background: FOG_COLORS[z.level] || FOG_COLORS.fog }}
              title={`${kmRange(z)} ${zoneTime(z)}`} />
          )
        })}
      </div>
    </div>
  )
}

export function FogRouteCard({ summary, satScene, aifsRun, totalKm, t }) {
  if (!summary) return null
  const { satZones, aifsZones, aifsFog, imdFog, checking, tone } = summary
  const km = Number(totalKm) > 0 ? Number(totalKm) : Math.max(1, ...[...satZones, ...aifsZones].map((z) => z.endKm))

  // headline: what matters most for the driver, in one sentence
  let head, sub
  if (satZones.length) {
    const z = satZones[0]
    head = t(`Fog on the road now · ${kmRange(z)}`, `अभी सड़क पर कोहरा · ${kmRange(z)}`)
    sub = t('Seen by satellite in the latest image. Slow down, use low beams.',
      'ताज़ा सैटेलाइट तस्वीर में दिखा। धीरे चलें, लो-बीम इस्तेमाल करें।')
  } else if (aifsFog.length) {
    const z = aifsFog[0]
    head = t(`Fog likely ${kmRange(z)} · around ${zoneTime(z)}`, `${kmRange(z)} पर कोहरे की संभावना · लगभग ${zoneTime(z)}`)
    sub = t('ECMWF forecast for the time you reach that stretch.', 'उस हिस्से तक पहुँचने के समय का ECMWF पूर्वानुमान।')
  } else if (aifsZones.length) {
    const z = aifsZones[0]
    head = t(`Possible fog ${kmRange(z)} · around ${zoneTime(z)}`, `${kmRange(z)} पर कोहरा हो सकता है · लगभग ${zoneTime(z)}`)
    sub = t('Air near saturation and calm wind when you pass. Keep an eye out.',
      'गुज़रते समय हवा लगभग नम और शांत रहेगी। ध्यान रखें।')
  } else if (imdFog.length) {
    head = t('IMD fog watch on your route', 'आपके रास्ते पर IMD कोहरा चेतावनी')
    sub = t('Satellite and ECMWF see no fog at your times — still, take care in these districts.',
      'सैटेलाइट और ECMWF को आपके समय पर कोहरा नहीं दिखा — फिर भी इन ज़िलों में सावधानी रखें।')
  } else if (checking) {
    head = t('Checking fog on your route…', 'आपके रास्ते पर कोहरा जाँच रहे हैं…')
    sub = t('Satellite now + ECMWF forecast at your times.', 'अभी का सैटेलाइट + आपके समय का ECMWF पूर्वानुमान।')
  } else {
    head = t('No fog expected on this drive', 'इस सफ़र में कोहरे की संभावना नहीं')
    sub = t('Satellite sees none now, and ECMWF expects none when you pass.',
      'सैटेलाइट को अभी कोहरा नहीं दिखा, और ECMWF के अनुसार गुज़रते समय भी नहीं।')
  }

  const satText = (() => {
    const st = summary.satState
    if (satScene == null && checking) return t('Checking…', 'जाँच रहे हैं…')
    if (!st || st === 'no_data') return t('Unavailable', 'उपलब्ध नहीं')
    if (st === 'stale') return t('Image delayed', 'तस्वीर देर से')
    if (st === 'daytime') return t('No fog', 'कोहरा नहीं')
    if (satZones.length) return satZones.map(kmRange).join(', ')
    return t('No fog seen', 'कोहरा नहीं दिखा')
  })()
  const satTime = satScene?.scene_time
    ? new Date(satScene.scene_time).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : ''

  const aifsText = (() => {
    const st = summary.aifsState
    if (aifsRun == null && checking) return t('Checking…', 'जाँच रहे हैं…')
    if (!st || st === 'no_data') return t('Unavailable', 'उपलब्ध नहीं')
    if (!aifsZones.length) return t('No fog expected', 'कोहरे की संभावना नहीं')
    return aifsZones.slice(0, 3).map((z) => `${z.level === 'thick' ? t('Fog', 'कोहरा') : t('Possible', 'संभव')} ${kmRange(z)} · ${zoneTime(z)}`).join('  ·  ')
  })()
  const aifsRunLabel = /T(\d\d)/.exec(aifsRun?.run || '')?.[1]

  return (
    <section className={`fogcard fogcard--${tone}`} aria-label={t('Fog on your route', 'आपके रास्ते पर कोहरा')}>
      <div className="fogcard__top">
        <span className="fogcard__eyebrow"><span aria-hidden>🌫</span> {t('Fog on your route', 'आपके रास्ते पर कोहरा')}</span>
        <span className={`fogcard__badge fogcard__badge--${tone}`}>
          {tone === 'fog' ? t('Fog risk', 'कोहरे का खतरा') : tone === 'watch' ? t('Stay alert', 'सतर्क रहें') : checking ? t('Checking', 'जाँच') : t('Clear', 'साफ़')}
        </span>
      </div>
      <p className="fogcard__head">{head}</p>
      <p className="fogcard__sub">{sub}</p>

      <div className="fogcard__strip" aria-hidden>
        <Lane label={t('Now', 'अभी')} zones={satZones} totalKm={km} kind="sat" />
        <Lane label={t('When you pass', 'गुज़रते समय')} zones={aifsZones} totalKm={km} kind="aifs" />
        <div className="fogcard__ends">
          <span>A · 0 km</span><span>{Math.round(km / 2)} km</span><span>B · {Math.round(km)} km</span>
        </div>
      </div>

      <ul className="fogcard__rows">
        <li>
          <span className="fogcard__src"><span aria-hidden>🛰</span> {t('Satellite · now', 'सैटेलाइट · अभी')}</span>
          <span className="fogcard__val">{satText}{satTime && <em> · {t('image', 'तस्वीर')} {satTime}</em>}</span>
        </li>
        <li>
          <span className="fogcard__src"><span aria-hidden>🌐</span> {t('ECMWF · when you pass', 'ECMWF · गुज़रते समय')}</span>
          <span className="fogcard__val">{aifsText}{aifsRunLabel && <em> · AIFS {aifsRunLabel} UTC</em>}</span>
        </li>
        {imdFog.length > 0 && (
          <li>
            <span className="fogcard__src"><span aria-hidden>⚠️</span> {t('IMD warning', 'IMD चेतावनी')}</span>
            <span className="fogcard__val">
              {t('Fog', 'कोहरा')} {imdFog[0].level?.toLowerCase() || t('watch', 'सतर्कता')}: {imdFog.map((d) => d.name).join(', ')}
            </span>
          </li>
        )}
      </ul>
      <div className="fogcard__legend">
        <span><i style={{ background: FOG_COLORS.thick }} /> {t('Fog', 'कोहरा')}</span>
        <span><i style={{ background: FOG_COLORS.fog }} /> {t('Possible / patchy', 'संभव / छिटपुट')}</span>
      </div>
    </section>
  )
}
