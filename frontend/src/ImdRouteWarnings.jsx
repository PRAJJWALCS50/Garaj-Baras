// "IMD warnings on your route" card (data from imdWarnings.js).

// Hazard code → Hindi (English names come from the backend).
const HAZARD_HI = {
  2: 'भारी बारिश',
  3: 'भारी बर्फबारी',
  4: 'आंधी-तूफ़ान व बिजली',
  5: 'ओलावृष्टि',
  6: 'धूल भरी आंधी',
  7: 'धूल उड़ाने वाली हवाएँ',
  8: 'तेज़ सतही हवाएँ',
  9: 'लू',
  10: 'गर्म दिन',
  11: 'गर्म रात',
  12: 'शीतलहर',
  13: 'ठंडा दिन',
  14: 'पाला',
  15: 'कोहरा',
  16: 'बहुत भारी बारिश',
  17: 'अत्यधिक भारी बारिश',
}

const LEVEL_HI = { warning: 'चेतावनी', alert: 'अलर्ट', watch: 'सतर्क रहें', none: 'कोई चेतावनी नहीं' }

function DayLine({ label, day, t }) {
  if (!day) return null
  if (!day.warned) {
    return (
      <div className="imd-day imd-day--none">
        <span className="imd-day__label">{label}</span>
        <span className="imd-day__text">{t('No warning', 'कोई चेतावनी नहीं')}</span>
      </div>
    )
  }
  const names = day.hazard_codes.length
    ? day.hazard_codes.map((c, i) => t(day.hazards[i], HAZARD_HI[c] || day.hazards[i])).join(', ')
    : t('Weather warning', 'मौसम चेतावनी')
  return (
    <div className={`imd-day imd-day--${day.level}`}>
      <span className="imd-day__label">{label}</span>
      <span className="imd-day__text">{names}</span>
      <span className="imd-level" style={{ '--imd-color': day.color }}>
        {t(day.level_name, LEVEL_HI[day.level] || day.level_name)}
      </span>
    </div>
  )
}

export function ImdRouteWarnings({ data, t }) {
  if (!data) return null
  const list = data.districts || []
  return (
    <section className="imd-warnings" aria-label={t('IMD warnings on your route', 'आपके रास्ते पर IMD चेतावनियाँ')}>
      <p className="imd-warnings__head">
        <span aria-hidden>⚠️</span> {t('IMD warnings on your route', 'आपके रास्ते पर IMD चेतावनियाँ')}
      </p>
      {!data.available ? (
        <p className="imd-warnings__empty">
          {data.districts_on_route
            ? t('IMD warnings not available right now.', 'IMD चेतावनियाँ अभी उपलब्ध नहीं हैं।')
            : t('IMD district warnings cover North India only.', 'IMD ज़िला चेतावनियाँ अभी केवल उत्तर भारत के लिए हैं।')}
        </p>
      ) : !list.length ? (
        <p className="imd-warnings__empty">
          {t(`No IMD warnings on your route (${data.districts_on_route} districts checked).`,
            `आपके रास्ते पर कोई IMD चेतावनी नहीं (${data.districts_on_route} ज़िले जाँचे गए)।`)}
        </p>
      ) : (
        <ul className="imd-warnings__list">
          {list.map((d) => (
            <li key={d.id} className="imd-district">
              <p className="imd-district__name">
                {d.district}
                {d.state && <span className="imd-district__state">, {d.state}</span>}
              </p>
              <DayLine label={t('Today', 'आज')} day={d.today} t={t} />
              <DayLine label={t('Tomorrow', 'कल')} day={d.tomorrow} t={t} />
            </li>
          ))}
        </ul>
      )}
      <p className="imd-warnings__src">{t('Source: IMD district-wise warnings', 'स्रोत: IMD ज़िलेवार चेतावनियाँ')}</p>
    </section>
  )
}
