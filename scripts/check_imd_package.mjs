// Fail builds when an IMD feature upload omits its code, data, or wiring.
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import assert from 'node:assert/strict'

const root = fileURLToPath(new URL('../', import.meta.url))
const required = [
  '.gitignore', 'ARCHITECTURE.md', '.github/workflows/keepalive.yml',
  'backend/main.py', 'backend/imd_warnings.py', 'backend/radar_overlay.py',
  'frontend/src/RadarOverlay.jsx', 'frontend/src/PlannerMap.jsx',
  'frontend/src/JourneyWeather.jsx', 'frontend/src/JourneyDesign.css',
  'backend/data/imd_north_districts.geojson', 'frontend/src/App.jsx',
  'frontend/src/App.css', 'frontend/src/imdWarnings.js',
  'frontend/src/ImdRouteWarnings.jsx',
  'backend/native_radar.py', 'backend/georef_native.py', 'backend/sohra_digits.json',
  'backend/radar_sohra.py', 'backend/georef_sohra.py',
  'backend/radar_mahabaleshwar.py', 'backend/georef_mahabaleshwar.py',
  'backend/radar_mangaluru.py', 'backend/georef_mangaluru.py', 'backend/mangaluru_glyphs.json',
  'backend/radar_thiruvananthapuram.py', 'backend/georef_thiruvananthapuram.py', 'backend/thiruvananthapuram_digits.json',
]
const missing = required.filter(path => !existsSync(resolve(root, path)))
assert.equal(missing.length, 0, `Incomplete IMD upload: ${missing.join(', ')}`)
const read = path => readFileSync(resolve(root, path), 'utf8')
const main = read('backend/main.py')
for (const station of ['sohra', 'mahabaleshwar', 'mangaluru', 'thiruvananthapuram']) {
  assert(main.includes(`import radar_${station}`) && main.includes(`import georef_${station}`))
  assert(main.includes(`"${station}":  {"refresh": _do_${station}_refresh`), `Missing ${station} registry`)
  assert(main.includes(`elif radar == "${station}"`), `Missing ${station} dispatch`)
}
assert.match(main, /^import imd_warnings\b/m, 'Backend does not load IMD warnings')
assert(main.includes('@app.post("/imd_warnings/route")'), 'IMD route endpoint is missing')
assert(main.includes('@app.get("/tasks/fetch_imd_warnings")'), 'IMD refresh endpoint is missing')
const geo = JSON.parse(read('backend/data/imd_north_districts.geojson'))
assert(geo.type === 'FeatureCollection' && geo.features?.length, 'District data is empty')
const ids = geo.features.map(feature => Number(feature.properties.ID))
assert(ids.every(id => Number.isInteger(id) && id > 0), 'Invalid district ID')
assert.equal(ids.length, new Set(ids).size, 'District IDs must be unique')
for (const feature of geo.features) {
  assert(['Polygon', 'MultiPolygon'].includes(feature.geometry.type))
  assert(feature.geometry.coordinates.length, 'District polygon is empty')
}
const app = read('frontend/src/App.jsx')
assert(app.includes("from './imdWarnings'") && app.includes("from './ImdRouteWarnings'"))
assert(app.includes('<ImdRouteWarnings '), 'Warning card is not mounted')
assert(read('.github/workflows/keepalive.yml').includes('/tasks/fetch_imd_warnings'), 'Scheduled IMD refresh is missing')
console.log(`IMD package complete: ${required.length} files, ${ids.length} districts, endpoints and UI wired`)
