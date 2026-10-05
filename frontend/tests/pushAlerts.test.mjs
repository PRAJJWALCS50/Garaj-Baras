import test from 'node:test'
import assert from 'node:assert/strict'
import { hasPushSupport, publicKeyBytes, subscriptionMatchesKey, activePushRegistration, pushSubscriptionForKey } from '../src/pushAlerts.js'
const key = new Uint8Array(65); key[0] = 4; key[1] = 7

test('unsupported or insecure browsers are detected', () => {
  const w = { Notification: { requestPermission() {} }, PushManager: {}, isSecureContext: true }
  assert.equal(hasPushSupport(w, { serviceWorker: {} }), true)
  assert.equal(hasPushSupport({}, { serviceWorker: {} }), false)
  assert.equal(hasPushSupport({ ...w, isSecureContext: false }, { serviceWorker: {} }), false)
  assert.equal(hasPushSupport(w, {}), false)
})
test('VAPID decoding validates key structure', () => {
  assert.deepEqual(publicKeyBytes(Buffer.from(key).toString('base64url')), key)
  assert.throws(() => publicKeyBytes('invalid'))
  assert.equal(subscriptionMatchesKey({ options: { applicationServerKey: key.buffer } }, key), true)
  assert.equal(subscriptionMatchesKey({}, key), false)
})
test('reuse valid subscription and replace rotated keys in order', async () => {
  const calls = []
  const old = { options: { applicationServerKey: key.buffer }, unsubscribe: async () => calls.push('unsubscribe') }
  const reg = { pushManager: { getSubscription: async () => old, subscribe: async options => { calls.push('subscribe'); assert.equal(options.userVisibleOnly, true); return 'new' } } }
  assert.equal(await pushSubscriptionForKey(reg, key), old)
  assert.deepEqual(calls, [])
  const rotated = key.slice(); rotated[1] = 9
  assert.equal(await pushSubscriptionForKey(reg, rotated), 'new')
  assert.deepEqual(calls, ['unsubscribe', 'subscribe'])
})
test('worker activation resolves or times out instead of hanging', async () => {
  const active = { pushManager: {} }
  assert.equal(await activePushRegistration({ register: async path => assert.equal(path, '/sw.js'), ready: Promise.resolve(active) }, 100), active)
  await assert.rejects(activePushRegistration({ register: async () => {}, ready: new Promise(() => {}) }, 5), /could not start/)
})
