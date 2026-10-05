export function hasPushSupport(browserWindow = window, browserNavigator = navigator) {
  return browserWindow.isSecureContext !== false &&
    typeof browserWindow.Notification?.requestPermission === 'function' && 'PushManager' in browserWindow &&
    'serviceWorker' in browserNavigator
}

export function publicKeyBytes(value) {
  const encoded = String(value || '').replace(/-/g, '+').replace(/_/g, '/')
  const bytes = Uint8Array.from(atob(encoded + '='.repeat((4 - encoded.length % 4) % 4)), c => c.charCodeAt(0))
  if (bytes.length !== 65 || bytes[0] !== 4) throw new Error('The notification server key is invalid. Please try again later.')
  return bytes
}

export function subscriptionMatchesKey(subscription, key) {
  const existing = subscription?.options?.applicationServerKey
  if (!existing) return false
  const bytes = new Uint8Array(existing)
  return bytes.length === key.length && bytes.every((value, index) => value === key[index])
}

export async function activePushRegistration(serviceWorker = navigator.serviceWorker, timeoutMs = 20000) {
  let timer
  try {
    return await Promise.race([
      (async () => { await serviceWorker.register('/sw.js'); return serviceWorker.ready })(),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Notifications could not start. Reload this page and try again.')), timeoutMs)
      }),
    ])
  } finally { clearTimeout(timer) }
}

export async function pushSubscriptionForKey(registration, key) {
  const existing = await registration.pushManager.getSubscription()
  if (existing && subscriptionMatchesKey(existing, key)) return existing
  if (existing) await existing.unsubscribe()
  return registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key })
}
