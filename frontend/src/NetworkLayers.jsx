import { useMemo, useState } from 'react'

const LAYERS = [
  {
    id: 'application',
    title: 'Application layer',
    subtitle: 'User-facing protocols & app data',
    pdu: 'Data',
    examples: ['HTTP/HTTPS', 'DNS', 'SMTP', 'FTP', 'WebSocket'],
    addressing: 'URLs, hostnames',
    devices: ['Proxies', 'WAF', 'Application gateways'],
    whatHappens:
      'Your app creates/reads meaningful data (requests, responses, queries). It does not know about ports, packets, or MAC addresses — it just asks the network stack to send bytes.',
  },
  {
    id: 'session',
    title: 'Session layer',
    subtitle: 'Conversation, state, and resuming',
    pdu: 'Data',
    examples: ['TLS session resumption', 'RPC sessions', 'SIP dialogs'],
    addressing: 'Session IDs / tokens',
    devices: ['Load balancers (sticky sessions)', 'Session border controllers'],
    whatHappens:
      'Creates and manages a “conversation” between two endpoints: establishes, maintains, and tears down sessions, and can support resume/retry semantics.',
  },
  {
    id: 'transport',
    title: 'Transport layer',
    subtitle: 'Process-to-process delivery',
    pdu: 'Segment (TCP) / Datagram (UDP)',
    examples: ['TCP', 'UDP', 'QUIC (over UDP)'],
    addressing: 'Ports (e.g., 443, 53)',
    devices: ['Firewalls', 'NAT'],
    whatHappens:
      'Splits data into segments, adds ports, reliability/ordering (TCP), or low-latency best-effort delivery (UDP). Handles retransmits, congestion control, and flow control (TCP).',
  },
  {
    id: 'datalink',
    title: 'Data Link layer',
    subtitle: 'Local network delivery (LAN/Wi‑Fi)',
    pdu: 'Frame',
    examples: ['Ethernet (802.3)', 'Wi‑Fi (802.11)', 'ARP'],
    addressing: 'MAC addresses',
    devices: ['Switches', 'Wireless APs'],
    whatHappens:
      'Moves frames on the local link using MAC addresses. This is where switching happens, and where your device talks to the next hop on the same network.',
  },
]

function wrap(label, payload) {
  return { label, payload }
}

function prettyNode(node, depth = 0) {
  if (!node) return null
  const pad = '  '.repeat(depth)
  if (typeof node === 'string') return `${pad}${node}`
  return `${pad}${node.label}\n${prettyNode(node.payload, depth + 1)}`
}

export default function NetworkLayers() {
  const [activeId, setActiveId] = useState('application')
  const [message, setMessage] = useState('GET /predict_waypoints HTTP/1.1')
  const [step, setStep] = useState(0) // 0..4
  const [direction, setDirection] = useState('send') // send | receive

  const active = useMemo(
    () => LAYERS.find((l) => l.id === activeId) || LAYERS[0],
    [activeId]
  )

  const stack = useMemo(() => {
    const app = String(message || '').trim() || '(empty message)'
    const appNode = wrap('Application: data', app)
    const sessionNode = wrap('Session: session context', appNode)
    const transportNode = wrap('Transport: ports + reliability', sessionNode)
    const linkNode = wrap('Data Link: MAC + frame check', transportNode)
    return { appNode, sessionNode, transportNode, linkNode }
  }, [message])

  const visibleNode = useMemo(() => {
    const order = [stack.appNode, stack.sessionNode, stack.transportNode, stack.linkNode]
    if (direction === 'send') {
      // step 0 shows app only; step 4 shows frame (full)
      return order[Math.max(0, Math.min(order.length - 1, step - 1))] || stack.appNode
    }
    // receive: start with frame; peel off each step
    const rev = [stack.linkNode, stack.transportNode, stack.sessionNode, stack.appNode]
    return rev[Math.max(0, Math.min(rev.length - 1, step - 1))] || stack.linkNode
  }, [direction, stack, step])

  const stepLabel = useMemo(() => {
    if (direction === 'send') {
      if (step === 0) return 'Ready'
      if (step === 1) return 'Application → Session'
      if (step === 2) return 'Session → Transport'
      if (step === 3) return 'Transport → Data Link'
      return 'Frame on the wire'
    }
    if (step === 0) return 'Ready'
    if (step === 1) return 'Frame received'
    if (step === 2) return 'Peel Data Link → Transport'
    if (step === 3) return 'Peel Transport → Session'
    return 'Deliver to application'
  }, [direction, step])

  function reset() {
    setStep(0)
  }

  function next() {
    setStep((s) => Math.min(4, s + 1))
  }

  function prev() {
    setStep((s) => Math.max(0, s - 1))
  }

  return (
    <div className="netScreen">
      <section className="card netHero">
        <div className="netTitle">Computer Networks — Layers</div>
        <div className="netSub">
          Tap a layer to learn it, then use the simulator to see how data is wrapped into headers
          (encapsulation) and unwrapped (decapsulation).
        </div>
      </section>

      <section className="netGrid">
        {LAYERS.map((l) => {
          const active = l.id === activeId
          return (
            <button
              key={l.id}
              type="button"
              className={`netLayerCard ${active ? 'netLayerCard--active' : ''}`}
              onClick={() => setActiveId(l.id)}
            >
              <div className="netLayerTop">
                <div className="netLayerName">{l.title}</div>
                <div className="netLayerPdu">{l.pdu}</div>
              </div>
              <div className="netLayerSub">{l.subtitle}</div>
            </button>
          )
        })}
      </section>

      <section className="card netDetail">
        <div className="netDetailHead">
          <div className="netDetailTitle">{active.title}</div>
          <div className="netDetailBadge">{active.pdu}</div>
        </div>

        <div className="netDetailBody">{active.whatHappens}</div>

        <div className="netDetailMeta">
          <div className="netMetaRow">
            <div className="netMetaLabel">Addressing</div>
            <div className="netMetaValue">{active.addressing}</div>
          </div>
          <div className="netMetaRow">
            <div className="netMetaLabel">Examples</div>
            <div className="netMetaValue">{active.examples.join(', ')}</div>
          </div>
          <div className="netMetaRow">
            <div className="netMetaLabel">Devices</div>
            <div className="netMetaValue">{active.devices.join(', ')}</div>
          </div>
        </div>
      </section>

      <section className="card netSim">
        <div className="netSimHead">
          <div>
            <div className="netSimTitle">Interactive encapsulation</div>
            <div className="netSimSub">
              Switch between “Send” (wrap headers) and “Receive” (peel headers).
            </div>
          </div>

          <div className="netSimMode">
            <button
              type="button"
              className={`netModeBtn ${direction === 'send' ? 'netModeBtn--active' : ''}`}
              onClick={() => {
                setDirection('send')
                reset()
              }}
            >
              Send
            </button>
            <button
              type="button"
              className={`netModeBtn ${direction === 'receive' ? 'netModeBtn--active' : ''}`}
              onClick={() => {
                setDirection('receive')
                reset()
              }}
            >
              Receive
            </button>
          </div>
        </div>

        <div className="netSimInputRow">
          <div className="netInputLabel">Message</div>
          <input
            className="netInput"
            value={message}
            onChange={(e) => {
              setMessage(e.target.value)
              reset()
            }}
            placeholder="Type any message…"
          />
        </div>

        <div className="netStepRow">
          <div className="netStepBadge">{stepLabel}</div>
          <div className="netStepBtns">
            <button type="button" className="netStepBtn" onClick={prev} disabled={step <= 0}>
              ←
            </button>
            <button type="button" className="netStepBtn" onClick={next} disabled={step >= 4}>
              →
            </button>
          </div>
        </div>

        <div className="netCodeWrap" role="region" aria-label="Encapsulation output">
          <pre className="netCode">{prettyNode(visibleNode) || ''}</pre>
        </div>

        <div className="netHint">
          Tip: in the real world, there’s also a <b>Network layer (IP)</b> between Transport and Data
          Link. You asked for these four, so this view focuses on them.
        </div>
      </section>
    </div>
  )
}

