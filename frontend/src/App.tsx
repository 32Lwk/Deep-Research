import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type ProviderInfo = { available: string[] }
type RunsList = { runs: { run_id: string; has_report: boolean; has_pdf: boolean; updated_at: number }[] }
type Graph = { nodes: { id: string; type: string; label: string }[]; edges: { source: string; target: string; type: string }[] }
type EventsTail = { run_id: string; events: string[] }

type AgentRoute = { provider: string; model: string }
type RoutingPreview = {
  roles: Record<string, AgentRoute>
  debate_routes: Record<string, AgentRoute>
  debate_diversify?: boolean
  available_providers?: string[]
}

function flattenRoutingPreview(p: RoutingPreview | null): Record<string, AgentRoute> {
  if (!p) return {}
  return { ...p.roles, ...p.debate_routes }
}

function parseRunStartRoutes(raw: Record<string, unknown>): Record<string, AgentRoute> | null {
  const ar = raw.agent_routes
  if (!ar || typeof ar !== 'object') return null
  const out: Record<string, AgentRoute> = {}
  for (const [k, v] of Object.entries(ar as Record<string, unknown>)) {
    if (!v || typeof v !== 'object') continue
    const o = v as Record<string, unknown>
    if (typeof o.provider === 'string' && typeof o.model === 'string') out[k] = { provider: o.provider, model: o.model }
  }
  return Object.keys(out).length ? out : null
}

type UiEvent =
  | { kind: 'ws_closed' }
  | { kind: 'raw'; text: string }
  | {
      kind: 'json'
      type?: string
      ts_ms?: number
      run_id?: string
      agent_id?: string
      role?: string
      phase?: string
      provider?: string
      model?: string
      text?: string
      topic?: string
      seed_urls?: string[]
      raw: Record<string, unknown>
    }

async function jTimeout<T>(path: string, init?: RequestInit, timeoutMs: number = 10_000): Promise<T> {
  const ac = new AbortController()
  const id = window.setTimeout(() => ac.abort(), timeoutMs)
  try {
    const r = await fetch(path, { ...init, signal: ac.signal })
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
    return (await r.json()) as T
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw new Error(`timeout after ${timeoutMs}ms`)
    throw e
  } finally {
    window.clearTimeout(id)
  }
}

async function tTimeout(path: string, init?: RequestInit, timeoutMs: number = 10_000): Promise<string> {
  const ac = new AbortController()
  const id = window.setTimeout(() => ac.abort(), timeoutMs)
  try {
    const r = await fetch(path, { ...init, signal: ac.signal })
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
    return await r.text()
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw new Error(`timeout after ${timeoutMs}ms`)
    throw e
  } finally {
    window.clearTimeout(id)
  }
}

function parseUiEvent(line: string): UiEvent {
  if (line === '[closed]') return { kind: 'ws_closed' }
  try {
    const raw = JSON.parse(line) as Record<string, unknown>
    return {
      kind: 'json',
      type: typeof raw.type === 'string' ? raw.type : undefined,
      ts_ms: typeof raw.ts_ms === 'number' ? raw.ts_ms : undefined,
      run_id: typeof raw.run_id === 'string' ? raw.run_id : undefined,
      agent_id: typeof raw.agent_id === 'string' ? raw.agent_id : undefined,
      role: typeof raw.role === 'string' ? raw.role : undefined,
      phase: typeof raw.phase === 'string' ? raw.phase : undefined,
      provider: typeof raw.provider === 'string' ? raw.provider : undefined,
      model: typeof raw.model === 'string' ? raw.model : undefined,
      text: typeof raw.text === 'string' ? raw.text : undefined,
      topic: typeof raw.topic === 'string' ? raw.topic : undefined,
      seed_urls: Array.isArray(raw.seed_urls) ? (raw.seed_urls as string[]) : undefined,
      raw,
    }
  } catch {
    return { kind: 'raw', text: line }
  }
}

type Preset = 'all' | 'chat' | 'progress' | 'errors'

function fmtTime(ts_ms?: number): string {
  if (!ts_ms) return ''
  try {
    const d = new Date(ts_ms)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return ''
  }
}

function badgeTone(type?: string): 'neutral' | 'good' | 'warn' | 'bad' | 'agent' {
  if (!type) return 'neutral'
  if (type === 'run_start') return 'good'
  if (type === 'run_cancelled') return 'warn'
  if (type.endsWith('_error') || type === 'run_error') return 'bad'
  if (type.startsWith('debate_')) return 'agent'
  if (type.endsWith('_raw')) return 'neutral'
  return 'neutral'
}

function EventCard({ ev }: { ev: UiEvent }) {
  if (ev.kind === 'ws_closed') {
    return (
      <div className="eventCard eventCard--system">
        <div className="eventHeader">
          <span className="badge badge--warn">ws</span>
          <span className="eventTitle">接続が閉じられました</span>
        </div>
      </div>
    )
  }

  if (ev.kind === 'raw') {
    return (
      <div className="eventCard">
        <div className="eventHeader">
          <span className="badge badge--neutral">raw</span>
          <span className="eventTitle">メッセージ</span>
        </div>
        <pre className="eventBody">{ev.text}</pre>
      </div>
    )
  }

  const tone = badgeTone(ev.type)
  const badgeClass =
    tone === 'good'
      ? 'badge--good'
      : tone === 'warn'
        ? 'badge--warn'
        : tone === 'bad'
          ? 'badge--bad'
          : tone === 'agent'
            ? 'badge--agent'
            : 'badge--neutral'

  const title =
    ev.type === 'run_start'
      ? `run_start${ev.topic ? `: ${ev.topic}` : ''}`
      : ev.type
        ? ev.type
        : 'event'

  const metaBits = [fmtTime(ev.ts_ms), ev.agent_id ? `agent: ${ev.agent_id}` : '', ev.run_id ? `run: ${ev.run_id}` : ''].filter(Boolean)
  const subMeta = [
    ev.role ? `role: ${ev.role}` : '',
    ev.provider ? `provider: ${ev.provider}` : '',
    ev.model ? `model: ${ev.model}` : '',
    ev.phase ? `phase: ${ev.phase}` : '',
  ].filter(Boolean)

  const bodyText = ev.text
  const showRawFallback = !bodyText && Object.keys(ev.raw ?? {}).length > 0
  const isLong = typeof bodyText === 'string' && bodyText.length > 700
  const preview = isLong ? bodyText.slice(0, 420).trimEnd() + '\n…' : bodyText

  return (
    <div className={`eventCard ${ev.agent_id ? 'eventCard--agent' : ''}`}>
      <div className="eventHeader">
        <span className={`badge ${badgeClass}`}>{ev.type ?? 'json'}</span>
        <span className="eventTitle">{title}</span>
        {metaBits.length > 0 && <span className="eventMeta">{metaBits.join(' • ')}</span>}
      </div>
      {subMeta.length > 0 && <div className="eventSubMeta">{subMeta.join(' • ')}</div>}
      {bodyText && !isLong && <pre className="eventBody">{bodyText}</pre>}
      {bodyText && isLong && (
        <details className="eventDetails" open={false}>
          <summary>本文を表示（長文）</summary>
          <pre className="eventBody">{preview}</pre>
          <pre className="eventBody">{bodyText}</pre>
        </details>
      )}
      {showRawFallback && (
        <details className="eventDetails">
          <summary>raw JSON</summary>
          <pre className="eventBody">{JSON.stringify(ev.raw, null, 2)}</pre>
        </details>
      )}
    </div>
  )
}

function isErrorType(t?: string): boolean {
  if (!t) return false
  return t.endsWith('_error') || t === 'run_error'
}

function isChatType(t?: string): boolean {
  if (!t) return false
  return (
    t === 'brief_raw' ||
    t === 'debate_notes' ||
    t === 'synthesis_chunk' ||
    t === 'verify_notes' ||
    t === 'panorama_raw'
  )
}

function isProgressType(t?: string): boolean {
  if (!t) return false
  return (
    t === 'run_start' ||
    t === 'round_start' ||
    t === 'synthesis_start' ||
    t === 'fetch_start' ||
    t === 'fetch_done' ||
    t === 'search_query' ||
    t === 'search_skipped' ||
    t === 'llm_call_start' ||
    t === 'llm_retry' ||
    t === 'run_done' ||
    t === 'run_cancelled'
  )
}

function matchesPreset(ev: UiEvent, preset: Preset): boolean {
  if (preset === 'all') return true
  if (ev.kind !== 'json') return false
  if (preset === 'errors') return isErrorType(ev.type) || ev.type === 'debate_agent_error'
  if (preset === 'chat') return isChatType(ev.type) || ev.type?.endsWith('_chunk') === true
  if (preset === 'progress') return isProgressType(ev.type) || ev.type === 'debate_agent_start'
  return true
}

function StatusSummary({ events }: { events: UiEvent[] }) {
  const jsonEvents = events.filter((e) => e.kind === 'json') as Extract<UiEvent, { kind: 'json' }>[]
  const lastTs = Math.max(0, ...jsonEvents.map((e) => e.ts_ms ?? 0))
  const lastAgeSec = lastTs ? Math.max(0, (Date.now() - lastTs) / 1000) : null

  const lastType = [...jsonEvents].reverse().find((e) => e.type)?.type
  const phase =
    lastType?.startsWith('brief') ? 'brief'
      : lastType?.startsWith('debate') || lastType === 'debate_agent_start' ? 'debate'
        : lastType?.startsWith('synthesis') ? 'synthesis'
          : lastType?.startsWith('verify') ? 'verify'
            : lastType?.startsWith('panorama') ? 'panorama'
              : lastType?.startsWith('round') ? 'round'
                : lastType ?? 'idle'

  const latestByAgent = new Map<string, Extract<UiEvent, { kind: 'json' }>>()
  for (const e of jsonEvents) {
    if (!e.agent_id) continue
    const prev = latestByAgent.get(e.agent_id)
    if (!prev || (e.ts_ms ?? 0) >= (prev.ts_ms ?? 0)) latestByAgent.set(e.agent_id, e)
  }

  const failedAgents = [...latestByAgent.values()]
    .filter((e) => isErrorType(e.type) || e.type === 'debate_agent_error')
    .map((e) => e.agent_id!)

  const runningAgents = [...latestByAgent.values()]
    .filter((e) => e.type === 'llm_call_start' || e.type === 'debate_agent_start')
    .filter((e) => (e.ts_ms ?? 0) > Date.now() - 60_000)
    .map((e) => e.agent_id!)

  return (
    <div className="summaryBar">
      <div className="summaryItem"><span className="summaryLabel">phase</span><span className="summaryValue">{phase}</span></div>
      <div className="summaryItem"><span className="summaryLabel">running</span><span className="summaryValue">{runningAgents.length ? runningAgents.join(', ') : 'none'}</span></div>
      <div className="summaryItem"><span className="summaryLabel">failed</span><span className="summaryValue">{failedAgents.length ? failedAgents.join(', ') : 'none'}</span></div>
      <div className="summaryItem"><span className="summaryLabel">last_event</span><span className="summaryValue">{lastAgeSec == null ? '—' : `${lastAgeSec.toFixed(1)}s ago`}</span></div>
    </div>
  )
}

function providerIcon(provider?: string): string | null {
  const p = (provider ?? '').toLowerCase()
  if (!p) return null
  const m: Record<string, string> = {
    openai: new URL('./assets/providers/openai.svg', import.meta.url).toString(),
    anthropic: new URL('./assets/providers/anthropic.svg', import.meta.url).toString(),
    gemini: new URL('./assets/providers/gemini.svg', import.meta.url).toString(),
    plamo: new URL('./assets/providers/plamo.png', import.meta.url).toString(),
  }
  return m[p] ?? null
}

type ArenaNode = {
  id: string
  label: string
  roleLabel: string
  provider?: string
  model?: string
  state: 'idle' | 'running' | 'error'
}

function Arena({
  events,
  yamlPlan,
  preset,
  onPresetChange,
  selectedAgent,
  onSelectAgent,
}: {
  events: UiEvent[]
  yamlPlan: Record<string, AgentRoute>
  preset: Preset
  onPresetChange: (p: Preset) => void
  selectedAgent: string | null
  onSelectAgent: (a: string | null) => void
}) {
  const jsonEvents = events.filter((e) => e.kind === 'json') as Extract<UiEvent, { kind: 'json' }>[]

  const runStartRoutes = useMemo(() => {
    let last: Record<string, AgentRoute> | null = null
    for (const e of jsonEvents) {
      if (e.type !== 'run_start') continue
      const parsed = parseRunStartRoutes(e.raw)
      if (parsed) last = parsed
    }
    return last
  }, [jsonEvents])

  const fixedOrder = [
    'user',
    'brief',
    'debater_general',
    'debater_contrarian',
    'debater_quant',
    'debater_risk',
    'debater_practice',
    'debater_sources',
    'debater_free',
    'synthesis',
    'verify',
    'panorama',
  ]

  const displayName: Record<string, string> = {
    user: 'User',
    brief: 'Brief',
    debater_general: 'General',
    debater_contrarian: 'Contra',
    debater_quant: 'Quant',
    debater_risk: 'Risk',
    debater_practice: 'Ops',
    debater_sources: 'Sources',
    debater_free: 'Free',
    synthesis: 'Synthesis',
    verify: 'Verify',
    panorama: 'Panorama',
  }

  const lastByAgent = new Map<string, Extract<UiEvent, { kind: 'json' }>>()
  for (const e of jsonEvents) {
    if (!e.agent_id) continue
    const prev = lastByAgent.get(e.agent_id)
    if (!prev || (e.ts_ms ?? 0) >= (prev.ts_ms ?? 0)) lastByAgent.set(e.agent_id, e)
  }

  const nodes: ArenaNode[] = fixedOrder.map((id) => {
    if (id === 'user') {
      return { id, label: 'User', roleLabel: 'topic', state: selectedAgent === 'user' ? 'running' : 'idle' }
    }
    const e = lastByAgent.get(id)
    const planned = runStartRoutes?.[id] ?? yamlPlan[id]
    const provider = e?.provider ?? planned?.provider
    const model = e?.model ?? planned?.model
    const roleLabel =
      id === 'brief' ? 'brief'
        : id === 'synthesis' ? 'synthesis'
          : id === 'verify' ? 'verify'
            : id === 'panorama' ? 'panorama'
              : id.startsWith('debater_') ? 'debate'
                : e?.role ?? 'agent'
    const state: ArenaNode['state'] =
      e && (isErrorType(e.type) || e.type === 'debate_agent_error') ? 'error'
        : e && (e.type === 'llm_call_start' || e.type === 'debate_agent_start') && (e.ts_ms ?? 0) > Date.now() - 60_000 ? 'running'
          : 'idle'
    return {
      id,
      label: displayName[id] ?? id,
      roleLabel,
      provider,
      model,
      state,
    }
  })

  const filtered = events.filter((e) => matchesPreset(e, preset)).filter((e) => {
    if (!selectedAgent || selectedAgent === 'user') return true
    return e.kind !== 'json' ? false : e.agent_id === selectedAgent
  })

  return (
    <div className="arenaLayout">
      <div className="arenaStage">
        <div className="arenaRing">
          {nodes.map((n, idx) => {
            const angle = (idx / nodes.length) * Math.PI * 2 - Math.PI / 2
            const r = 170
            const x = Math.cos(angle) * r
            const y = Math.sin(angle) * r
            const icon = providerIcon(n.provider)
            const plamo = (n.provider ?? '').toLowerCase() === 'plamo'
            return (
              <button
                key={n.id}
                className={`arenaNode arenaNode--${n.state} ${selectedAgent === n.id ? 'arenaNode--selected' : ''} ${plamo ? 'arenaNode--plamoMark' : ''}`}
                style={{ transform: `translate(${x}px, ${y}px)` }}
                onClick={() => onSelectAgent(selectedAgent === n.id ? null : n.id)}
                title={[n.label, `役割: ${n.roleLabel}`, n.provider && n.model ? `${n.provider} / ${n.model}` : n.provider ?? ''].filter(Boolean).join(' • ')}
              >
                <div className={`arenaFace ${plamo ? 'arenaFace--plamo' : ''}`}>
                  {icon ? <img className={`arenaIcon ${plamo ? 'arenaIcon--plamo' : ''}`} src={icon} alt={n.provider ?? 'provider'} /> : <span className="arenaMonogram">{(n.provider ?? n.id).slice(0, 2).toUpperCase()}</span>}
                </div>
                <div className="arenaNodeText">
                  <div className="arenaNodeLabel">{n.label}</div>
                  <div className="arenaNodeSub">
                    <span className="arenaNodeRole">{n.roleLabel}</span>
                    {n.provider ? <span className="arenaNodeProv">{` · ${n.provider}`}</span> : null}
                  </div>
                </div>
              </button>
            )
          })}
          <div className="arenaCenter">
            <div className="arenaCenterTitle">マルチエージェント</div>
            <div className="arenaCenterSub">
              {selectedAgent ? `選択中: ${selectedAgent}` : '各ノードは固定役割。アイコンはプロバイダ（計画は routing / run_start）'}
            </div>
          </div>
        </div>
      </div>

      <div className="arenaSide">
        <div className="arenaControls">
          <div className="presetRow">
            <button className={preset === 'all' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => onPresetChange('all')}>すべて</button>
            <button className={preset === 'chat' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => onPresetChange('chat')}>会話</button>
            <button className={preset === 'progress' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => onPresetChange('progress')}>進捗</button>
            <button className={preset === 'errors' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => onPresetChange('errors')}>エラー</button>
          </div>
          <div className="arenaHint">
            {selectedAgent ? (
              <>
                <span>絞り込み中: <code>{selectedAgent}</code></span>{' '}
                <button className="linkBtn" onClick={() => onSelectAgent(null)}>解除</button>
              </>
            ) : (
              <span>ノードをクリックして絞り込み</span>
            )}
          </div>
        </div>

        <div className="eventsPanel arenaEvents" role="log" aria-live="polite">
          {filtered.length === 0 ? (
            <div className="eventsEmpty">該当イベントがありません</div>
          ) : (
            filtered.map((ev, idx) => <EventCard key={idx} ev={ev} />)
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<'run' | 'report' | 'settings' | 'arena'>('run')
  const [providers, setProviders] = useState<string[]>([])

  const [topic, setTopic] = useState('')
  const [seedUrls, setSeedUrls] = useState('')
  const [runId, setRunId] = useState<string | null>(null)
  const [events, setEvents] = useState<string[]>([])
  const eventsView = events.map(parseUiEvent)
  const [selectedEventIdx, setSelectedEventIdx] = useState<number | null>(null)
  const [reportMd, setReportMd] = useState<string>('')
  const [runs, setRuns] = useState<RunsList['runs']>([])
  const [graph, setGraph] = useState<Graph | null>(null)
  const [files, setFiles] = useState<FileList | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const [startPhase, setStartPhase] = useState<string | null>(null)

  const [routingYaml, setRoutingYaml] = useState<string>('')
  const [arenaYamlPlan, setArenaYamlPlan] = useState<Record<string, AgentRoute>>({})
  const wsRef = useRef<WebSocket | null>(null)
  const [wsError, setWsError] = useState<string | null>(null)
  const eventsScrollRef = useRef<HTMLDivElement | null>(null)
  const autoScrollRef = useRef(true)
  const wsBufferRef = useRef<string[]>([])
  const wsFlushTimerRef = useRef<number | null>(null)
  const [preset, setPreset] = useState<Preset>('all')
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [runSetupOpen, setRunSetupOpen] = useState(false)

  useEffect(() => {
    jTimeout<ProviderInfo>('/api/providers', undefined, 3000)
      .then((x) => setProviders(x.available))
      .catch(() => setProviders([]))
  }, [])

  async function loadRoutingPreview() {
    try {
      const p = await jTimeout<RoutingPreview>('/api/settings/routing-preview', undefined, 10_000)
      setArenaYamlPlan(flattenRoutingPreview(p))
    } catch {
      setArenaYamlPlan({})
    }
  }

  useEffect(() => {
    void loadRoutingPreview()
  }, [])

  useEffect(() => {
    if (tab === 'arena') void loadRoutingPreview()
  }, [tab])

  useEffect(() => {
    const el = eventsScrollRef.current
    if (!el) return
    if (!autoScrollRef.current) return
    el.scrollTop = el.scrollHeight
  }, [events.length])

  function onEventsScroll() {
    const el = eventsScrollRef.current
    if (!el) return
    const threshold = 32
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= threshold
    autoScrollRef.current = atBottom
  }

  function scheduleWsFlush() {
    if (wsFlushTimerRef.current != null) return
    wsFlushTimerRef.current = window.requestAnimationFrame(() => {
      wsFlushTimerRef.current = null
      const batch = wsBufferRef.current.splice(0, wsBufferRef.current.length)
      if (batch.length === 0) return
      setEvents((prev) => [...prev, ...batch])
    })
  }

  useEffect(() => {
    if (tab !== 'run') return
    if (events.length > 0) return
    // 初回導線: 実行設定を最初だけ開く
    setRunSetupOpen(true)
  }, [tab])

  async function refreshRuns() {
    const x = await jTimeout<RunsList>('/api/runs', undefined, 5000)
    setRuns(x.runs)
  }

  async function loadEventsTail(id: string) {
    const x = await jTimeout<EventsTail>(`/api/runs/${id}/events`, undefined, 5000)
    setEvents(x.events)
  }

  async function start() {
    if (isStarting) return
    setIsStarting(true)
    setStartError(null)
    setStartPhase('run作成中…')
    try {
      setEvents([])
      setReportMd('')
      setGraph(null)

      const runResp = await jTimeout<Record<string, unknown>>(
        '/api/runs',
        {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        },
        10_000,
      )

      const run_id = typeof runResp.run_id === 'string' ? runResp.run_id : null
      const apiErr = typeof runResp.error === 'string' ? runResp.error : null
      if (!run_id) {
        setStartError(apiErr ? `Run作成に失敗しました: ${apiErr}` : 'Run作成に失敗しました（run_id が返りませんでした）')
        return
      }

      setRunId(run_id)

      setStartPhase('WebSocket接続中…')
      wsRef.current?.close()
      setWsError(null)
      // Vite(5173)ではWS proxyが効かない場合があるため、FastAPI(8000)へ直接接続する
      const ws = new WebSocket(`ws://127.0.0.1:8000/api/runs/${run_id}/events`)
      ws.onmessage = (ev) => {
        wsBufferRef.current.push(ev.data)
        scheduleWsFlush()
      }
      ws.onclose = () => {
        wsBufferRef.current.push('[closed]')
        scheduleWsFlush()
      }
      ws.onerror = () => setWsError('WebSocket接続に失敗しました（FastAPIが起動しているか確認してください）')
      wsRef.current = ws

      const seeds = seedUrls.split('\n').map((s) => s.trim()).filter(Boolean)

      if (files && files.length > 0) {
        setStartPhase('ファイルアップロード中…')
        const fd = new FormData()
        for (const f of Array.from(files)) fd.append('files', f)
        const up = await fetch(`/api/runs/${run_id}/files`, { method: 'POST', body: fd })
        if (!up.ok) throw new Error(`files upload failed: ${up.status} ${up.statusText}`)
      }

      setStartPhase('バックエンド開始要求中…')
      await jTimeout(
        `/api/runs/${run_id}/start`,
        {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, seed_urls: seeds }),
        },
        10_000,
      )
      setStartPhase(null)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setStartError(`Startに失敗しました: ${msg}`)
    } finally {
      setIsStarting(false)
      setStartPhase(null)
    }
  }

  async function stop() {
    if (!runId) return
    await jTimeout(`/api/runs/${runId}/stop`, { method: 'POST' }, 5000)
  }

  async function loadReport() {
    if (!runId) return
    const md = await tTimeout(`/api/runs/${runId}/report.md`, undefined, 10_000)
    setReportMd(md)
    try {
      const g = await jTimeout<Graph>(`/api/runs/${runId}/graph.json`, undefined, 10_000)
      if (!('error' in (g as any))) setGraph(g)
    } catch {}
    setTab('report')
  }

  async function loadRouting() {
    const y = await tTimeout('/api/settings/routing.yaml', undefined, 10_000)
    setRoutingYaml(y)
  }

  async function saveRouting() {
    await jTimeout('/api/settings/routing.yaml', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: routingYaml }),
    }, 10_000)
    await loadRoutingPreview()
  }

  return (
    <div className="appShell">
      <header className="appHeader">
        <div className="appBrand">DeepResearch MAS</div>
        <nav className="appNav" aria-label="メイン">
          <button type="button" className={tab === 'run' ? 'navBtn navBtn--active' : 'navBtn'} onClick={() => setTab('run')}>
            Run
          </button>
          <button type="button" className={tab === 'report' ? 'navBtn navBtn--active' : 'navBtn'} onClick={() => setTab('report')} disabled={!runId}>
            Report
          </button>
          <button type="button" className={tab === 'arena' ? 'navBtn navBtn--active' : 'navBtn'} onClick={() => setTab('arena')} disabled={!runId}>
            Arena
          </button>
          <button
            type="button"
            className={tab === 'settings' ? 'navBtn navBtn--active' : 'navBtn'}
            onClick={async () => {
              setTab('settings')
              await loadRouting()
            }}
          >
            Settings
          </button>
        </nav>
        <div className="appProviders">providers: {providers.join(', ') || 'none'}</div>
      </header>

      <StatusSummary events={eventsView} />

      <main className="appMain">
        {tab === 'run' && (
          <div className="runLayout">
            <div className="runToolbar">
              <button type="button" className="ghostBtn" onClick={() => setRunSetupOpen((o) => !o)}>
                {runSetupOpen ? '実行設定を隠す' : '実行設定を表示'}
              </button>
              <button
                type="button"
                className="ghostBtn"
                onClick={async () => {
                  await refreshRuns()
                  setRunSetupOpen(true)
                }}
              >
                既存runを開く
              </button>
              {runId && (
                <span style={{ fontSize: 12, opacity: 0.75 }}>
                  run: <code>{runId}</code>
                </span>
              )}
            </div>

            {runSetupOpen && (
              <div className="runFormCard">
                <label htmlFor="topic">Topic</label>
                <textarea id="topic" value={topic} onChange={(e) => setTopic(e.target.value)} style={{ width: '100%', height: 100 }} />
                <label htmlFor="seeds">Seed URLs（1行に1つ）</label>
                <textarea id="seeds" value={seedUrls} onChange={(e) => setSeedUrls(e.target.value)} style={{ width: '100%', height: 88 }} />
                <label htmlFor="files">Files（PDF / MD）</label>
                <input id="files" type="file" multiple accept=".pdf,.md,.markdown" onChange={(e) => setFiles(e.target.files)} />
                <div className="runFormActions">
                  <button type="button" className="primaryBtn" onClick={start} disabled={!topic.trim() || isStarting}>
                    {isStarting ? 'Starting…' : 'Start'}
                  </button>
                  <button type="button" className="secondaryBtn" onClick={stop} disabled={!runId}>
                    Stop
                  </button>
                  <button type="button" className="secondaryBtn" onClick={loadReport} disabled={!runId}>
                    Load report
                  </button>
                  <button type="button" className="secondaryBtn" onClick={refreshRuns}>
                    Refresh runs
                  </button>
                </div>
                {isStarting && startPhase && <div style={{ opacity: 0.8, marginTop: 10, fontSize: 13 }}>{startPhase}</div>}
                {startError && <div style={{ color: 'tomato', marginTop: 10, fontSize: 13 }}>{startError}</div>}
                {runs.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: 'var(--text-h)' }}>Recent runs</div>
                    <ul style={{ paddingLeft: 18, margin: 0, textAlign: 'left' }}>
                      {runs.slice(0, 10).map((r) => (
                        <li key={r.run_id} style={{ marginBottom: 6 }}>
                          <button type="button" className="secondaryBtn" style={{ padding: '4px 10px', fontSize: 12 }} onClick={async () => { setRunId(r.run_id); await loadEventsTail(r.run_id) }}>
                            open
                          </button>{' '}
                          <code>{r.run_id}</code> {r.has_report ? '· report' : ''}
                          {r.has_pdf ? ' · pdf' : ''}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <section className="liveFeed" aria-label="ライブイベント">
              <div className="liveFeedHeader">
                <div className="liveFeedTitle">
                  Live events
                  <span>WebSocket ストリーム · プリセットで絞り込み</span>
                </div>
                <div className="presetRow presetRow--compact" style={{ margin: 0 }}>
                  <button type="button" className={preset === 'all' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => setPreset('all')}>
                    すべて
                  </button>
                  <button type="button" className={preset === 'chat' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => setPreset('chat')}>
                    会話
                  </button>
                  <button type="button" className={preset === 'progress' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => setPreset('progress')}>
                    進捗
                  </button>
                  <button type="button" className={preset === 'errors' ? 'presetBtn presetBtn--on' : 'presetBtn'} onClick={() => setPreset('errors')}>
                    エラー
                  </button>
                </div>
              </div>
              {wsError && <div style={{ color: 'tomato', padding: '0 4px', fontSize: 13 }}>{wsError}</div>}
              <div className="eventsPanel eventsPanel--fullscreen" ref={eventsScrollRef} role="log" aria-live="polite" onScroll={onEventsScroll}>
                {eventsView.length === 0 ? (
                  <div className="eventsEmpty">まだイベントはありません。Start で run を開始するか、Recent runs から開いてください。</div>
                ) : (
                  eventsView
                    .map((ev, idx) => ({ ev, idx }))
                    .filter(({ ev }) => matchesPreset(ev, preset))
                    .map(({ ev, idx }) => (
                      <button key={idx} type="button" className="eventRowBtn" onClick={() => setSelectedEventIdx(idx)}>
                        <EventCard ev={ev} />
                      </button>
                    ))
                )}
              </div>
            </section>

            {selectedEventIdx != null && eventsView[selectedEventIdx] && (
              <div className="drawerOverlay" role="presentation" onClick={() => setSelectedEventIdx(null)}>
                <aside className="drawer" role="dialog" aria-label="イベント詳細" onClick={(e) => e.stopPropagation()}>
                  <div className="drawerHeader">
                    <div className="drawerTitle">Event details</div>
                    <button type="button" className="secondaryBtn" onClick={() => setSelectedEventIdx(null)}>
                      Close
                    </button>
                  </div>
                  <div className="drawerBody">
                    {(() => {
                      const ev = eventsView[selectedEventIdx]
                      return (
                        <>
                    <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 8 }}>
                      index: <code>{selectedEventIdx}</code>
                    </div>
                    <pre className="drawerPre">
                      {ev.kind === 'json'
                        ? JSON.stringify(ev.raw, null, 2)
                        : ev.kind === 'raw'
                          ? ev.text
                          : '[closed]'}
                    </pre>
                    {ev.kind === 'json' && (
                      <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                        <button
                          type="button"
                          className="secondaryBtn"
                          onClick={async () => {
                            await navigator.clipboard.writeText(JSON.stringify(ev.raw, null, 2))
                          }}
                        >
                          Copy JSON
                        </button>
                        {typeof ev.raw?.run_id === 'string' && (
                          <button
                            type="button"
                            className="secondaryBtn"
                            onClick={async () => {
                              setRunId(String(ev.raw.run_id))
                              await loadEventsTail(String(ev.raw.run_id))
                            }}
                          >
                            Open run
                          </button>
                        )}
                      </div>
                    )}
                        </>
                      )
                    })()}
                  </div>
                </aside>
              </div>
            )}
          </div>
        )}

        {tab === 'report' && (
          <div style={{ marginTop: 12 }}>
            <h3 style={{ color: 'var(--text-h)', textAlign: 'left' }}>Report (Markdown)</h3>
            <pre style={{ whiteSpace: 'pre-wrap', background: 'var(--code-bg)', color: 'var(--text-h)', padding: 16, borderRadius: 14, border: '1px solid var(--border)', textAlign: 'left' }}>
              {reportMd || '(load report)'}
            </pre>
            {graph && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ color: 'var(--text-h)', textAlign: 'left' }}>Graph (simple)</h3>
                <pre style={{ whiteSpace: 'pre-wrap', background: 'var(--code-bg)', color: 'var(--text-h)', padding: 16, borderRadius: 14, border: '1px solid var(--border)', textAlign: 'left' }}>
                  {JSON.stringify(graph, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {tab === 'settings' && (
          <div style={{ marginTop: 12, textAlign: 'left' }}>
            <h3 style={{ color: 'var(--text-h)' }}>routing.yaml</h3>
            <textarea value={routingYaml} onChange={(e) => setRoutingYaml(e.target.value)} style={{ width: '100%', height: 360, borderRadius: 12, border: '1px solid var(--border)', padding: 12 }} />
            <div className="runFormActions" style={{ marginTop: 12 }}>
              <button type="button" className="secondaryBtn" onClick={loadRouting}>
                Reload
              </button>
              <button type="button" className="primaryBtn" onClick={saveRouting}>
                Save
              </button>
            </div>
          </div>
        )}

        {tab === 'arena' && (
          <div className="arenaRoot" style={{ marginTop: 12 }}>
            <Arena
              events={eventsView}
              yamlPlan={arenaYamlPlan}
              preset={preset}
              onPresetChange={setPreset}
              selectedAgent={selectedAgent}
              onSelectAgent={setSelectedAgent}
            />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
