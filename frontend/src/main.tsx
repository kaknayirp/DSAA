import { useCallback, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { api, Comparison, Health, Job, labels, Run, strategies, SystemMetrics, Token } from './api'
import { BenchmarkLab, PromptForm, Tuning } from './experiments'
import { bytes, CompareResults, MetricTable, percent, TokenExplorer } from './results'
import { IntegrationNotice, strategyRequirements } from './integration'
import './style.css'

const pages = {
  overview: ['Overview', 'Monitor host memory, model readiness, and the latest cache measurements.'],
  playground: ['LLM Playground', 'Run one prompt against a selected cache strategy.'],
  explorer: ['Cache Explorer', 'Inspect tokenizer positions, retention decisions, and scoring signals.'],
  compare: ['Compare Modes', 'Run Original and DSAA with the same prompt and generation limit.'],
  benchmarks: ['Benchmark Lab', 'Evaluate long-context recall across the available cache strategies.'],
  tuning: ['Algorithm Tuning', 'Configure weights, retention budget, and threshold for future runs.'],
  reports: ['Reports', 'Download completed experiments with settings, per-case results, and aggregate metrics.'],
  configuration: ['Configuration', 'Inspect the connected model and registered strategies.'],
} as const
type Page = keyof typeof pages

function currentPage(): Page {
  const key = window.location.hash.slice(1)
  return Object.hasOwn(pages, key) ? key as Page : 'overview'
}

function ModelStatus({ health, system }: { health: Health; system?: SystemMetrics }) {
  const pending = strategies.filter((key) => !health.strategies[key])
  return <section className="panel">
    <h2>Runtime configuration</h2>
    <dl className="runtime">
      <div><dt>Model</dt><dd>{health.model ?? 'Not configured'}</dd></div>
      <div><dt>Device</dt><dd>{health.device ?? 'Unavailable'}</dd></div>
      <div><dt>Process CUDA tensor memory</dt><dd>{system?.gpu_memory_bytes == null ? 'Unavailable' : bytes(system.gpu_memory_bytes)}</dd></div>
    </dl>
    <div className="strategy-list">{strategies.map((key) => <span className="badge" key={key}>
      {labels[key]} · {health.strategies[key] ? 'Ready' : key === 'original' ? 'Not configured' : 'Not implemented'}
    </span>)}</div>
    {pending.length > 0 && <IntegrationNotice title="Pending strategy integrations">
      <ul>{pending.map((key) => <li key={key}><strong>{labels[key]}.</strong> {strategyRequirements[key]}</li>)}</ul>
    </IntegrationNotice>}
  </section>
}

function Overview({ health, system, history, comparison, latest }: {
  health: Health; system?: SystemMetrics; history: number[]; comparison?: Comparison; latest?: Run
}) {
  const measured = comparison?.source === 'measured' ? comparison : undefined
  const savings = measured?.baseline.kv_cache_bytes
    ? 1 - measured.dsaa.kv_cache_bytes / measured.baseline.kv_cache_bytes : null
  const points = history.map((value, index) => `${index * 400 / Math.max(1, history.length - 1)},${100 - value * 100}`).join(' ')
  return <>
    <div className="cards">
      {[
        ['Host RAM used', system ? bytes(system.ram_used_bytes) : '—'],
        ['Host memory pressure', system ? percent(system.memory_pressure) : '—'],
        ['API process RAM', system ? bytes(system.process_ram_bytes) : '—'],
        ['Measured KV savings', savings == null ? '—' : percent(savings)],
      ].map(([label, value]) => <article className="panel kpi" key={label}><p>{label}</p><strong>{value}</strong></article>)}
    </div>
    <section className="panel">
      <div className="section-heading"><h2>Host memory pressure</h2><span className="badge">{system?.busy ? 'Compute busy' : system ? 'Compute idle' : 'Awaiting telemetry'}</span></div>
      {history.length > 1 ? <div className="chart">
        <div className="chart-scale"><span>100%</span><span>50%</span><span>0%</span></div>
        <svg viewBox="0 0 400 100" preserveAspectRatio="none" role="img" aria-label="Host memory pressure over recent samples">
          <line x1="0" y1="50" x2="400" y2="50" /><polyline points={points} vectorEffect="non-scaling-stroke" />
        </svg>
      </div> : <p className="muted">Waiting for additional RAM samples.</p>}
      {system && <p className="muted">Last sample {new Date(system.timestamp).toLocaleTimeString()} · {bytes(system.ram_total_bytes)} total</p>}
    </section>
    <ModelStatus health={health} system={system} />
    {measured ? <CompareResults result={measured} /> : latest ? <section className="panel">
      <h2>Latest measured run</h2><MetricTable columns={[{ label: labels[latest.strategy], metrics: latest.metrics }]} />
    </section> : <section className="panel"><h2>Measured cache results</h2><p className="muted">Run inference or a comparison to populate this view.</p></section>}
  </>
}

function App() {
  const [page, setPage] = useState(currentPage)
  const [health, setHealth] = useState<Health>()
  const [system, setSystem] = useState<SystemMetrics>()
  const [history, setHistory] = useState<number[]>([])
  const [demo, setDemo] = useState<Comparison>()
  const [comparison, setComparison] = useState<Comparison>()
  const [latest, setLatest] = useState<Run>()
  const [jobs, setJobs] = useState<Job[]>([])
  const [startupError, setStartupError] = useState('')
  const [jobError, setJobError] = useState('')
  const [lookup, setLookup] = useState('')
  const [loaded, setLoaded] = useState<{ id: string; entries: Token[] }>()
  const [lookupError, setLookupError] = useState('')

  useEffect(() => {
    const navigate = () => setPage(currentPage())
    window.addEventListener('hashchange', navigate)
    return () => window.removeEventListener('hashchange', navigate)
  }, [])

  useEffect(() => {
    Promise.all([api<Health>('health'), api<Comparison>('demo')])
      .then(([status, sample]) => { setHealth(status); setDemo(sample) })
      .catch(() => setStartupError('Could not reach FastAPI. Start the backend and reload.'))
  }, [])

  useEffect(() => {
    let active = true
    let streaming = false
    function receive(sample: SystemMetrics) {
      if (!active) return
      setSystem(sample)
      setHistory((values) => [...values.slice(-29), sample.memory_pressure])
    }
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(`${protocol}//${location.host}/ws/metrics`)
    socket.onmessage = (event) => { streaming = true; receive(JSON.parse(event.data)) }
    socket.onclose = () => { streaming = false }
    const poll = () => {
      if (!streaming) api<SystemMetrics>('system/metrics').then(receive).catch(() => {})
    }
    poll()
    const timer = window.setInterval(poll, 2500)
    return () => { active = false; window.clearInterval(timer); socket.close() }
  }, [])

  const refreshJobs = useCallback(async () => {
    setJobs(await api<Job[]>('benchmarks'))
    setJobError('')
  }, [])

  useEffect(() => {
    if (page !== 'benchmarks' && page !== 'reports') return
    const poll = () => refreshJobs().catch((error) => setJobError(error.message))
    poll()
    const timer = window.setInterval(poll, 2000)
    return () => window.clearInterval(timer)
  }, [page, refreshJobs])

  const inspected = loaded ?? (comparison ? { id: comparison.dsaa_run_id, entries: comparison.entries }
    : latest ? { id: latest.id, entries: latest.entries } : undefined)

  return <div className="studio">
    <aside className="sidebar">
      <a className="brand" href="#overview"><span className="mark" aria-hidden="true">D</span><span>DSAA Studio<small>Adaptive Memory Intelligence</small></span></a>
      <nav aria-label="Studio navigation">
        {Object.entries(pages).map(([key, [label]], index) => (
          <a key={key} href={`#${key}`} aria-current={page === key ? 'page' : undefined}>
            <span aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>{label}
          </a>
        ))}
      </nav>
    </aside>
    <main>
      <header>
        <p className="eyebrow">DSAA / RESEARCH WORKSPACE</p>
        <h1>{pages[page][0]}</h1>
        <p>{pages[page][1]}</p>
      </header>
      {startupError && <p role="alert" className="error">{startupError}</p>}
      {!health && !startupError && <p role="status">Connecting to the backend…</p>}
      {health && <>
        {page === 'overview' && <Overview {...{ health, system, history, comparison, latest }} />}
        {page === 'configuration' && <ModelStatus health={health} system={system} />}
        {page === 'playground' && <>
          <PromptForm compare={false} health={health} submit={async (body) => {
            const run = await api<Run>('inference', 'POST', body)
            setLatest(run)
            setLoaded({ id: run.id, entries: run.entries })
          }} />
          {latest && <section className="panel"><div className="section-heading"><h2>{labels[latest.strategy]} response</h2>
            <a href="#explorer" onClick={() => setLoaded({ id: latest.id, entries: latest.entries })}>Inspect cache</a></div>
            <p className="answer-text">{latest.metrics.answer ?? 'No response returned.'}</p>
            <p className="muted">{latest.context_tokens ?? '—'} input tokens · Run <code>{latest.id}</code></p>
            <MetricTable columns={[{ label: labels[latest.strategy], metrics: latest.metrics }]} />
          </section>}
        </>}
        {page === 'compare' && <>
          <PromptForm compare health={health} submit={async ({ prompt, max_new_tokens }) => {
            const result = await api<Comparison>('compare', 'POST', { prompt, max_new_tokens })
            setComparison(result)
            setLoaded(undefined)
          }} />
          {(comparison ?? demo) && <CompareResults result={(comparison ?? demo)!} />}
          {comparison && <a className="action" href="#explorer" onClick={() => setLoaded(undefined)}>Inspect DSAA decisions</a>}
        </>}
        {page === 'explorer' && <>
          {(!inspected || inspected.id === '') && <IntegrationNotice
            title={health.strategies.dsaa ? 'Sample cache decisions' : 'Live DSAA token telemetry required'}>
            <p>{health.strategies.dsaa
              ? 'Run DSAA in the Playground or Compare Modes to inspect measured token decisions.'
              : 'The DSAA runner needs to return token IDs, positions, scores, and retention decisions. The explorer currently shows an illustrative sample.'}</p>
          </IntegrationNotice>}
          <form className="panel" onSubmit={async (event) => {
            event.preventDefault()
            setLookupError('')
            const id = lookup.trim()
            try { setLoaded({ id, entries: await api<Token[]>(`cache/${encodeURIComponent(id)}/tokens`) }) }
            catch (error) { setLookupError((error as Error).message) }
          }}>
            <div className="form-row"><label>Run ID<input value={lookup} onChange={(event) => setLookup(event.target.value)} /></label>
              <button disabled={!lookup.trim()}>Load run</button>
              <button type="button" className="secondary" onClick={() => { setLoaded({ id: '', entries: demo?.entries ?? [] }); setLookupError('') }}>Show sample</button></div>
            {inspected?.id && <p className="muted">Inspecting <code>{inspected.id}</code></p>}
            {lookupError && <p role="alert" className="error">{lookupError}</p>}
          </form>
          <TokenExplorer key={inspected?.id ?? (inspected ? 'comparison' : 'demo')}
            entries={inspected?.entries ?? demo?.entries ?? []} sample={!inspected || inspected.id === ''} />
        </>}
        {jobError && (page === 'benchmarks' || page === 'reports') && <p role="alert" className="error">{jobError}</p>}
        {page === 'benchmarks' && <BenchmarkLab health={health} jobs={jobs} refresh={refreshJobs} />}
        {page === 'tuning' && <Tuning available={health.strategies.dsaa} />}
        {page === 'reports' && <section className="panel">
          <h2>Completed experiments</h2>
          {!health.benchmark_available ? <IntegrationNotice title="Evaluation adapter required">
            <p>Connect the evaluation module and complete a benchmark to generate a downloadable report.</p>
          </IntegrationNotice> : !jobs.some((job) => job.status === 'completed') &&
            <p className="muted">Completed benchmarks will appear here.</p>}
          <ul className="reports">{jobs.filter((job) => job.status === 'completed').map((job) => (
            <li key={job.id}><span>{new Date(job.created_at).toLocaleString()} · {job.completed} runs</span>
              <a href={`/api/benchmarks/${job.id}/report`}>Download JSON</a></li>
          ))}</ul>
        </section>}
      </>}
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
