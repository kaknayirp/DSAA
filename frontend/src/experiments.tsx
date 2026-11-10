import { FormEvent, useEffect, useState } from 'react'
import { api, Health, Job, labels, Settings, strategies, Strategy } from './api'
import { MetricTable } from './results'
import { IntegrationNotice, strategyRequirements } from './integration'

const recallPrompt = [
  'The maintenance crew named its Arctic relay BLUE ORCHID.',
  ...Array.from({ length: 32 }, (_, i) =>
    `Log ${i + 1}: The crew calibrated sensors, checked battery levels, and recorded the weather.`),
  '\nWhat is the name of the Arctic relay? Reply with only its name.',
].join('\n')

export function PromptForm({ compare, health, submit }: {
  compare: boolean
  health: Health
  submit: (body: { prompt: string; max_new_tokens: number; strategy: Strategy }) => Promise<void>
}) {
  const [prompt, setPrompt] = useState('')
  const [strategy, setStrategy] = useState<Strategy>()
  const [limit, setLimit] = useState(64)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const selected = strategy ?? strategies.find((key) => health.strategies[key]) ?? 'original'
  const available = compare ? health.comparison_available : health.strategies[selected]

  async function run(event: FormEvent) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try { await submit({ prompt, max_new_tokens: limit, strategy: selected }) }
    catch (error) { setError((error as Error).message) }
    finally { setBusy(false) }
  }

  return <form className="panel" onSubmit={run}>
    <div className="section-heading">
      <label htmlFor="prompt">{compare ? 'Shared prompt' : 'Prompt'}</label>
      <button className="secondary" type="button" onClick={() => setPrompt(recallPrompt)}>Load recall challenge</button>
    </div>
    <textarea id="prompt" required rows={6} maxLength={100000} placeholder="Enter context and a question…" value={prompt}
      onChange={(event) => setPrompt(event.target.value)} />
    <div className="form-row">
      {!compare && <label>Cache strategy<select value={selected} onChange={(event) => setStrategy(event.target.value as Strategy)}>
        {strategies.map((key) => <option key={key} value={key} disabled={!health.strategies[key]}>
          {labels[key]}{health.strategies[key] ? '' : ' · Unavailable'}
        </option>)}
      </select></label>}
      <label>Max new tokens<input type="number" min="1" max="256" required value={limit || ''} onChange={(event) => setLimit(Number(event.target.value))} /></label>
      <button disabled={busy || !available || !prompt.trim()}>{busy ? 'Running…' : compare ? 'Run comparison' : 'Run inference'}</button>
    </div>
    {!available && <IntegrationNotice title={compare ? 'Comparison runners required' : 'Model runner required'}>
      {compare ? <>
        {!health.strategies.original && <p>{strategyRequirements.original}</p>}
        {!health.strategies.dsaa && <p>{strategyRequirements.dsaa}</p>}
      </> : <p>{strategyRequirements.original}</p>}
    </IntegrationNotice>}
    {busy && <p role="status">Generating with the selected settings…</p>}
    {error && <p className="error" role="alert">{error}</p>}
  </form>
}

export function BenchmarkLab({ health, jobs, refresh }: { health: Health; jobs: Job[]; refresh: () => Promise<void> }) {
  const [selected, select] = useState<Strategy[]>(() => strategies.filter((key) => health.strategies[key]))
  const [caseCount, setCases] = useState(3)
  const [context, setContext] = useState(8)
  const [limit, setLimit] = useState(32)
  const [jobId, setJob] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const job = jobs.find(({ id }) => id === jobId) ?? jobs[0]

  async function start(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const created = await api<Job>('benchmarks', 'POST', {
        dataset: 'recall', strategies: selected, case_count: caseCount, context_blocks: context, max_new_tokens: limit,
      })
      setJob(created.id)
      await refresh()
    } catch (error) { setError((error as Error).message) }
    finally { setBusy(false) }
  }

  return <>
    <form className="panel" onSubmit={start}>
      <h2>Configure experiment</h2>
      <p className="muted">Each selected strategy receives the same prompts. The evaluator defines how answers are matched and summarized.</p>
      {!health.benchmark_available && <IntegrationNotice title="Evaluation adapter required">
        <p>The evaluation team needs to connect its case generator, answer scoring, and result summaries. Benchmarking also requires at least one registered cache strategy.</p>
      </IntegrationNotice>}
      {health.benchmark_available && !strategies.some((key) => health.strategies[key]) &&
        <IntegrationNotice title="Model runner required"><p>{strategyRequirements.original}</p></IntegrationNotice>}
      <div className="form-row">
        <label>Dataset<select defaultValue="recall"><option value="recall">Fact recall</option></select></label>
        <label>Context blocks<input required type="number" min="1" max="64" value={context || ''} onChange={(event) => setContext(Number(event.target.value))} /></label>
        <label>Test cases<input required type="number" min="1" max="3" value={caseCount || ''} onChange={(event) => setCases(Number(event.target.value))} /></label>
        <label>Max new tokens<input required type="number" min="1" max="256" value={limit || ''} onChange={(event) => setLimit(Number(event.target.value))} /></label>
      </div>
      <fieldset><legend>Cache strategies</legend><div className="choices">{strategies.map((key) => (
        <label key={key}><input type="checkbox" disabled={!health.strategies[key]} checked={selected.includes(key)}
          onChange={(event) => select(event.target.checked ? [...selected, key] : selected.filter((item) => item !== key))} />
          {labels[key]}{health.strategies[key] ? '' : ' · Not configured'}
        </label>
      ))}</div></fieldset>
      <button disabled={busy || !health.benchmark_available || !selected.length}>{busy ? 'Submitting…' : 'Queue benchmark'}</button>
      {error && <p className="error" role="alert">{error}</p>}
    </form>
    <section className="panel">
      <div className="section-heading"><h2>Benchmark results</h2>
        {jobs.length > 0 && <label>Experiment<select value={job?.id ?? ''} onChange={(event) => setJob(event.target.value)}>
          {jobs.map(({ id, created_at, status }) => <option key={id} value={id}>{new Date(created_at).toLocaleString()} · {status}</option>)}
        </select></label>}
      </div>
      {job && <p role="status">{job.status} · {job.completed} / {job.total} strategy runs</p>}
      {job?.error && <p className="error" role="alert">{job.error}</p>}
      <MetricTable columns={strategies.map((key) => ({
        label: labels[key], metrics: job?.results?.[key],
        status: job?.results?.[key] ? undefined : health.strategies[key] ? 'Awaiting results' : key === 'original' ? 'Not configured' : 'Not implemented',
      }))} />
      {job?.status === 'completed' && <a className="action" href={`/api/benchmarks/${job.id}/report`}>Download JSON report</a>}
    </section>
  </>
}

const settingsFields = [
  ['attention', 'Attention weight', 0, 1, 0.01],
  ['semantic', 'Semantic weight', 0, 1, 0.01],
  ['recency', 'Recency weight', 0, 1, 0.01],
  ['retention_budget', 'Retention budget (tokens)', 1, 32768, 1],
  ['eviction_threshold', 'Eviction threshold', 0, 1, 0.01],
] as const

export function Tuning({ available }: { available: boolean }) {
  const [settings, setSettings] = useState<Settings>()
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api<Settings>('config').then(setSettings).catch((error) => setError(error.message))
  }, [])

  async function save(event: FormEvent) {
    event.preventDefault()
    setError('')
    setSaved(false)
    setSaving(true)
    try { setSettings(await api<Settings>('config', 'PUT', settings)); setSaved(true) }
    catch (error) { setError((error as Error).message) }
    finally { setSaving(false) }
  }

  return <form className="panel" onSubmit={save}>
    <h2>Retention policy</h2>
    <p className="muted">The three weights must sum to 1. The DSAA runner controls eviction aggressiveness using memory pressure and the configured budget.</p>
    {!available && <IntegrationNotice title="DSAA integration required">
      <p>Settings can be saved now. The DSAA runner must read the weights, retention budget, and threshold when it runs.</p>
    </IntegrationNotice>}
    {settings && <>
      <div className="form-row tuning-fields">{settingsFields.map(([key, label, min, max, step]) => (
        <label key={key}>{label}<input required type="number" min={min} max={max} step={step}
          value={Number.isNaN(settings[key]) ? '' : settings[key]} onChange={(event) => {
            setSettings({ ...settings, [key]: event.target.valueAsNumber })
            setSaved(false)
          }} /></label>
      ))}</div>
      <button disabled={saving}>Save settings</button>
    </>}
    {saved && <p role="status">Settings saved for new experiments.</p>}
    {error && <p className="error" role="alert">{error}</p>}
  </form>
}
