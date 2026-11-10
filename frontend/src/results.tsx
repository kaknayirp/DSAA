import { useState } from 'react'
import { Comparison, Metrics, Token } from './api'
import { IntegrationNotice } from './integration'

export const number = new Intl.NumberFormat('en', { maximumFractionDigits: 2 })
export const percent = (value: number) => `${number.format(value * 100)}%`
export const bytes = (value: number) => {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${number.format(value)} ${units[unit]}`
}

const metricRows = [
  ['KV cache memory', 'kv_cache_bytes', bytes],
  ['Cached tokens', 'cached_tokens', (value: number) => number.format(value)],
  ['Peak process RAM', 'peak_ram_bytes', bytes],
  ['Time to first token', 'first_token_ms', (value: number) => `${number.format(value)} ms`],
  ['Generation speed', 'tokens_per_second', (value: number) => `${number.format(value)} tok/s`],
  ['Answer accuracy', 'answer_accuracy', percent],
  ['Retention ratio', 'retention_ratio', percent],
] as const

export function MetricTable({ columns }: { columns: { label: string; metrics?: Partial<Metrics>; status?: string }[] }) {
  return <div className="table-scroll"><table>
    <thead><tr><th scope="col">Metric</th>{columns.map(({ label }) => <th key={label} scope="col">{label}</th>)}</tr></thead>
    <tbody>{metricRows.map(([label, key, format]) => (
      <tr key={key}><th scope="row">{label}</th>
        {columns.map((column) => {
          const value = column.metrics?.[key]
          return <td key={column.label}>{value == null ? column.status ?? '—' : format(value)}</td>
        })}
      </tr>
    ))}</tbody>
  </table></div>
}

export function CompareResults({ result }: { result: Comparison }) {
  const { baseline, dsaa, source } = result
  const max = Math.max(baseline.kv_cache_bytes, dsaa.kv_cache_bytes, 1)
  const remaining = baseline.kv_cache_bytes ? dsaa.kv_cache_bytes / baseline.kv_cache_bytes : null
  return <>
    <section className="panel">
      <div className="section-heading"><h2>Cache memory</h2><span className="badge">{source === 'demo' ? 'Illustrative sample' : 'Measured run'}</span></div>
      <div className="bars">
        {([['Original', baseline], ['DSAA', dsaa]] as const).map(([label, row]) => (
          <div className="bar-row" key={label}>
            <span>{label}</span>
            <div className="track" aria-hidden="true"><div className={label === 'DSAA' ? 'fill optimized' : 'fill'}
              style={{ width: percent(row.kv_cache_bytes / max) }} /></div>
            <strong>{bytes(row.kv_cache_bytes)}</strong>
          </div>
        ))}
      </div>
      {remaining != null && <p className="muted">{percent(remaining)} of Original KV memory remains with DSAA.</p>}
      <MetricTable columns={[{ label: 'Original', metrics: baseline }, { label: 'DSAA', metrics: dsaa }]} />
    </section>
    {(baseline.answer != null || dsaa.answer != null) && <section className="panel">
      <h2>Model answers</h2>
      <div className="answers">
        <article><h3>Original</h3><p>{baseline.answer ?? 'Unavailable'}</p></article>
        <article><h3>DSAA</h3><p>{dsaa.answer ?? 'Unavailable'}</p></article>
      </div>
    </section>}
  </>
}

export function TokenExplorer({ entries, sample }: { entries: Token[]; sample: boolean }) {
  const [filter, setFilter] = useState('all')
  const [selection, select] = useState<Token>()
  const retained = entries.filter((entry) => entry.retained).length
  const visible = entries.filter((entry) => filter === 'all' || entry.retained === (filter === 'retained'))
  const active = selection && visible.includes(selection) ? selection : visible[0]
  const details = active ? [
    ['Token ID', active.token_id], ['Position', active.position], ['Layer', active.layer], ['Head', active.head],
    ['Attention', active.attention], ['Semantic', active.semantic], ['Recency', active.recency],
    ['Retention score', active.score], ['Cache bytes saved', active.saved_bytes == null ? null : bytes(active.saved_bytes)],
  ] : []

  return <section className="panel">
    <div className="section-heading"><h2>Token decisions</h2><span className="badge">{sample ? 'Illustrative sample' : 'Measured run'}</span></div>
    {!sample && entries.length === 0 && <IntegrationNotice title="Token telemetry required">
      <p>The runner returned no token entries for this run. It needs to provide token positions and retention decisions for inspection.</p>
    </IntegrationNotice>}
    <p className="muted">{retained} retained · {entries.length - retained} evicted</p>
    <label htmlFor="token-filter">Show</label>
    <select id="token-filter" value={filter} onChange={(event) => setFilter(event.target.value)}>
      <option value="all">All tokens</option><option value="retained">Retained</option><option value="evicted">Evicted</option>
    </select>
    <div className="explorer">
      <div className="tokens" aria-label="Token positions">
        {visible.map((entry, index) => <button key={index} type="button" aria-pressed={entry === active}
          onClick={() => select(entry)} className={entry.retained ? 'retained' : 'evicted'}>
          <code>{entry.token}</code><small>#{entry.position} · {entry.retained ? 'Retained' : 'Evicted'}</small>
        </button>)}
        {!visible.length && <p className="muted">No matching tokens were reported.</p>}
      </div>
      {active && <aside className="token-detail">
        <h3>{active.retained ? 'Retained' : 'Evicted'} · <code>{active.token}</code></h3>
        <dl>{details.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value ?? '—'}</dd></div>)}</dl>
        <p className="muted">{active.reason ?? 'No reason was reported for this token.'}</p>
      </aside>}
    </div>
  </section>
}
