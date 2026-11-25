import { useEffect, useState } from 'react'

export interface GenerationSettingsValue {
  maxNewTokens: number
  temperature: number
  topK: number | null
  topP: number | null
  doSample: boolean
}

export const initialGenerationSettings: GenerationSettingsValue = {
  maxNewTokens: 128,
  temperature: 0.7,
  topK: null,
  topP: null,
  doSample: false,
}

interface GenerationSettingsProps {
  value: GenerationSettingsValue
  onChange: (value: GenerationSettingsValue) => void
  disabled?: boolean
}

export function GenerationSettings({ value, onChange, disabled = false }: GenerationSettingsProps) {
  const [expanded, setExpanded] = useState(true)
  const [topKInput, setTopKInput] = useState(value.topK?.toString() ?? '')
  const [topPInput, setTopPInput] = useState(value.topP?.toString() ?? '')

  useEffect(() => setTopKInput(value.topK?.toString() ?? ''), [value.topK])
  useEffect(() => setTopPInput(value.topP?.toString() ?? ''), [value.topP])

  const update = <K extends keyof GenerationSettingsValue>(key: K, nextValue: GenerationSettingsValue[K]) => {
    onChange({ ...value, [key]: nextValue })
  }

  return (
    <details className="settings-panel panel" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary className="settings-heading">
        <span className="settings-icon" aria-hidden="true">
          <svg viewBox="0 0 20 20"><path d="M3 5h14M5.5 10h9M8 15h4" /><circle cx="7" cy="5" r="1.5" /><circle cx="12" cy="10" r="1.5" /><circle cx="9.5" cy="15" r="1.5" /></svg>
        </span>
        <span className="settings-heading-copy">
          <span className="eyebrow">Controls</span>
          <span className="settings-title">Generation settings</span>
        </span>
        <svg className="settings-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
      </summary>

      <div className="settings-content">
        <div className="setting-field">
          <div className="setting-label-row">
            <label htmlFor="max-new-tokens">Maximum new tokens</label>
            <span className="setting-number">{value.maxNewTokens}</span>
          </div>
          <input
            id="max-new-tokens"
            type="range"
            min="1"
            max="4096"
            step="1"
            value={value.maxNewTokens}
            disabled={disabled}
            onChange={(event) => update('maxNewTokens', Number(event.target.value))}
          />
          <div className="range-ends"><span>1</span><span>4096</span></div>
        </div>

        <div className="setting-field">
          <div className="setting-label-row">
            <label htmlFor="temperature">Temperature</label>
            <span className="setting-number">{value.temperature.toFixed(1)}</span>
          </div>
          <input
            id="temperature"
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={value.temperature}
            disabled={disabled}
            onChange={(event) => update('temperature', Number(event.target.value))}
          />
          <div className="range-ends"><span>Focused</span><span>Creative</span></div>
        </div>

        <div className="setting-field">
          <div className="setting-label-row">
            <label htmlFor="top-k">Top-K</label>
            <span className="setting-optional">Optional</span>
          </div>
          <input
            className="number-input"
            id="top-k"
            type="number"
            min="1"
            step="1"
            inputMode="numeric"
            placeholder="Not set"
            value={topKInput}
            disabled={disabled}
            onChange={(event) => {
              const raw = event.target.value
              setTopKInput(raw)
              const next = raw === '' ? null : Number(raw)
              if (next === null || (Number.isInteger(next) && next >= 1)) update('topK', next)
            }}
          />
          <p className="setting-hint">Limit selection to the K most likely tokens.</p>
        </div>

        <div className="setting-field">
          <div className="setting-label-row">
            <label htmlFor="top-p">Top-P</label>
            <span className="setting-optional">Optional</span>
          </div>
          <input
            className="number-input"
            id="top-p"
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            inputMode="decimal"
            placeholder="Not set"
            value={topPInput}
            disabled={disabled}
            onChange={(event) => {
              const raw = event.target.value
              setTopPInput(raw)
              const next = raw === '' ? null : Number(raw)
              if (next === null || (next > 0 && next <= 1)) update('topP', next)
            }}
          />
          <p className="setting-hint">Use the smallest likely token set reaching P.</p>
        </div>

        <div className="sampling-setting">
          <span>
            <span className="sampling-label">Sampling</span>
            <span className="setting-hint">Stochastic token selection</span>
          </span>
          <label className="switch" htmlFor="sampling-enabled">
            <span className="sr-only">Enable sampling</span>
            <input
              id="sampling-enabled"
              type="checkbox"
              checked={value.doSample}
              disabled={disabled}
              onChange={(event) => update('doSample', event.target.checked)}
            />
            <span className="switch-track" aria-hidden="true"><span /></span>
          </label>
        </div>

        <p className="settings-footnote">Initial values follow the existing FastAPI generation schema.</p>
      </div>
    </details>
  )
}
