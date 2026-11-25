import { useEffect, useState } from 'react'
import { getHealth, getModels } from '../api/client'
import type { HealthResponse, ModelListResponse } from '../api/types'

const REFRESH_INTERVAL_MS = 15_000

type ConnectionState = 'checking' | 'online' | 'offline'

export function BackendStatus() {
  const [connection, setConnection] = useState<ConnectionState>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [models, setModels] = useState<ModelListResponse | null>(null)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)
  const [modelsError, setModelsError] = useState<string | null>(null)

  useEffect(() => {
    let disposed = false
    let activeController: AbortController | undefined

    const refresh = async () => {
      activeController?.abort()
      const controller = new AbortController()
      activeController = controller

      const [healthResult, modelsResult] = await Promise.allSettled([
        getHealth(controller.signal),
        getModels(controller.signal),
      ])

      if (disposed || controller.signal.aborted) return

      setLastChecked(new Date())
      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value)
        setConnection('online')
      } else {
        setConnection('offline')
        setHealth(null)
      }

      if (modelsResult.status === 'fulfilled') {
        setModels(modelsResult.value)
        setModelsError(null)
      } else {
        setModels(null)
        setModelsError('Model information is unavailable.')
      }
    }

    void refresh()
    const intervalId = window.setInterval(() => void refresh(), REFRESH_INTERVAL_MS)

    return () => {
      disposed = true
      window.clearInterval(intervalId)
      activeController?.abort()
    }
  }, [])

  const activeModel = models?.active_model || health?.model_id
  const device = health?.device && health.device !== 'unknown' ? health.device : null

  return (
    <section className="panel backend-panel" aria-labelledby="backend-title">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Runtime</p>
          <h2 id="backend-title">Backend status</h2>
        </div>
        <span className={`connection-pill connection-${connection}`} role="status" aria-live="polite">
          <span className="status-dot" aria-hidden="true" />
          {connection === 'checking' ? 'Checking' : connection === 'online' ? 'API reachable' : 'Unavailable'}
        </span>
      </div>

      {connection === 'offline' ? (
        <div className="status-message status-error" role="alert">
          The API could not be reached. Start the FastAPI server and check the configured API URL.
        </div>
      ) : (
        <div className="status-grid">
          <div className="status-item">
            <span className="status-label">Model runtime</span>
            <span className={`status-value ${health?.model_loaded ? 'value-ready' : 'value-muted'}`}>
              {health ? (health.model_loaded ? 'Loaded' : 'Not loaded') : 'Checking…'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Active model</span>
            <span className="status-value status-model" title={activeModel || undefined}>
              {activeModel || 'Waiting for API'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Device</span>
            <span className="status-value">{device || 'Not reported'}</span>
          </div>
          <div className="status-item">
            <span className="status-label">Requests</span>
            <span className="status-value">
              {health ? `${health.active_requests} active · ${health.queued_requests} queued` : '—'}
            </span>
          </div>
        </div>
      )}

      {modelsError && connection === 'online' && (
        <p className="status-note" role="status">{modelsError}</p>
      )}
      {health && !health.model_loaded && (
        <p className="status-note" role="status">The model loads on the first generation request, so the initial response may take longer.</p>
      )}
      <p className="status-footnote">
        {lastChecked ? `Last checked ${lastChecked.toLocaleTimeString()}` : 'Checking every 15 seconds'}
        {health?.status && health.status !== 'healthy' ? ` · API reports ${health.status}` : ''}
      </p>
    </section>
  )
}
