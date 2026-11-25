import type {
  CancelResponse,
  GenerateStreamRequest,
  HealthResponse,
  ModelInfo,
  ModelListResponse,
  StreamCompletionEvent,
  StreamTokenEvent,
} from './types'

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api'
const apiBaseUrl = configuredBaseUrl.replace(/\/+$/, '')

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isHealthResponse(value: unknown): value is HealthResponse {
  return isRecord(value)
    && typeof value.status === 'string'
    && typeof value.model_loaded === 'boolean'
    && typeof value.model_id === 'string'
    && typeof value.device === 'string'
    && typeof value.active_requests === 'number'
    && typeof value.queued_requests === 'number'
    && typeof value.uptime_seconds === 'number'
}

function isModelInfo(value: unknown): value is ModelInfo {
  return isRecord(value)
    && typeof value.model_id === 'string'
    && typeof value.device === 'string'
    && typeof value.precision === 'string'
    && typeof value.context_length === 'number'
    && typeof value.is_active === 'boolean'
}

function isModelListResponse(value: unknown): value is ModelListResponse {
  return isRecord(value)
    && typeof value.active_model === 'string'
    && Array.isArray(value.models)
    && value.models.every(isModelInfo)
}

function isTokenEvent(value: unknown): value is StreamTokenEvent {
  return isRecord(value)
    && typeof value.request_id === 'string'
    && typeof value.token_id === 'number'
    && Number.isFinite(value.token_id)
    && typeof value.text === 'string'
}

function isCompletionEvent(value: unknown): value is StreamCompletionEvent {
  return isRecord(value)
    && typeof value.request_id === 'string'
    && typeof value.finish_reason === 'string'
    && typeof value.num_prompt_tokens === 'number'
    && typeof value.num_generated_tokens === 'number'
    && typeof value.total_time_seconds === 'number'
    && typeof value.tokens_per_second === 'number'
}

function formatErrorDetail(value: unknown): string | undefined {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    const messages = value
      .map((item) => isRecord(item) && typeof item.msg === 'string' ? item.msg : undefined)
      .filter((item): item is string => Boolean(item))
    return messages.length > 0 ? messages.join('; ') : undefined
  }
  return undefined
}

async function httpError(response: Response): Promise<ApiError> {
  let message = `API request failed (${response.status})`
  try {
    const body: unknown = await response.json()
    if (isRecord(body)) {
      const detail = formatErrorDetail(body.detail)
      const errorName = typeof body.error === 'string' ? body.error : undefined
      if (detail) message = errorName ? `${errorName}: ${detail}` : detail
      else if (errorName) message = errorName
    }
  } catch {
    // Keep a useful status-based message for non-JSON error bodies.
  }
  return new ApiError(message, response.status)
}

async function getJson<T>(
  path: string,
  isExpectedShape: (value: unknown) => value is T,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new Error('Could not reach the DSAA API. Check that the backend is running.')
  }

  if (!response.ok) throw await httpError(response)

  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new Error(`The DSAA API returned invalid JSON for ${path}.`)
  }
  if (!isExpectedShape(body)) throw new Error(`The DSAA API returned an unexpected response for ${path}.`)
  return body
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>('/health', isHealthResponse, signal)
}

export function getModels(signal?: AbortSignal): Promise<ModelListResponse> {
  return getJson<ModelListResponse>('/models', isModelListResponse, signal)
}

export interface StreamCallbacks {
  signal: AbortSignal
  onStarted: () => void
  onRequestId: (requestId: string) => void
  onToken: (event: StreamTokenEvent) => void
  onComplete: (event: StreamCompletionEvent) => void
  onCancelled: () => void
}

export type StreamOutcome =
  | { kind: 'complete'; completion: StreamCompletionEvent }
  | { kind: 'cancelled' }

function sseData(frame: string): string | null {
  const dataLines: string[] = []
  for (const line of frame.split(/\r\n|\r|\n/)) {
    if (line.startsWith(':')) continue
    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    if (field !== 'data') continue
    let value = separator === -1 ? '' : line.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    dataLines.push(value)
  }
  return dataLines.length > 0 ? dataLines.join('\n') : null
}

export async function generateStream(
  payload: GenerateStreamRequest,
  callbacks: StreamCallbacks,
): Promise<StreamOutcome> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}/generate`, {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: callbacks.signal,
    })
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new Error('Could not connect to the DSAA generation endpoint.')
  }

  if (!response.ok) throw await httpError(response)
  if (!response.headers.get('content-type')?.toLowerCase().includes('text/event-stream')) {
    throw new Error('The DSAA API did not return a Server-Sent Events stream.')
  }
  if (!response.body) throw new Error('The DSAA API returned an empty generation stream.')

  callbacks.onStarted()
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completion: StreamCompletionEvent | null = null
  let cancelled = false
  let receivedDone = false

  const dispatch = (frame: string) => {
    const data = sseData(frame)
    if (data === null || data.length === 0) return
    if (data === '[DONE]') {
      receivedDone = true
      return
    }

    let event: unknown
    try {
      event = JSON.parse(data)
    } catch {
      // Ignore malformed frames; a missing completion event is reported at EOF.
      return
    }

    if (!isRecord(event)) return
    if (typeof event.error === 'string') {
      throw new ApiError(event.error, typeof event.code === 'number' ? event.code : 500)
    }
    if (event.cancelled === true) {
      cancelled = true
      callbacks.onCancelled()
      return
    }
    if (isCompletionEvent(event)) {
      completion = event
      callbacks.onRequestId(event.request_id)
      callbacks.onComplete(event)
      return
    }
    if (isTokenEvent(event)) {
      callbacks.onRequestId(event.request_id)
      callbacks.onToken(event)
    }
  }

  try {
    while (!receivedDone) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })

      let separator = /\r\n\r\n|\n\n|\r\r/.exec(buffer)
      while (separator) {
        dispatch(buffer.slice(0, separator.index))
        buffer = buffer.slice(separator.index + separator[0].length)
        if (receivedDone) break
        separator = /\r\n\r\n|\n\n|\r\r/.exec(buffer)
      }

      if (done) {
        if (!receivedDone && buffer.trim()) dispatch(buffer)
        break
      }
    }
  } catch (error) {
    try {
      await reader.cancel(error)
    } catch {
      // The request may already have been aborted or the stream may have closed.
    }
    throw error
  } finally {
    reader.releaseLock()
  }

  if (!receivedDone) throw new Error('The generation stream ended before the [DONE] event.')
  if (cancelled) return { kind: 'cancelled' }
  if (!completion) throw new Error('The generation stream ended without completion metadata.')
  return { kind: 'complete', completion }
}

function isCancelResponse(value: unknown): value is CancelResponse {
  return isRecord(value)
    && typeof value.request_id === 'string'
    && typeof value.cancelled === 'boolean'
    && typeof value.message === 'string'
}

export async function cancelRequest(requestId: string, signal?: AbortSignal): Promise<CancelResponse> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}/requests/${encodeURIComponent(requestId)}/cancel`, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new Error('Could not reach the DSAA request cancellation endpoint.')
  }
  if (!response.ok) throw await httpError(response)

  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new Error('The DSAA API returned invalid cancellation JSON.')
  }
  if (!isCancelResponse(body)) throw new Error('The DSAA API returned an unexpected cancellation response.')
  return body
}
