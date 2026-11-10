export const strategies = ['original', 'h2o', 'snapkv', 'dsaa'] as const
export type Strategy = typeof strategies[number]
export const labels: Record<Strategy, string> = { original: 'Original', h2o: 'H2O', snapkv: 'SnapKV', dsaa: 'DSAA' }

export interface Health {
  strategies: Record<Strategy, boolean>
  comparison_available: boolean
  benchmark_available: boolean
  model: string | null
  device: string | null
}

export interface SystemMetrics {
  timestamp: string
  ram_total_bytes: number
  ram_used_bytes: number
  process_ram_bytes: number
  memory_pressure: number
  gpu_memory_bytes: number | null
  busy: boolean
}

export interface Metrics {
  kv_cache_bytes: number
  cached_tokens: number
  peak_ram_bytes: number | null
  first_token_ms: number | null
  tokens_per_second: number | null
  answer_accuracy: number | null
  retention_ratio?: number | null
  answer: string | null
}

export interface Token {
  position: number
  token: string
  retained: boolean
  token_id?: number | null
  layer?: number | null
  head?: number | null
  attention?: number | null
  semantic?: number | null
  recency?: number | null
  score?: number | null
  reason?: string | null
  saved_bytes?: number | null
}

export interface Run {
  id: string
  strategy: Strategy
  metrics: Metrics
  entries: Token[]
  context_tokens: number | null
  memory_pressure: number | null
}

export interface Comparison {
  baseline: Metrics
  dsaa: Metrics
  entries: Token[]
  source: 'demo' | 'measured'
  dsaa_run_id: string | null
}

export interface Settings {
  attention: number
  semantic: number
  recency: number
  retention_budget: number
  eviction_threshold: number
}

export interface Job {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  completed: number
  total: number
  results?: Partial<Record<Strategy, Partial<Metrics>>>
  error?: string
}

export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method,
    ...(body === undefined ? {} : {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    const detail = Array.isArray(data?.detail) ? data.detail[0]?.msg : data?.detail
    throw new Error(detail ?? `Request failed (HTTP ${response.status}).`)
  }
  return response.json()
}
