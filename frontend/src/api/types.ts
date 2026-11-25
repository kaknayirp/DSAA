export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_id: string
  device: string
  active_requests: number
  queued_requests: number
  uptime_seconds: number
}

export interface ModelInfo {
  model_id: string
  device: string
  precision: string
  context_length: number
  is_active: boolean
}

export interface ModelListResponse {
  models: ModelInfo[]
  active_model: string
}
