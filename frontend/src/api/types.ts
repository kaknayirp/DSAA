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

export interface GenerateStreamRequest {
  prompt: string
  max_new_tokens: number
  temperature: number
  top_k: number | null
  top_p: number | null
  do_sample: boolean
  stream: true
}

export interface StreamTokenEvent {
  request_id: string
  token_id: number
  text: string
}

export interface StreamCompletionEvent {
  request_id: string
  finish_reason: string
  num_prompt_tokens: number
  num_generated_tokens: number
  total_time_seconds: number
  tokens_per_second: number
}

export interface CancelResponse {
  request_id: string
  cancelled: boolean
  message: string
}
