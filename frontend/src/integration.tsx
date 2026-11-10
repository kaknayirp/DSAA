import { ReactNode } from 'react'
import { Strategy } from './api'

export const strategyRequirements: Record<Strategy, string> = {
  original: 'Configure the backend with a local Qwen2 or Qwen2.5 checkpoint, or register the Original model runner.',
  h2o: 'Implement and register the H2O cache strategy to collect measured H2O results.',
  snapkv: 'Implement and register the SnapKV cache strategy to collect measured SnapKV results.',
  dsaa: 'Connect the DSAA scoring algorithm and KV-cache eviction logic, then register the DSAA runner.',
}

export function IntegrationNotice({ title, children }: { title: string; children: ReactNode }) {
  return <aside className="notice integration-notice">
    <h3>{title}</h3>
    <div>{children}</div>
  </aside>
}
