import type { ReactNode } from 'react'

interface HeaderProps {
  status: ReactNode
}

export function Header({ status }: HeaderProps) {
  return (
    <header className="topbar">
      <a className="brand" href="#main" aria-label="DSAA Studio home">
        <span className="brand-mark" aria-hidden="true">D</span>
        <span className="brand-copy">
          <span className="brand-name">DSAA Studio</span>
          <span className="brand-caption">LLM infrastructure</span>
        </span>
      </a>
      <div className="topbar-status">{status}</div>
    </header>
  )
}
