import { BackendStatus } from './components/BackendStatus'
import { ChatWorkspace } from './components/ChatWorkspace'
import { Header } from './components/Header'

function ConnectionBadge() {
  return (
    <span className="header-label">
      <span className="header-indicator" aria-hidden="true" />
      Local workspace
    </span>
  )
}

export default function App() {
  return (
    <div className="app-shell">
      <Header status={<ConnectionBadge />} />
      <main id="main" className="main-content">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">Model workspace <span className="eyebrow-line" /></p>
            <h1 id="page-title">A workspace for local models.</h1>
            <p className="hero-description">
              Draft a conversation and configure generation settings while keeping an eye on your inference runtime.
            </p>
          </div>
          <div className="hero-orbit" aria-hidden="true">
            <span className="orbit orbit-one" />
            <span className="orbit orbit-two" />
            <span className="orbit-core">D</span>
            <span className="orbit-node orbit-node-one" />
            <span className="orbit-node orbit-node-two" />
          </div>
        </section>

        <BackendStatus />
        <ChatWorkspace />

        <footer className="page-footer">
          <span>DSAA <span className="footer-separator">/</span> Dynamic Smart Attention-based Allocation</span>
          <span>Infrastructure console</span>
        </footer>
      </main>
    </div>
  )
}
