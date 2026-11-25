import { BackendStatus } from './components/BackendStatus'
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
            <h1 id="page-title">Your inference workspace.</h1>
            <p className="hero-description">
              A clear view into your local model runtime. Prompt controls and generation tools will live here.
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

        <section className="panel prompt-placeholder" aria-labelledby="prompt-title">
          <div className="placeholder-topline">
            <div>
              <p className="eyebrow">Workspace</p>
              <h2 id="prompt-title">Prompt interface</h2>
            </div>
            <span className="coming-soon">Coming next</span>
          </div>
          <div className="placeholder-body" aria-disabled="true">
            <div className="placeholder-copy">
              <span className="placeholder-icon" aria-hidden="true">✳</span>
              <div>
                <p className="placeholder-title">Start with a prompt</p>
                <p className="placeholder-description">
                  Generation controls and model responses will appear in this workspace.
                </p>
              </div>
            </div>
            <button className="prepare-button" type="button" disabled>
              <span aria-hidden="true">＋</span> New prompt
            </button>
          </div>
          <div className="placeholder-meta">
            <span><i className="meta-dot" aria-hidden="true" /> Connected to the DSAA API</span>
            <span>Generation is not enabled in this foundation phase</span>
          </div>
        </section>

        <footer className="page-footer">
          <span>DSAA <span className="footer-separator">/</span> Dynamic Smart Attention-based Allocation</span>
          <span>Infrastructure console</span>
        </footer>
      </main>
    </div>
  )
}
