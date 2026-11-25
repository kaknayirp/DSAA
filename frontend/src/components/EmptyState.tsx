interface EmptyStateProps {
  onChoosePrompt: (prompt: string) => void
}

const examplePrompts = [
  {
    label: 'Explain KV cache reuse',
    prompt: 'Explain how KV cache reuse speeds up autoregressive generation.',
    icon: '↗',
  },
  {
    label: 'Compare decoding strategies',
    prompt: 'Compare greedy decoding, top-k sampling, and top-p sampling.',
    icon: '⌘',
  },
  {
    label: 'Inspect an inference trace',
    prompt: 'What should I look for when inspecting an LLM inference trace?',
    icon: '⌁',
  },
]

export function EmptyState({ onChoosePrompt }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <div className="welcome-mark" aria-hidden="true">
        <span className="welcome-spark">✳</span>
      </div>
      <p className="eyebrow">Ready when you are</p>
      <h3>What would you like to explore?</h3>
      <p className="empty-description">
        Send a prompt to the active model. Earlier turns stay visible, but each request contains one prompt.
      </p>
      <div className="example-prompts" aria-label="Example prompts">
        {examplePrompts.map((example) => (
          <button
            className="example-prompt"
            type="button"
            key={example.label}
            onClick={() => onChoosePrompt(example.prompt)}
          >
            <span className="example-icon" aria-hidden="true">{example.icon}</span>
            <span>{example.label}</span>
            <svg viewBox="0 0 16 16" aria-hidden="true" className="example-arrow">
              <path d="M3 8h9m-4-4 4 4-4 4" />
            </svg>
          </button>
        ))}
      </div>
    </div>
  )
}
