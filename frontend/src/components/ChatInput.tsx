import type { FormEvent, KeyboardEvent } from 'react'

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void | Promise<void>
  onStop: () => void | Promise<void>
  isGenerating: boolean
  isStopping: boolean
  error: string
}

export function ChatInput({ value, onChange, onSubmit, onStop, isGenerating, isStopping, error }: ChatInputProps) {
  const canSubmit = value.trim().length > 0 && !isGenerating

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (canSubmit) void onSubmit()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      if (canSubmit) void onSubmit()
    }
  }

  return (
    <form className="chat-composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="chat-prompt">Write a prompt</label>
      <textarea
        id="chat-prompt"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Write a prompt…"
        rows={2}
        disabled={isGenerating}
        aria-describedby="composer-help composer-local-note"
      />
      {error && <p className="generation-error" role="alert">{error}</p>}
      <div className="composer-toolbar">
        <span className="composer-help" id="composer-help">
          <span className="composer-key">Enter</span> to send · <span className="composer-key">Shift + Enter</span> for a new line
        </span>
        {isGenerating ? (
          <button className="stop-button" type="button" onClick={() => void onStop()} disabled={isStopping}>
            <span className="stop-square" aria-hidden="true" />
            {isStopping ? 'Stopping…' : 'Stop generation'}
          </button>
        ) : (
          <button className="send-button" type="submit" disabled={!canSubmit} aria-label="Send prompt">
            <span>Send</span>
            <svg viewBox="0 0 18 18" aria-hidden="true" className="send-icon">
              <path d="M3 9h11m-4-4 4 4-4 4" />
            </svg>
          </button>
        )}
      </div>
      <p className="local-composer-note" id="composer-local-note">Requests are streamed from the configured FastAPI endpoint.</p>
    </form>
  )
}
