import type { StreamCompletionEvent } from '../api/types'

export interface ConversationMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: Date
  status?: 'streaming' | 'complete' | 'cancelled' | 'cancel-unconfirmed' | 'error'
  metadata?: StreamCompletionEvent
  error?: string
}

interface ChatMessageProps {
  message: ConversationMessage
}

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
})

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <article className={`chat-message chat-message--${message.role}`} aria-label={`${isUser ? 'You' : 'Assistant'}, ${timeFormatter.format(message.createdAt)}`}>
      <div className="message-avatar" aria-hidden="true">{isUser ? 'Y' : 'D'}</div>
      <div className="message-content-wrap">
        <div className="message-meta">
          <span className="message-author">{isUser ? 'You' : 'DSAA assistant'}</span>
          <time dateTime={message.createdAt.toISOString()}>{timeFormatter.format(message.createdAt)}</time>
          {isUser && <span className="message-delivered-tag">sent to API</span>}
        </div>
        {message.content && <div className="message-bubble">{message.content}</div>}
        {!isUser && message.status === 'streaming' && (
          <div className="assistant-progress" role="status" aria-live="polite">
            {message.content ? <span className="streaming-label">Generating</span> : <><span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span><span>Waiting for Qwen tokens…</span></>}
          </div>
        )}
        {!isUser && message.status === 'cancelled' && <p className="message-state message-state--cancelled">Generation cancelled.</p>}
        {!isUser && message.status === 'cancel-unconfirmed' && (
          <div className="message-state message-state--warning" role="status">
            Stream stopped. Backend cancellation was not confirmed{message.error ? `: ${message.error}` : '.'}
          </div>
        )}
        {!isUser && message.status === 'error' && (
          <div className="message-state message-state--error" role="alert">Generation failed: {message.error}</div>
        )}
        {!isUser && message.metadata && <GenerationMetadata metadata={message.metadata} />}
      </div>
    </article>
  )
}

function GenerationMetadata({ metadata }: { metadata: StreamCompletionEvent }) {
  return (
    <div className="message-metadata" aria-label="Generation metadata">
      <span>{metadata.finish_reason.replaceAll('_', ' ')}</span>
      <span>{metadata.num_prompt_tokens} prompt tokens</span>
      <span>{metadata.num_generated_tokens} generated</span>
      <span>{metadata.total_time_seconds.toFixed(2)} s</span>
      <span>{metadata.tokens_per_second.toFixed(1)} tok/s</span>
    </div>
  )
}
