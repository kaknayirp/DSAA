import { useEffect, useRef, useState } from 'react'
import { cancelRequest, generateStream } from '../api/client'
import type { StreamCompletionEvent } from '../api/types'
import { ChatInput } from './ChatInput'
import { ChatMessage, type ConversationMessage } from './ChatMessage'
import { EmptyState } from './EmptyState'
import { GenerationSettings, initialGenerationSettings, type GenerationSettingsValue } from './GenerationSettings'

function messageText(error: unknown): string {
  return error instanceof Error ? error.message : 'An unexpected error occurred while generating a response.'
}

export function ChatWorkspace() {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [prompt, setPrompt] = useState('')
  const [settings, setSettings] = useState<GenerationSettingsValue>(initialGenerationSettings)
  const [generationError, setGenerationError] = useState('')
  const [generationNotice, setGenerationNotice] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const messageAreaRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const requestIdRef = useRef<string | null>(null)
  const assistantIdRef = useRef<string | null>(null)
  const acceptedRef = useRef(false)
  const stopIntentRef = useRef(false)
  const generationLockRef = useRef(false)

  useEffect(() => {
    const messageArea = messageAreaRef.current
    if (messageArea) messageArea.scrollTop = messageArea.scrollHeight
  }, [messages])

  const updateAssistant = (assistantId: string, update: (message: ConversationMessage) => ConversationMessage) => {
    setMessages((current) => current.map((message) => (
      message.id === assistantId ? update(message) : message
    )))
  }

  const submitPrompt = async () => {
    const content = prompt.trim()
    if (!content || generationLockRef.current) return

    generationLockRef.current = true
    setIsGenerating(true)
    setIsStopping(false)
    setGenerationError('')
    setGenerationNotice('')
    requestIdRef.current = null
    acceptedRef.current = false
    stopIntentRef.current = false

    const controller = new AbortController()
    controllerRef.current = controller
    const userMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      createdAt: new Date(),
      status: 'complete',
    }
    const assistantMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      createdAt: new Date(),
      status: 'streaming',
    }
    assistantIdRef.current = assistantMessage.id

    try {
      const outcome = await generateStream({
        prompt: content,
        max_new_tokens: settings.maxNewTokens,
        temperature: settings.temperature,
        top_k: settings.topK,
        top_p: settings.topP,
        do_sample: settings.doSample,
        stream: true,
      }, {
        signal: controller.signal,
        onStarted: () => {
          acceptedRef.current = true
          setPrompt('')
          setMessages((current) => [...current, userMessage, assistantMessage])
        },
        onRequestId: (requestId) => {
          requestIdRef.current = requestId
        },
        onToken: (event) => {
          if (stopIntentRef.current) return
          updateAssistant(assistantMessage.id, (message) => ({
            ...message,
            content: message.content + event.text,
            status: 'streaming',
          }))
        },
        onComplete: (completion: StreamCompletionEvent) => {
          updateAssistant(assistantMessage.id, (message) => ({
            ...message,
            status: 'complete',
            metadata: completion,
          }))
        },
        onCancelled: () => {
          updateAssistant(assistantMessage.id, (message) => ({
            ...message,
            status: 'cancelled',
          }))
        },
      })

      if (outcome.kind === 'cancelled') {
        setPrompt(content)
      }
    } catch (error) {
      if (controller.signal.aborted) {
        if (acceptedRef.current) {
          setPrompt(content)
          updateAssistant(assistantMessage.id, (message) => (
            message.status === 'streaming' ? { ...message, status: 'cancelled' } : message
          ))
        }
      } else {
        const detail = messageText(error)
        if (acceptedRef.current) {
          setPrompt(content)
          updateAssistant(assistantMessage.id, (message) => ({
            ...message,
            status: 'error',
            error: detail,
          }))
        } else {
          setGenerationError(detail)
        }
      }
    } finally {
      controllerRef.current = null
      requestIdRef.current = null
      assistantIdRef.current = null
      generationLockRef.current = false
      stopIntentRef.current = false
      setIsGenerating(false)
      setIsStopping(false)
    }
  }

  const stopGeneration = async () => {
    const controller = controllerRef.current
    if (!controller || isStopping) return

    const requestId = requestIdRef.current
    const assistantId = assistantIdRef.current
    stopIntentRef.current = true
    setIsStopping(true)
    setGenerationError('')

    if (!requestId) {
      controller.abort()
      setGenerationNotice(acceptedRef.current
        ? 'The stream was closed before a request ID arrived. The backend disconnect handler will attempt cancellation.'
        : 'Stopped while waiting for the backend. Cancellation could not be confirmed because no request ID was returned.')
      return
    }

    const cancellationController = new AbortController()
    const cancellationTimeout = window.setTimeout(() => cancellationController.abort(), 5_000)
    try {
      const result = await cancelRequest(requestId, cancellationController.signal)
      if (result.cancelled) {
        controller.abort()
        if (assistantId) {
          updateAssistant(assistantId, (message) => ({ ...message, status: 'cancelled' }))
        }
        setGenerationNotice('The backend confirmed the request was cancelled.')
      } else {
        stopIntentRef.current = false
        setGenerationNotice('The backend reports this request is already complete or cancelled.')
      }
    } catch (error) {
      controller.abort()
      if (assistantId) {
        updateAssistant(assistantId, (message) => (
          message.status === 'streaming'
            ? { ...message, status: 'cancel-unconfirmed', error: messageText(error) }
            : message
        ))
      }
      setGenerationNotice('The stream was closed, but backend cancellation could not be confirmed.')
    } finally {
      window.clearTimeout(cancellationTimeout)
    }
  }

  const clearConversation = () => {
    if (isGenerating) return
    setMessages([])
    setPrompt('')
    setGenerationError('')
    setGenerationNotice('Conversation cleared.')
  }

  return (
    <section className="chat-workspace" aria-labelledby="chat-title">
      <div className="conversation-panel panel">
        <div className="conversation-header">
          <div className="conversation-title-group">
            <span className="conversation-glyph" aria-hidden="true">✳</span>
            <div>
              <p className="eyebrow">Workspace <span className="eyebrow-line" /></p>
              <h2 id="chat-title">New conversation</h2>
            </div>
          </div>
          <div className="conversation-actions">
            {isGenerating && <span className="generating-badge"><span />{isStopping ? 'Stopping' : 'Generating'}</span>}
            {!isGenerating && <span className="draft-badge"><span aria-hidden="true" /> Chat session</span>}
            <button
              className="quiet-button clear-button"
              type="button"
              onClick={clearConversation}
              disabled={isGenerating || (messages.length === 0 && prompt.length === 0)}
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              <svg viewBox="0 0 20 20" aria-hidden="true" className="icon-svg">
                <path d="M3.5 5.5h13m-11.5 0 .8 11h7.4l.8-11M7.5 5.5V3.8h5v1.7M8 8.5v5m4-5v5" />
              </svg>
              <span>Clear</span>
            </button>
          </div>
        </div>

        <div className="conversation-body" ref={messageAreaRef} aria-live="polite" aria-relevant="additions text">
          {messages.length === 0 ? (
            <EmptyState onChoosePrompt={setPrompt} />
          ) : (
            <div className="message-list">
              <div className="conversation-context-note">
                The API accepts one prompt per request. Earlier turns stay visible but are not included in later requests.
              </div>
              {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
            </div>
          )}
        </div>

        <ChatInput
          value={prompt}
          onChange={setPrompt}
          onSubmit={submitPrompt}
          onStop={stopGeneration}
          isGenerating={isGenerating}
          isStopping={isStopping}
          error={generationError}
        />
        <p className="interaction-notice" role={generationNotice ? 'status' : undefined} aria-live="polite">{generationNotice}</p>
      </div>

      <GenerationSettings value={settings} onChange={setSettings} disabled={isGenerating} />
    </section>
  )
}
