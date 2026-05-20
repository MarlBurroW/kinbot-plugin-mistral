/**
 * Mistral AI provider for KinBot.
 *
 * Talks to api.mistral.ai with raw fetch + a hand-rolled SSE parser
 * (the Mistral REST API is OpenAI-compatible, so no SDK dependency
 * is needed — the wire shape is small enough that wrapping the
 * official `@mistralai/mistralai` package would just add weight).
 *
 * Endpoints used:
 *   - GET  /v1/models               — model listing
 *   - POST /v1/chat/completions     — chat (streamed when stream:true)
 *
 * Native function calling is supported. Tool calls flow through the
 * standard `tool_calls` array on assistant deltas; this plugin
 * parses them into KinBot's `tool-use` ChatChunks.
 */

import type {
  PluginContext,
  PluginExports,
  LLMProvider,
  LLMModel,
  ChatRequest,
  ChatChunk,
  KinbotMessage,
  KinbotMessageBlock,
  KinbotTool,
  SystemPrompt,
  ProviderConfig,
  AuthResult,
  FinishReason,
  Usage,
  ConfigField,
} from '@kinbot-developer/sdk'

// ─── Config schema ───────────────────────────────────────────────────────────

const CONFIG_SCHEMA: readonly ConfigField[] = [
  {
    key: 'apiKey',
    type: 'secret',
    label: 'API Key',
    required: true,
    placeholder: 'Bearer …',
    description: 'Get one at https://console.mistral.ai/api-keys',
  },
]

const API_BASE = 'https://api.mistral.ai/v1'

// ─── Model classification ───────────────────────────────────────────────────
//
// Mistral's `/v1/models` listing exposes every model the account can
// touch — chat, embedding, moderation, etc. — and not every release
// of the API tags the modality cleanly. We filter to "chat-capable"
// via the `capabilities.completion_chat` flag when present, and fall
// back to a name-pattern exclusion otherwise. No hardcoded model-id
// allowlist; future Mistral models pass through automatically.

const NON_CHAT_NAME_PATTERN = /(embed|moderation|ocr)/i

// Vision-capable model families (Pixtral, recent large/medium with
// vision). Mistral's listing exposes `capabilities.vision` on recent
// API versions; this regex is the fallback when it's missing.
const VISION_PATTERN = /^(pixtral-|mistral-medium-2|mistral-large-2)/i

// ─── Wire types ─────────────────────────────────────────────────────────────

interface MistralModelListing {
  data?: Array<{
    id: string
    capabilities?: {
      completion_chat?: boolean
      completion_fim?: boolean
      function_calling?: boolean
      vision?: boolean
    }
    max_context_length?: number
    /** Canonical model name — multiple entries can share the same
     *  `name` when they're aliases of the same underlying model. */
    name?: string
    /** Other ids that point to the same underlying model. */
    aliases?: string[]
    description?: string
  }>
}

interface MistralMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | Array<{ type: string; text?: string; image_url?: { url: string } }>
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
}

interface MistralTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

interface MistralChatRequest {
  model: string
  messages: MistralMessage[]
  temperature?: number
  max_tokens?: number
  tools?: MistralTool[]
  tool_choice?: 'auto' | 'none' | 'any'
  stream?: boolean
  random_seed?: number
}

interface MistralChatChunk {
  id?: string
  choices?: Array<{
    delta?: {
      role?: string
      content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

// ─── KinBot → Mistral conversions ───────────────────────────────────────────

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!)
  return btoa(binary)
}

function systemToMistral(system: SystemPrompt | undefined): MistralMessage | null {
  if (!system || system.length === 0) return null
  const text = system.map((b) => b.text).filter(Boolean).join('\n\n')
  if (!text) return null
  return { role: 'system', content: text }
}

function blockToMistralParts(
  block: KinbotMessageBlock,
): Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }> {
  switch (block.type) {
    case 'text':
      return block.text ? [{ type: 'text', text: block.text }] : []
    case 'image': {
      const base64 = uint8ToBase64(block.data)
      return [{
        type: 'image_url',
        image_url: { url: `data:${block.mediaType};base64,${base64}` },
      }]
    }
    case 'tool-use':
    case 'tool-result':
    case 'thinking':
      // Handled at the message level, not as parts.
      return []
  }
}

/**
 * Translate KinBot's discriminated-union messages into Mistral's
 * OpenAI-compatible shape. Three subtleties:
 *
 * 1. Mistral's tool-result message uses { role: 'tool', tool_call_id,
 *    content }. KinBot's tool-result block lives on a user turn —
 *    we split user turns whenever tool-result blocks appear so each
 *    tool result becomes its own Mistral message.
 * 2. Multi-modal user turns (text + image) need the content as an
 *    array; pure-text turns use the simpler string form.
 * 3. Thinking blocks have no analog in Mistral and are dropped.
 */
function messagesToMistral(messages: KinbotMessage[]): MistralMessage[] {
  const out: MistralMessage[] = []
  for (const m of messages) {
    if (m.role === 'assistant') {
      const textParts: string[] = []
      const toolCalls: NonNullable<MistralMessage['tool_calls']> = []
      for (const b of m.content) {
        if (b.type === 'text' && b.text) textParts.push(b.text)
        else if (b.type === 'tool-use') {
          toolCalls.push({
            id: b.id,
            type: 'function',
            function: {
              name: b.name,
              arguments: typeof b.args === 'string' ? b.args : JSON.stringify(b.args ?? {}),
            },
          })
        }
      }
      const msg: MistralMessage = { role: 'assistant' }
      const text = textParts.join('\n').trim()
      if (text) msg.content = text
      if (toolCalls.length > 0) msg.tool_calls = toolCalls
      if (msg.content || msg.tool_calls) out.push(msg)
    } else {
      // user role — may contain a mix of text/image and tool-result.
      const userParts: ReturnType<typeof blockToMistralParts> = []
      const flushUser = () => {
        if (userParts.length === 0) return
        out.push({
          role: 'user',
          content: userParts.length === 1 && userParts[0]!.type === 'text'
            ? userParts[0]!.text
            : [...userParts],
        })
        userParts.length = 0
      }
      for (const b of m.content) {
        if (b.type === 'tool-result') {
          flushUser()
          out.push({
            role: 'tool',
            tool_call_id: b.toolUseId,
            content: b.content,
          })
        } else {
          for (const p of blockToMistralParts(b)) userParts.push(p)
        }
      }
      flushUser()
    }
  }
  return out
}

function toolsToMistral(tools: KinbotTool[] | undefined): MistralTool[] | undefined {
  if (!tools || tools.length === 0) return undefined
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
    },
  }))
}

// ─── SSE parser ─────────────────────────────────────────────────────────────

async function* parseSSE(response: Response): AsyncIterable<MistralChatChunk> {
  if (!response.body) throw new Error('Mistral returned an empty body')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const rawMessage = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        const dataLines = rawMessage
          .split('\n')
          .filter((l) => l.startsWith('data:'))
          .map((l) => l.slice(5).trimStart())
        if (dataLines.length === 0) continue
        const payload = dataLines.join('\n')
        if (payload === '[DONE]') return
        try {
          yield JSON.parse(payload) as MistralChatChunk
        } catch {
          // Malformed event — skip rather than abort the whole stream.
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

// ─── Finish-reason mapping ──────────────────────────────────────────────────

function finishReasonFromMistral(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case 'stop': return 'stop'
    case 'length':
    case 'model_length':
      return 'length'
    case 'tool_calls': return 'tool-calls'
    case 'error': return 'error'
    default: return reason ? 'unknown' : 'stop'
  }
}

// ─── Errors ─────────────────────────────────────────────────────────────────

class MistralError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message)
    this.name = 'MistralError'
  }
}

async function errorFromResponse(res: Response): Promise<MistralError> {
  const text = await res.text().catch(() => '')
  let message = text || res.statusText
  try {
    const parsed = JSON.parse(text) as { message?: string; error?: { message?: string } }
    message = parsed.error?.message ?? parsed.message ?? message
  } catch { /* keep raw body */ }
  return new MistralError(`Mistral ${res.status}: ${message}`, res.status)
}

// ─── Stream → ChatChunk ─────────────────────────────────────────────────────

async function* streamMistral(
  apiKey: string,
  body: MistralChatRequest,
  signal: AbortSignal | undefined,
): AsyncIterable<ChatChunk> {
  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })
  if (!res.ok) throw await errorFromResponse(res)

  // Tool calls arrive in fragmented deltas keyed by index — we
  // accumulate (id, name, args-string) per index and emit one
  // `tool-use` chunk per fully-formed call when the stream ends.
  const toolCalls = new Map<number, { id: string; name: string; args: string }>()
  let usage: Usage = {}
  let finishReason: FinishReason = 'unknown'

  for await (const chunk of parseSSE(res)) {
    if (chunk.usage) {
      usage = {
        ...(chunk.usage.prompt_tokens != null ? { inputTokens: chunk.usage.prompt_tokens } : {}),
        ...(chunk.usage.completion_tokens != null ? { outputTokens: chunk.usage.completion_tokens } : {}),
      }
    }
    const choice = chunk.choices?.[0]
    if (!choice) continue
    const delta = choice.delta
    if (delta) {
      if (typeof delta.content === 'string' && delta.content) {
        yield { type: 'text-delta', text: delta.content }
      }
      for (const tc of delta.tool_calls ?? []) {
        const existing = toolCalls.get(tc.index) ?? { id: '', name: '', args: '' }
        if (tc.id) existing.id = tc.id
        if (tc.function?.name) existing.name = tc.function.name
        if (tc.function?.arguments) existing.args += tc.function.arguments
        toolCalls.set(tc.index, existing)
      }
    }
    if (choice.finish_reason) {
      finishReason = finishReasonFromMistral(choice.finish_reason)
    }
  }

  // Flush accumulated tool calls.
  for (const call of toolCalls.values()) {
    if (!call.name) continue
    let parsedArgs: unknown = {}
    if (call.args) {
      try { parsedArgs = JSON.parse(call.args) } catch { parsedArgs = call.args }
    }
    yield {
      type: 'tool-use',
      id: call.id || `mistral_${Math.random().toString(36).slice(2, 10)}`,
      name: call.name,
      args: parsedArgs,
    }
  }

  yield { type: 'finish', reason: finishReason, usage }
}

// ─── Provider implementation ────────────────────────────────────────────────

class MistralProvider implements LLMProvider {
  readonly type = 'mistral'
  readonly displayName = 'Mistral AI'
  readonly apiKeyUrl = 'https://console.mistral.ai/api-keys'
  readonly lobehubIcon = 'Mistral'
  readonly configSchema = CONFIG_SCHEMA
  // Mistral documents 128 as the per-request function-declaration cap.
  readonly defaultMaxTools = 128
  readonly billing = 'per-token' as const

  async authenticate(config: ProviderConfig): Promise<AuthResult> {
    const apiKey = config['apiKey']
    if (!apiKey) return { valid: false, error: 'Missing Mistral API key' }
    try {
      const res = await fetch(`${API_BASE}/models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      })
      if (!res.ok) {
        const err = await errorFromResponse(res)
        return { valid: false, error: err.message }
      }
      return { valid: true }
    } catch (err) {
      return { valid: false, error: err instanceof Error ? err.message : 'Network error' }
    }
  }

  async listModels(config: ProviderConfig): Promise<LLMModel[]> {
    const apiKey = config['apiKey']
    if (!apiKey) throw new Error('Missing Mistral API key')
    const res = await fetch(`${API_BASE}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    })
    if (!res.ok) throw await errorFromResponse(res)
    const payload = (await res.json()) as MistralModelListing

    // Mistral's /v1/models returns one row per model AND one extra row
    // per alias, all sharing the same `name`. Result: `mistral-medium-2508`
    // shows up 4 times (canonical + `mistral-medium-latest` +
    // `mistral-medium` + an internal vibe-cli alias). Plus the
    // endpoint occasionally returns exact ID duplicates (verified for
    // `mistral-large-2512`). Dedupe in two passes:
    //
    //   1. Group by `name` (the canonical identifier on every row).
    //   2. Within each group, prefer the entry where `id === name` —
    //      that's the version-pinned id (e.g. `mistral-medium-2508`).
    //      Pinned beats `-latest` for reproducibility: a Kin pointed at
    //      `mistral-medium-2508` stays on that exact version when Mistral
    //      pushes 2509 next month. Users who want auto-rolling can edit
    //      their Kin to whatever `-latest` alias they prefer.
    const byName = new Map<string, MistralModelListing['data'] extends Array<infer T> ? T : never>()
    for (const m of payload.data ?? []) {
      const groupKey = m.name ?? m.id
      const existing = byName.get(groupKey)
      // Prefer the canonical id (id === name) when we encounter it.
      if (!existing || m.id === groupKey) {
        byName.set(groupKey, m)
      }
    }

    const out: LLMModel[] = []
    for (const m of byName.values()) {
      const chatCapable = m.capabilities?.completion_chat ?? !NON_CHAT_NAME_PATTERN.test(m.id)
      if (!chatCapable) continue

      const supportsImageInput =
        m.capabilities?.vision ?? VISION_PATTERN.test(m.id)
      const supportsTools = m.capabilities?.function_calling ?? true

      const model: LLMModel = {
        id: m.id,
        name: m.name ?? m.id,
        contextWindow: m.max_context_length ?? 0,
        supportsImageInput,
        supportsParallelTools: true,
      }
      if (!supportsTools) model.maxTools = 0
      out.push(model)
    }
    return out
  }

  chat(model: LLMModel, request: ChatRequest, config: ProviderConfig): AsyncIterable<ChatChunk> {
    const apiKey = config['apiKey']
    if (!apiKey) throw new Error('Missing Mistral API key')

    const messages: MistralMessage[] = []
    const sys = systemToMistral(request.system)
    if (sys) messages.push(sys)
    for (const m of messagesToMistral(request.messages)) messages.push(m)

    const body: MistralChatRequest = {
      model: model.id,
      messages,
    }
    if (request.temperature != null) body.temperature = request.temperature
    if (request.maxOutputTokens != null) body.max_tokens = request.maxOutputTokens
    const tools = toolsToMistral(request.tools)
    if (tools) body.tools = tools

    return streamMistral(apiKey, body, request.signal)
  }
}

// ─── Plugin entry point ─────────────────────────────────────────────────────

export default function mistralPlugin(ctx: PluginContext): PluginExports {
  ctx.log.info('mistral plugin loaded')
  return {
    providers: [new MistralProvider()],
  }
}
