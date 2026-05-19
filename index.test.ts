/**
 * Unit tests for the Mistral provider. Network calls are stubbed
 * via fetch mocking; the goal is to assert the request body shape
 * and the stream → ChatChunk normalisation, not to hit api.mistral.ai.
 *
 * To run: `bun test`
 */

import { describe, it, expect, beforeEach, afterEach } from 'bun:test'
import mistralPlugin from './index'
import type { LLMProvider, PluginContext, ChatRequest, KinbotMessage } from '@kinbot-developer/sdk'

// ─── Plugin context stub ────────────────────────────────────────────────────

function makeCtx(): PluginContext {
  return {
    config: {},
    log: { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} },
    storage: {
      get: async () => null,
      set: async () => {},
      delete: async () => {},
      list: async () => [],
      clear: async () => {},
    },
    http: { fetch: globalThis.fetch },
    vault: {
      getSecret: async () => null,
      setSecret: async () => {},
      deleteSecret: async () => {},
      listKeys: async () => [],
    },
    manifest: { name: 'kinbot-plugin-mistral', version: '0.1.0' },
    cards: {
      emit: async () => ({ messageId: 'm', cardInstanceId: 'c' }),
      update: async () => {},
    },
  } as PluginContext
}

function getProvider(): LLMProvider {
  const exports = mistralPlugin(makeCtx())
  const provider = exports.providers?.[0]
  if (!provider) throw new Error('no provider exported')
  return provider as LLMProvider
}

// ─── fetch capture helper ───────────────────────────────────────────────────

interface CapturedCall {
  url: string
  init: RequestInit
}

async function captureFetch<T>(
  responder: (call: CapturedCall) => Response,
  body: () => Promise<T> | T,
): Promise<{ result: T; calls: CapturedCall[] }> {
  const calls: CapturedCall[] = []
  const original = globalThis.fetch
  ;(globalThis as any).fetch = async (url: string, init: RequestInit = {}) => {
    const call = { url, init }
    calls.push(call)
    return responder(call)
  }
  try {
    const result = await body()
    return { result, calls }
  } finally {
    ;(globalThis as any).fetch = original
  }
}

// ─── Metadata ───────────────────────────────────────────────────────────────

describe('mistral provider — metadata', () => {
  it('declares the right type, displayName, and billing', () => {
    const provider = getProvider()
    expect(provider.type).toBe('mistral')
    expect(provider.displayName).toBe('Mistral AI')
    expect(provider.billing).toBe('per-token')
    expect(provider.defaultMaxTools).toBe(128)
    expect(provider.apiKeyUrl).toBe('https://console.mistral.ai/api-keys')
  })

  it('rejects authenticate without an API key (no fetch fired)', async () => {
    const provider = getProvider()
    const result = await provider.authenticate({})
    expect(result.valid).toBe(false)
    expect(result.error).toMatch(/missing/i)
  })
})

// ─── listModels ─────────────────────────────────────────────────────────────

describe('mistral provider — listModels', () => {
  it('hits /v1/models with Bearer auth and maps the response', async () => {
    const provider = getProvider()
    const { result, calls } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          {
            id: 'mistral-large-latest',
            name: 'Mistral Large',
            capabilities: { completion_chat: true, function_calling: true, vision: false },
            max_context_length: 131072,
          },
          {
            id: 'pixtral-large-latest',
            capabilities: { completion_chat: true, vision: true },
            max_context_length: 131072,
          },
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    expect(calls[0]!.url).toBe('https://api.mistral.ai/v1/models')
    expect((calls[0]!.init.headers as Record<string, string>)['Authorization']).toBe('Bearer sk-test')
    expect(result).toHaveLength(2)
    expect(result[0]).toMatchObject({
      id: 'mistral-large-latest',
      contextWindow: 131072,
      supportsImageInput: false,
    })
    expect(result[1]?.supportsImageInput).toBe(true)
  })

  it('filters out embedding and moderation models when capabilities.completion_chat is missing', async () => {
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          { id: 'mistral-large-latest' },
          { id: 'mistral-embed' },
          { id: 'mistral-moderation-latest' },
          { id: 'pixtral-12b-2409' },
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    expect(result.map((m) => m.id)).toEqual(['mistral-large-latest', 'pixtral-12b-2409'])
  })

  it('dedupes by canonical `name` and keeps the version-pinned id (real Mistral payload shape)', async () => {
    // This is the exact shape Mistral's /v1/models returns: one row
    // per model AND one extra row per alias, all sharing the same
    // `name`. See the diagnostic capture from 2026-05 — 4 rows for
    // mistral-medium-2508 (canonical + 3 aliases).
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          {
            id: 'mistral-medium-2508',
            name: 'mistral-medium-2508',
            aliases: ['mistral-medium-latest', 'mistral-medium', 'mistral-vibe-cli-with-tools'],
            capabilities: { completion_chat: true, function_calling: true, vision: true },
            max_context_length: 131072,
          },
          {
            id: 'mistral-medium-latest',
            name: 'mistral-medium-2508',
            aliases: ['mistral-medium-2508', 'mistral-medium', 'mistral-vibe-cli-with-tools'],
            capabilities: { completion_chat: true, function_calling: true, vision: true },
            max_context_length: 131072,
          },
          {
            id: 'mistral-medium',
            name: 'mistral-medium-2508',
            aliases: ['mistral-medium-2508', 'mistral-medium-latest', 'mistral-vibe-cli-with-tools'],
            capabilities: { completion_chat: true, function_calling: true, vision: true },
            max_context_length: 131072,
          },
          {
            id: 'mistral-vibe-cli-with-tools',
            name: 'mistral-medium-2508',
            aliases: ['mistral-medium-2508', 'mistral-medium-latest', 'mistral-medium'],
            capabilities: { completion_chat: true, function_calling: true, vision: true },
            max_context_length: 131072,
          },
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    // 4 input rows → 1 output row
    expect(result).toHaveLength(1)
    // Keeps the version-pinned id (id === name).
    expect(result[0]?.id).toBe('mistral-medium-2508')
  })

  it('handles exact duplicate rows from Mistral (verified mistral-large-2512 ships twice)', async () => {
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          { id: 'mistral-large-2512', name: 'mistral-large-2512', capabilities: { completion_chat: true } },
          { id: 'mistral-large-latest', name: 'mistral-large-2512', capabilities: { completion_chat: true } },
          { id: 'mistral-large-2512', name: 'mistral-large-2512', capabilities: { completion_chat: true } }, // exact dup
          { id: 'mistral-large-latest', name: 'mistral-large-2512', capabilities: { completion_chat: true } }, // exact dup
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    expect(result).toHaveLength(1)
    expect(result[0]?.id).toBe('mistral-large-2512')
  })

  it('falls back to id as the group key when the entry has no `name`', async () => {
    // Some entries (Mistral internal variants) come with no `name`
    // and no aliases — they're their own canonical, treat each id
    // as its own group.
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          { id: 'mistral-medium-3-5', capabilities: { completion_chat: true } },
          { id: 'mistral-medium-3.5', capabilities: { completion_chat: true } },
          { id: 'mistral-medium-3', capabilities: { completion_chat: true } },
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    expect(result.map((m) => m.id).sort()).toEqual(['mistral-medium-3', 'mistral-medium-3-5', 'mistral-medium-3.5'])
  })

  it('marks the model maxTools: 0 when capabilities.function_calling is false', async () => {
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(JSON.stringify({
        data: [
          { id: 'mistral-large-latest', capabilities: { completion_chat: true, function_calling: true } },
          { id: 'legacy-completion-only', capabilities: { completion_chat: true, function_calling: false } },
        ],
      }), { status: 200 }),
      () => provider.listModels({ apiKey: 'sk-test' }),
    )
    expect(result.find((m) => m.id === 'mistral-large-latest')?.maxTools).toBeUndefined()
    expect(result.find((m) => m.id === 'legacy-completion-only')?.maxTools).toBe(0)
  })
})

// ─── chat — request body shape ──────────────────────────────────────────────

describe('mistral provider — chat request shape', () => {
  const model = { id: 'mistral-large-latest', name: 'Mistral Large' }

  function emptyStreamResponse(): Response {
    // Empty SSE stream with one finish chunk so the iterator terminates.
    const body =
      'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0}}\n\n' +
      'data: [DONE]\n\n'
    return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
  }

  it('hoists the SDK system blocks into a role=system message', async () => {
    const provider = getProvider()
    const request: ChatRequest = {
      system: [{ type: 'text', text: 'You are a helpful assistant.' }],
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    }
    const { calls } = await captureFetch(
      () => emptyStreamResponse(),
      async () => {
        for await (const _ of provider.chat(model, request, { apiKey: 'sk-test' })) { /* drain */ }
      },
    )
    const body = JSON.parse(calls[0]!.init.body as string)
    expect(body.messages[0]).toEqual({ role: 'system', content: 'You are a helpful assistant.' })
    expect(body.messages[1]).toEqual({ role: 'user', content: 'hi' })
  })

  it('encodes multimodal user content as parts array with image_url data URL', async () => {
    const provider = getProvider()
    const request: ChatRequest = {
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'caption this' },
          { type: 'image', data: new Uint8Array([0xde, 0xad, 0xbe]), mediaType: 'image/png' },
        ],
      }],
    }
    const { calls } = await captureFetch(
      () => emptyStreamResponse(),
      async () => {
        for await (const _ of provider.chat(model, request, { apiKey: 'sk-test' })) { /* drain */ }
      },
    )
    const body = JSON.parse(calls[0]!.init.body as string)
    expect(body.messages[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: 'caption this' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,3q2+' } },
      ],
    })
  })

  it('splits a user turn so tool-result blocks become role=tool messages with tool_call_id', async () => {
    const provider = getProvider()
    const history: KinbotMessage[] = [
      { role: 'user', content: [{ type: 'text', text: 'what is the weather?' }] },
      {
        role: 'assistant',
        content: [{ type: 'tool-use', id: 'call_1', name: 'get_weather', args: { city: 'Paris' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool-result', toolUseId: 'call_1', content: '22°C sunny' }],
      },
    ]
    const { calls } = await captureFetch(
      () => emptyStreamResponse(),
      async () => {
        for await (const _ of provider.chat(model, { messages: history }, { apiKey: 'sk-test' })) { /* drain */ }
      },
    )
    const body = JSON.parse(calls[0]!.init.body as string)
    expect(body.messages).toHaveLength(3)
    expect(body.messages[1]).toMatchObject({
      role: 'assistant',
      tool_calls: [{
        id: 'call_1',
        type: 'function',
        function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
      }],
    })
    expect(body.messages[2]).toEqual({
      role: 'tool',
      tool_call_id: 'call_1',
      content: '22°C sunny',
    })
  })

  it('maps KinbotTool[] into Mistral function-call definitions', async () => {
    const provider = getProvider()
    const request: ChatRequest = {
      messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }],
      tools: [
        {
          name: 'get_weather',
          description: 'Look up current weather.',
          inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
        },
      ],
    }
    const { calls } = await captureFetch(
      () => emptyStreamResponse(),
      async () => {
        for await (const _ of provider.chat(model, request, { apiKey: 'sk-test' })) { /* drain */ }
      },
    )
    const body = JSON.parse(calls[0]!.init.body as string)
    expect(body.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Look up current weather.',
          parameters: { type: 'object', properties: { city: { type: 'string' } } },
        },
      },
    ])
  })
})

// ─── chat — stream parsing ──────────────────────────────────────────────────

describe('mistral provider — stream parsing', () => {
  const model = { id: 'mistral-large-latest', name: 'Mistral Large' }

  async function collect(sseBody: string): Promise<Array<unknown>> {
    const provider = getProvider()
    const { result } = await captureFetch(
      () => new Response(sseBody, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
      async () => {
        const chunks: unknown[] = []
        for await (const c of provider.chat(
          model,
          { messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }] },
          { apiKey: 'sk-test' },
        )) {
          chunks.push(c)
        }
        return chunks
      },
    )
    return result
  }

  it('yields text-delta chunks for each delta.content and finishes with reason=stop', async () => {
    const body =
      'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n' +
      'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n' +
      'data: [DONE]\n\n'
    const chunks = await collect(body)
    expect(chunks).toEqual([
      { type: 'text-delta', text: 'Hel' },
      { type: 'text-delta', text: 'lo' },
      { type: 'finish', reason: 'stop', usage: { inputTokens: 5, outputTokens: 2 } },
    ])
  })

  it('assembles fragmented tool_calls deltas into one tool-use chunk and finishes with tool-calls', async () => {
    const body =
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather"}}]}}]}\n\n' +
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]}}]}\n\n' +
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"Paris\\"}"}}]}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n' +
      'data: [DONE]\n\n'
    const chunks = await collect(body)
    const toolUse = chunks.find((c: any) => c.type === 'tool-use') as any
    expect(toolUse).toBeDefined()
    expect(toolUse.id).toBe('call_1')
    expect(toolUse.name).toBe('get_weather')
    expect(toolUse.args).toEqual({ city: 'Paris' })
    const finish = chunks[chunks.length - 1] as any
    expect(finish).toMatchObject({ type: 'finish', reason: 'tool-calls' })
  })

  it('maps MAX_TOKENS-style finish reasons to length', async () => {
    const body =
      'data: {"choices":[{"delta":{"content":"truncated"},"finish_reason":"length"}]}\n\n' +
      'data: [DONE]\n\n'
    const chunks = await collect(body)
    const finish = chunks[chunks.length - 1] as any
    expect(finish.reason).toBe('length')
  })

  it('skips malformed JSON in an SSE event rather than aborting the stream', async () => {
    const body =
      'data: {invalid json}\n\n' +
      'data: {"choices":[{"delta":{"content":"survived"},"finish_reason":"stop"}]}\n\n' +
      'data: [DONE]\n\n'
    const chunks = await collect(body)
    const textChunks = chunks.filter((c: any) => c.type === 'text-delta')
    expect(textChunks).toHaveLength(1)
    expect((textChunks[0] as any).text).toBe('survived')
  })
})
