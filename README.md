# 🌬️ kinbot-plugin-mistral

[Mistral AI](https://mistral.ai) provider for [KinBot](https://github.com/MarlBurroW/kinbot). Brings Mistral's chat models (`mistral-large-*`, `mistral-medium-*`, `mistral-small-*`, `pixtral-*` for vision, …) **and Voxtral speech-to-text** (`voxtral-mini-2507`, `voxtral-small-2507`) to KinBot with **native tool calling**, **streaming**, and **vision** (Pixtral family).

## Install

In KinBot's web UI:

1. **Settings → Plugins → Browse → npm tab**
2. Search for `mistral`
3. Click **Install** on `kinbot-plugin-mistral`
4. **Settings → Providers → Add provider → Mistral AI**
5. Paste your API key from [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys)

Alternative: install from git instead of npm via **Settings → Plugins → Install from git** with `https://github.com/MarlBurroW/kinbot-plugin-mistral` as the URL.

## What this plugin does

Contributes two native providers to KinBot's provider registry, both under `type: 'mistral'` so a single API-key row covers chat + transcription (same pattern the host's built-in OpenAI provider uses across LLM / Embedding / Image / TTS / STT):

- **LLMProvider** — chat against `/v1/chat/completions`
- **STTProvider** — Voxtral against `/v1/audio/transcriptions`

From there the model picker on every Kin sees Mistral models alongside Anthropic / OpenAI / Gemini built-ins — no special-casing.

**Chat capabilities**
- ✅ Streaming chat via SSE (`POST /v1/chat/completions`)
- ✅ Native function / tool calling (`tools` + `tool_calls`)
- ✅ Vision (Pixtral and recent large/medium models — images passed as `data:` URLs)
- ✅ Live model catalogue (`GET /v1/models`) — embedding / moderation / OCR models filtered out
- ✅ Schema-driven capability detection — when Mistral adds a new model with `capabilities.function_calling: true`, it just works

**Voxtral STT capabilities**
- ✅ Two production models: `voxtral-mini-2507`, `voxtral-small-2507`
- ✅ Language hint (`language` form field, ISO 639-1)
- ✅ Auto-detected language returned in the response
- ✅ Segment-level timestamps when the host asks for them (`timestamp_granularities[]=segment`)
- ❌ Diarization (speaker labels) — Voxtral doesn't ship it; diarize hints arrive as a warning
- ❌ Prompt biasing (Whisper-style vocabulary biasing) — no `prompt` field on the upstream API

**Out of scope (for now)**
- Embeddings (`mistral-embed`) — could be a third provider in this plugin; raise an issue if you need it
- Moderation (`mistral-moderation-latest`) — KinBot doesn't have a moderation primitive yet
- OCR (`mistral-ocr-*`) — not yet a KinBot primitive
- Codestral fill-in-the-middle — completion-style, not chat

## Development

```bash
git clone https://github.com/MarlBurroW/kinbot-plugin-mistral
cd kinbot-plugin-mistral
bun install
bun test
```

To iterate against a local KinBot install: drop this directory (or a symlink to it) into KinBot's `plugins/` folder and reload the plugins from the UI. The plugin loader hot-reloads on file change.

## Permissions

The plugin declares a single permission in its manifest:

- `http:api.mistral.ai` — required for chat / model-listing / transcription requests

No vault, storage, cards, cron, or kins permissions needed.

## License

MIT
