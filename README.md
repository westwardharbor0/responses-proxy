# Responses Proxy

OpenAI-compatible shim for Azure OpenAI **Responses**. It exposes `/v1/chat/completions` and `/v1/models` so tools that only speak the OpenAI API (e.g., Xcode’s intelligence chat or other SDKs) can talk to an Azure deployment that only provides Responses. Typical use: Xcode external models expect the OpenAI `chat/completions` endpoint, while Microsoft Foundry Responses API only exposes `responses`—this proxy bridges that gap. It rewrites chat payloads into the Responses `input` shape, forwards to `/openai/responses`, and returns an OpenAI-style chat completion (streaming or non-streaming) back to the client.

## Highlights
- `/v1/chat/completions` -> `/openai/responses`
- Message mapping: user/system -> `input_text`, assistant -> `output_text`
- Streaming (SSE) and non-streaming
- 300s upstream timeout
- Minimal logging by default; `LOG_VERBOSE=true` for full headers/body and upstream info
- Static `/v1/models` response

## Quick start
```bash
cp .env.example .env
# edit .env with your endpoint, key, and deployment name
make run
# listens on :8080
```

## Configuration
- `LLM_ENDPOINT` (required) e.g. `https://<resource>.openai.azure.com`
- `LLM_API_KEY` (required) Azure API key
- `LLM_API_VERSION` (optional) default `2025-04-01-preview`
- `LLM_DEPLOYMENT` or `LLM_MODEL` **required**: your Responses deployment name; proxy forces this and ignores client-supplied model
- `LOG_VERBOSE` `true|false` verbose logging toggle

## Endpoints
- `GET /v1/models` returns a single **static** model entry (placeholder; not pulled from Azure)
- `POST /v1/chat/completions` forwarded to `/openai/responses`; supports `stream: true`

## Behavior
- SSE replies use `chat.completion.chunk` events and end with `data: [DONE]`
- Errors are returned as HTTP 502 with details logged
- Binds to `:8080` by default

## Requirements
- Go 1.22+
- Azure OpenAI resource with a Responses-enabled deployment
