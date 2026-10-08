# codex-gateway fork

A personal fork of [LiteLLM](https://github.com/BerriAI/litellm) that lets one
[Codex](https://github.com/openai/codex) session drive both the ChatGPT
subscription backend and third-party Responses-API providers (DeepSeek, vLLM,
SCNet, the OpenLux/ofapp/sss proxies) through a single gateway.

Base: `v1.96.2` (`83d6d84bfb`). Branch: `codex-gateway`.

## What this fork adds

### 1. Byte-exact passthrough for the ChatGPT subscription

Codex speaks a private dialect of the Responses API against
`chatgpt.com/backend-api/codex`, and some of it cannot survive a
parse-and-re-serialize trip:

* inter-agent task payloads are encrypted content blocks that only that
  backend can verify
* freeform tool calls carry `ctc_`/`ctco_` ids
* reasoning items hold opaque `gAAAA...` payloads
* every turn is pinned to a prompt cache through `prompt_cache_key` plus a
  stable `session-id` header

Deployments whose upstream is `chatgpt/...` are therefore forwarded **raw**:
the request body and the response body pass through as bytes, only hop-by-hop
headers are dropped and the `Authorization` header is swapped for the OAuth
token. The JSON body, the SSE stream, the status code, the response id and the
remaining headers reach Codex exactly as the backend produced them.

```yaml
- model_name: gpt-6-luna
  litellm_params:
    model: chatgpt/gpt-6-luna
    raw_codex_passthrough: true   # default: on for chatgpt/... upstreams
```

Set `raw_codex_passthrough: false` to route a subscription deployment through
the normal pipeline instead.

Trade-offs, by design: these calls bypass Router retries, fallbacks and usage
accounting, and a conversation whose history contains reasoning items from a
non-ChatGPT provider is rejected by the subscription backend after switching
to a GPT model. Keep GPT sessions on GPT models.

### 2. Responses-API compatibility for third-party providers

The `openai`-shaped transforms normalise Codex's dialect for upstreams that
do not implement it, so DeepSeek and friends can serve as the primary model in
a Codex session:

* `custom_tool_call` / `custom_tool_call_output` history items become
  `function_call` / `function_call_output`
* encrypted-content blocks are flattened to plain text for backends that
  cannot decrypt them (`encrypted_content_passthrough` opts out per deployment)
* truncated function-call arguments are replaced with `{}` instead of being
  replayed verbatim
* reasoning items with non-empty `content` or opaque `encrypted_content`
  (which such backends reject with `array_above_max_length`) are normalised
* Codex's `agent_message` items are rewritten to plain `message` items so
  sub-agent task text survives on backends that ignore the item type

The guard that decides "native dialect or not" keys off the `chatgpt/`
model prefix and an unset `api_base`, rather than `api_base` alone — a
deployment that leaves `api_base` unset means "the provider's own default",
which for the OpenAI config class is OpenAI itself.

## Running it

```bash
pip install -e .            # or point PYTHONPATH at this checkout
litellm --config /path/to/config.yaml --port 11001 --host 127.0.0.1
```

Codex reaches it through a model provider entry:

```toml
[model_providers.litellm]
name = "litellm-gateway"
base_url = "http://127.0.0.1:11001/v1"
wire_api = "responses"
experimental_bearer_token = "<your proxy key>"
```

## Commits on top of upstream

| Commit | Purpose |
| --- | --- |
| `feat(proxy): byte-exact passthrough for ChatGPT subscription traffic` | the raw forwarder (`litellm/proxy/response_api_endpoints/raw_codex_passthrough.py`) and its hook in `endpoints.py` |
| `fix(chatgpt): keep codex prompt-cache routing, drop foreign reasoning items` | forwards `prompt_cache_key`/`client_metadata`/`parallel_tool_calls`/`text`, keeps the client session id in the header |
| `fix(responses): forward client_metadata on the Responses API` | stops the optional-params TypedDict from silently dropping `client_metadata` |
| `local: Codex Responses-API compatibility patches (mixed-model gateway)` | the third-party normalisation described above, plus `ctc_` id rewriting for the subscription route |
