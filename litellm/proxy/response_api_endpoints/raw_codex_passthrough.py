"""Byte-exact passthrough for Codex <-> ChatGPT subscription traffic.

Codex speaks a private dialect of the Responses API when it runs against a
ChatGPT subscription (``chatgpt.com/backend-api/codex``): freeform tool calls
carry ``ctc_``/``ctco_`` ids, inter-agent payloads arrive as encrypted content
blocks, reasoning items hold opaque ``gAAAA...`` payloads that only that
backend can verify, and every turn is pinned to a prompt cache through
``prompt_cache_key`` plus a stable ``session-id`` header.

Routing that traffic through the regular Responses pipeline re-serializes it:
the gateway prepends its own Codex system prompt, rewrites tool-call ids,
filters request fields through an allow-list, re-encodes the response id and
parses/re-serializes the SSE stream. Each of those is a small behavioural
divergence from a direct connection, and some of them (encrypted payloads that
must round-trip untouched) cannot be reproduced after parsing.

This module implements the other option: for deployments that opt in, forward
the request body and the response body as raw bytes. Only the transport headers
that authenticate the call are replaced; everything else — the JSON body, the
SSE stream, the status code, the response headers — reaches Codex exactly as
the subscription backend produced it.

Opting in
---------
Per deployment, in ``litellm_params``:

    raw_codex_passthrough: true     # force on
    raw_codex_passthrough: false    # force off

Deployments whose upstream is ``chatgpt/...`` default to on, because that
prefix means "the subscription backend" and the whole point of this module is
to keep that backend's dialect intact.

What is deliberately skipped
----------------------------
The request never enters ``Router``/``litellm.acompletion``, so for these
calls the gateway does not apply fallbacks, retries or usage/cost accounting.
Client authentication (the proxy master key) still happens, and the deployment
must exist in ``model_list`` — an unknown model name is not passed through.

The one thing that cannot be byte-exact is the ``model`` field: when a
deployment's upstream model name differs from the name Codex asked for, the
field is rewritten and the body is re-serialized (logged as a warning).
Pointing ``model_name`` at the same string the backend expects keeps the whole
request byte-identical.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.llms.chatgpt.authenticator import Authenticator

_CHATGPT_UPSTREAM_PREFIX = "chatgpt/"

# Headers that describe *this* hop. They are recomputed by httpx/Starlette and
# must not be copied from the client request or the upstream response.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        b"host",
        b"content-length",
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)

# Replaced with the gateway's own credential rather than forwarded.
_REPLACED_REQUEST_HEADERS = frozenset({b"authorization"})

# Recomputed when streaming, or meaningless for a body we re-emit ourselves.
_DROPPED_RESPONSE_HEADERS = frozenset(
    {
        b"content-length",
        b"connection",
        b"keep-alive",
        b"transfer-encoding",
        b"upgrade",
    }
)

_CLIENT_TIMEOUT = httpx.Timeout(connect=15.0, read=900.0, write=60.0, pool=15.0)

_shared_client: httpx.AsyncClient | None = None
_shared_client_lock = asyncio.Lock()


class RawPassthroughTarget:
    """Resolved deployment for a byte-exact forward."""

    __slots__ = ("model_name", "upstream_model", "upstream_url")

    def __init__(self, model_name: str, upstream_model: str, upstream_url: str) -> None:
        self.model_name = model_name
        self.upstream_model = upstream_model
        self.upstream_url = upstream_url


def _deployment_flag(litellm_params: Any, key: str) -> Any:
    """Read a custom key off a deployment's litellm_params."""
    getter = getattr(litellm_params, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(litellm_params, key, None)


def _resolve_target(llm_router: Any, model_name: str) -> RawPassthroughTarget | None:
    """Map a requested model name onto a passthrough deployment, if enabled."""
    if llm_router is None:
        return None
    try:
        deployment = llm_router.get_deployment_by_model_group_name(model_group_name=model_name)
    except Exception:  # unknown/invalid model name -> not a passthrough model
        return None
    if deployment is None:
        return None

    litellm_params = getattr(deployment, "litellm_params", None)
    upstream_model = _deployment_flag(litellm_params, "model")
    if not isinstance(upstream_model, str) or not upstream_model:
        return None

    is_subscription_upstream = upstream_model.startswith(_CHATGPT_UPSTREAM_PREFIX)
    enabled = _deployment_flag(litellm_params, "raw_codex_passthrough")
    if enabled is None:
        enabled = is_subscription_upstream
    if not enabled:
        return None

    if is_subscription_upstream:
        upstream_model = upstream_model[len(_CHATGPT_UPSTREAM_PREFIX) :]

    api_base = _deployment_flag(litellm_params, "api_base")
    upstream_url = api_base if isinstance(api_base, str) and api_base else Authenticator().get_api_base()

    # ``api_base``/``CHATGPT_API_BASE`` point at the Codex backend root
    # (``.../backend-api/codex``); the Responses endpoint hangs off it. A base
    # that already names the endpoint is left alone.
    upstream_url = upstream_url.rstrip("/")
    if not upstream_url.endswith("/responses"):
        upstream_url = f"{upstream_url}/responses"

    return RawPassthroughTarget(
        model_name=model_name,
        upstream_model=upstream_model,
        upstream_url=upstream_url,
    )


async def _get_shared_client() -> httpx.AsyncClient:
    """A single connection pool for the subscription backend.

    ``trust_env`` is on by default, so the gateway's HTTP(S)_PROXY settings
    (the egress used to reach chatgpt.com) apply here as they do elsewhere.
    """
    global _shared_client
    async with _shared_client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
        return _shared_client


async def _reset_shared_client() -> None:
    global _shared_client
    async with _shared_client_lock:
        client, _shared_client = _shared_client, None
    if client is not None:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - teardown of a broken client
            pass


def _session_id_from(payload: dict, request: Request) -> str:
    """The session id Codex pinned this conversation to, if it sent one."""
    header_session = request.headers.get("session-id") or request.headers.get("session_id")
    if header_session:
        return header_session
    cache_key = payload.get("prompt_cache_key")
    if isinstance(cache_key, str) and cache_key:
        return cache_key
    client_metadata = payload.get("client_metadata")
    if isinstance(client_metadata, dict):
        for key in ("session_id", "thread_id"):
            value = client_metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return str(uuid4())


def _build_forward_headers(request: Request, payload: dict, access_token: str) -> list[tuple[bytes, bytes]]:
    """Client headers, minus this hop's, plus our credential."""
    raw_headers: list[tuple[bytes, bytes]] = list(request.scope.get("headers") or [])
    forwarded = [
        (name, value)
        for name, value in raw_headers
        if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() not in _REPLACED_REQUEST_HEADERS
    ]
    forwarded.append((b"authorization", f"Bearer {access_token}".encode()))
    if not any(name.lower() == b"session_id" for name, _ in forwarded):
        forwarded.append((b"session_id", _session_id_from(payload, request).encode()))

    authenticator = Authenticator()
    account_id = authenticator.get_account_id()
    if account_id and not any(name.lower() == b"chatgpt-account-id" for name, _ in forwarded):
        forwarded.append((b"chatgpt-account-id", account_id.encode()))
    return forwarded


def _body_for_upstream(raw_body: bytes, payload: dict, target: RawPassthroughTarget) -> bytes:
    """The exact request body, unless the upstream model name must be rewritten."""
    if payload.get("model") == target.upstream_model:
        return raw_body
    rewritten = dict(payload)
    rewritten["model"] = target.upstream_model
    verbose_proxy_logger.warning(
        "raw_codex_passthrough: deployment %s maps to upstream model %s; rewriting the model field "
        "re-serializes this request and is no longer byte-exact",
        target.model_name,
        target.upstream_model,
    )
    return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False).encode()


def _force_token_refresh() -> str | None:
    """Mint a new access token after the backend rejected the current one."""
    authenticator = Authenticator()
    auth_data = authenticator._read_auth_file() or {}
    refresh_token = auth_data.get("refresh_token")
    if not refresh_token:
        return None
    try:
        refreshed = authenticator._refresh_tokens(refresh_token)
    except Exception as exc:  # noqa: BLE001 - surfaces as the original 401
        verbose_proxy_logger.warning("raw_codex_passthrough: token refresh failed: %s", exc)
        return None
    return refreshed.get("access_token")


async def _send(
    client: httpx.AsyncClient,
    target: RawPassthroughTarget,
    headers: list[tuple[bytes, bytes]],
    body: bytes,
) -> httpx.Response:
    request = client.build_request("POST", target.upstream_url, headers=headers, content=body)
    return await client.send(request, stream=True)


async def _forward(
    request: Request,
    raw_body: bytes,
    payload: dict,
    target: RawPassthroughTarget,
) -> Response:
    access_token = Authenticator().get_access_token()
    headers = _build_forward_headers(request, payload, access_token)
    body = _body_for_upstream(raw_body, payload, target)

    client = await _get_shared_client()
    try:
        upstream = await _send(client, target, headers, body)
    except (httpx.TransportError, RuntimeError) as exc:
        # A stale pooled client (e.g. the event loop was replaced) is the one
        # recoverable case here; anything else propagates as a 502.
        verbose_proxy_logger.warning("raw_codex_passthrough: upstream send failed (%s); rebuilding client", exc)
        await _reset_shared_client()
        client = await _get_shared_client()
        upstream = await _send(client, target, headers, body)

    if upstream.status_code == 401:
        refreshed = _force_token_refresh()
        if refreshed:
            await upstream.aclose()
            headers = _build_forward_headers(request, payload, refreshed)
            upstream = await _send(client, target, headers, body)

    response_headers = {
        name.decode("latin-1"): value.decode("latin-1")
        for name, value in upstream.headers.raw
        if name.lower() not in _DROPPED_RESPONSE_HEADERS
    }

    payload_is_stream = payload.get("stream")
    if payload_is_stream is False:
        content = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    async def body_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        content=body_iterator(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def try_raw_codex_passthrough(request: Request, llm_router: Any) -> Response | None:
    """Forward this request byte-for-byte when its deployment opted in.

    Returns ``None`` when the request should take the normal pipeline.
    """
    try:
        raw_body = await request.body()
    except Exception as exc:  # noqa: BLE001 - let the normal path surface it
        verbose_proxy_logger.debug("raw_codex_passthrough: could not read body: %s", exc)
        return None
    if not raw_body:
        return None
    try:
        payload = json.loads(raw_body)
    except Exception:  # noqa: BLE001 - normal pipeline reports malformed JSON
        return None
    if not isinstance(payload, dict):
        return None
    model_name = payload.get("model")
    if not isinstance(model_name, str) or not model_name:
        return None
    # Background/polling mode is served by the gateway's own cache layer and
    # cannot be forwarded as a single upstream call.
    if payload.get("background") is True:
        return None

    target = _resolve_target(llm_router, model_name)
    if target is None:
        return None

    verbose_proxy_logger.debug(
        "raw_codex_passthrough: %s -> %s (byte-exact)", model_name, target.upstream_url
    )
    return await _forward(request, raw_body, payload, target)
