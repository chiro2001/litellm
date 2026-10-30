"""Unit tests for the byte-exact ChatGPT subscription passthrough."""

import gzip
import json
from unittest.mock import MagicMock

import pytest
import zstandard

from litellm.proxy.response_api_endpoints.raw_codex_passthrough import (
    RawPassthroughTarget,
    _body_for_upstream,
    _decode_request_body,
    _decoded_payload,
    _prepare_upstream_body,
    _resolve_target,
    _web_search_compat_payload,
)


def _router_with(deployments: list[tuple[str, dict]]):
    """A router stub whose model group lookup mirrors the real Router."""
    by_name = {}
    for name, litellm_params in deployments:
        deployment = MagicMock()
        deployment.litellm_params = litellm_params
        by_name[name] = deployment
    router = MagicMock()
    router.get_deployment_by_model_group_name.side_effect = lambda model_group_name: by_name.get(model_group_name)
    return router


def test_subscription_upstream_is_passthrough_by_default():
    router = _router_with([("gpt-6-luna", {"model": "chatgpt/gpt-6-luna"})])

    target = _resolve_target(router, "gpt-6-luna")

    assert target is not None
    assert target.upstream_model == "gpt-6-luna"
    assert target.upstream_url.endswith("/responses")


def test_third_party_upstream_is_not_passthrough():
    router = _router_with(
        [
            (
                "deepseek-flash",
                {"model": "openai/deepseek-v4-flash", "api_base": "https://api.deepseek.com/v1"},
            )
        ]
    )

    assert _resolve_target(router, "deepseek-flash") is None


def test_explicit_flag_overrides_the_default():
    router = _router_with(
        [
            ("off", {"model": "chatgpt/gpt-6-luna", "raw_codex_passthrough": False}),
            (
                "on",
                {"model": "openai/whatever", "api_base": "https://example.test/v1", "raw_codex_passthrough": True},
            ),
        ]
    )

    assert _resolve_target(router, "off") is None
    assert _resolve_target(router, "on") is not None


def test_unknown_model_is_not_passthrough():
    router = _router_with([])

    assert _resolve_target(router, "does-not-exist") is None


def test_api_base_naming_the_endpoint_is_not_doubled():
    router = _router_with(
        [("gpt", {"model": "chatgpt/gpt-6-luna", "api_base": "https://example.test/backend-api/codex/responses"})]
    )

    target = _resolve_target(router, "gpt")

    assert target is not None
    assert target.upstream_url == "https://example.test/backend-api/codex/responses"


def test_zstd_request_compression_is_resolved_from_deployment():
    router = _router_with(
        [
            (
                "gpt",
                {
                    "model": "chatgpt/gpt-6-luna",
                    "raw_codex_request_compression": "zstd",
                },
            )
        ]
    )

    target = _resolve_target(router, "gpt")

    assert target is not None
    assert target.request_compression == "zstd"


def test_unknown_request_compression_is_rejected():
    router = _router_with(
        [
            (
                "gpt",
                {
                    "model": "chatgpt/gpt-6-luna",
                    "raw_codex_request_compression": "gzip",
                },
            )
        ]
    )

    with pytest.raises(ValueError, match="must be omitted, false, or 'zstd'"):
        _resolve_target(router, "gpt")


def test_body_is_forwarded_untouched_when_the_model_names_match():
    raw = b'{"model":"gpt-6-luna","input":[],"store":false}'
    target = RawPassthroughTarget("gpt-6-luna", "gpt-6-luna", "https://example.test/responses")

    assert _body_for_upstream(raw, {"model": "gpt-6-luna"}, target) is raw


def test_body_is_rewritten_when_the_upstream_model_differs():
    raw = b'{"model":"alias","input":[]}'
    target = RawPassthroughTarget("alias", "gpt-6-luna", "https://example.test/responses")

    rewritten = _body_for_upstream(raw, {"model": "alias", "input": []}, target)

    assert rewritten != raw
    assert b'"model":"gpt-6-luna"' in rewritten


def _compaction_shaped_payload(tools_marker=..., tool_choice_marker=...):
    payload = {
        "model": "gpt-6.1-sol",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "web_search_call",
                "id": "ws_00000000000000000001",
                "status": "completed",
                "action": {"type": "search", "query": "example"},
            },
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "found"}]},
        ],
        "stream": True,
        "store": False,
    }
    if tools_marker is not ...:
        payload["tools"] = tools_marker
    if tool_choice_marker is not ...:
        payload["tool_choice"] = tool_choice_marker
    return payload


def test_compaction_request_gets_the_missing_web_search_declaration():
    payload = _compaction_shaped_payload(tools_marker=[], tool_choice_marker="auto")

    rewritten = _web_search_compat_payload(payload)

    assert rewritten is not None
    assert rewritten["tools"] == [{"type": "web_search", "external_web_access": False}]
    assert rewritten["tool_choice"] == "none"
    assert rewritten["input"] == payload["input"]


def test_declared_web_search_is_left_alone():
    payload = _compaction_shaped_payload(tools_marker=[{"type": "web_search"}], tool_choice_marker="auto")

    assert _web_search_compat_payload(payload) is None


def test_search_free_history_needs_no_rewrite():
    payload = _compaction_shaped_payload(tools_marker=[])
    payload["input"] = [item for item in payload["input"] if item.get("type") != "web_search_call"]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    target = RawPassthroughTarget("gpt-6.1-sol", "gpt-6.1-sol", "https://example.test/responses")

    assert _body_for_upstream(raw, payload, target) is raw


def test_model_and_declaration_rewrites_combine():
    payload = _compaction_shaped_payload(tools_marker=[])
    raw = json.dumps(payload, separators=(",", ":")).encode()
    target = RawPassthroughTarget("gpt-6.1-sol", "gpt-6.1-sol-2026", "https://example.test/responses")

    parsed = json.loads(_body_for_upstream(raw, payload, target))

    assert parsed["model"] == "gpt-6.1-sol-2026"
    assert parsed["tools"] == [{"type": "web_search", "external_web_access": False}]


def test_configured_zstd_compression_covers_all_request_rewrites():
    payload = _compaction_shaped_payload(tools_marker=[])
    raw = json.dumps(payload, separators=(",", ":")).encode()
    target = RawPassthroughTarget(
        "gpt-6.1-sol",
        "gpt-6.1-sol-2026",
        "https://example.test/responses",
        request_compression="zstd",
    )

    compressed, encoding = _prepare_upstream_body(raw, raw, payload, target, None)

    assert encoding == "zstd"
    decoded = json.loads(zstandard.ZstdDecompressor().decompress(compressed))
    assert decoded["model"] == "gpt-6.1-sol-2026"
    assert decoded["tools"] == [{"type": "web_search", "external_web_access": False}]
    assert decoded["input"] == payload["input"]


def test_configured_zstd_compression_replaces_incoming_gzip_encoding():
    raw_payload = b'{"model":"gpt-6-luna","input":[]}'
    gzip_body = gzip.compress(raw_payload)
    decoded = gzip.decompress(gzip_body)
    target = RawPassthroughTarget(
        "gpt-6-luna",
        "gpt-6-luna",
        "https://example.test/responses",
        request_compression="zstd",
    )

    compressed, encoding = _prepare_upstream_body(
        gzip_body,
        decoded,
        json.loads(decoded),
        target,
        "gzip",
    )

    assert encoding == "zstd"
    assert zstandard.ZstdDecompressor().decompress(compressed) == raw_payload


def test_compressed_request_is_not_altered_without_configured_compression():
    raw_payload = b'{"model":"gpt-6-luna","input":[{"type":"reasoning","encrypted_content":"gAAAA_KEEP"}]}'
    compressed = zstandard.ZstdCompressor().compress(raw_payload)
    decoded = _decode_request_body(compressed, "zstd")
    assert decoded is not None
    payload = _decoded_payload(decoded)
    assert payload is not None
    target = RawPassthroughTarget("gpt-6-luna", "gpt-6-luna", "https://example.test/responses")

    forwarded, encoding = _prepare_upstream_body(
        compressed,
        decoded,
        payload,
        target,
        "zstd",
    )

    assert forwarded is compressed
    assert encoding == "zstd"
    assert payload["input"][0]["encrypted_content"] == "gAAAA_KEEP"
