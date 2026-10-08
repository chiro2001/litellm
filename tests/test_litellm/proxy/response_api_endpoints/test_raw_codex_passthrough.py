"""Unit tests for the byte-exact ChatGPT subscription passthrough."""

from unittest.mock import MagicMock

from litellm.proxy.response_api_endpoints.raw_codex_passthrough import (
    RawPassthroughTarget,
    _body_for_upstream,
    _resolve_target,
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
