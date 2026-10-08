from typing import TYPE_CHECKING, Any, cast, get_type_hints

import httpx
import json
from openai.types.responses import ResponseReasoningItem
from pydantic import BaseModel, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.litellm_completion_transformation.custom_tools import (
    normalize_custom_tool_call_item,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import *
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import OpenAIError

OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS = 16

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenAIResponsesAPIConfig(BaseResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def supports_native_file_search(self) -> bool:
        return True

    @staticmethod
    def _is_gpt_5_model(model: str) -> bool:
        """Return True only for actual OpenAI GPT-5 models.

        Excludes pass-through models from other providers that happen to
        reference gpt-5 in their name (e.g. perplexity/openai/gpt-5.2).
        """
        parts = model.split("/")
        if len(parts) > 1 and parts[0] not in ("openai",):
            return False
        return "gpt-5" in model and "gpt-5-chat" not in model

    @staticmethod
    def _supports_reasoning_effort_none(model: str) -> bool:
        """Return True if the model supports reasoning.effort='none'."""
        from litellm.utils import _supports_factory

        return _supports_factory(
            model=model,
            custom_llm_provider=None,
            key="supports_none_reasoning_effort",
        )

    @staticmethod
    def _enforce_min_max_output_tokens(max_output_tokens: "int | None") -> "int | None":
        """Raise sub-minimum max_output_tokens up to the OpenAI Responses API minimum.

        OpenAI's Responses API rejects max_output_tokens below 16 for every model
        (not gpt-5 specific), so a client like Claude Code that sends a max_tokens=1
        warmup probe on model switch would otherwise 400. Values that are None or
        already at/above the minimum are returned unchanged.
        """
        if isinstance(max_output_tokens, int) and max_output_tokens < OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS:
            return OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS
        return max_output_tokens

    def get_supported_openai_params(self, model: str) -> list:
        """
        All OpenAI Responses API params are supported
        """
        supported_params = get_type_hints(ResponsesAPIRequestParams).keys()
        return list(
            set(
                [
                    "input",
                    "model",
                    "extra_headers",
                    "extra_query",
                    "extra_body",
                    "timeout",
                ]
                + list(supported_params)
            )
        )

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """No mapping applied since inputs are in OpenAI spec already.

        GPT-5 models have restrictions on temperature (only temperature=1
        is accepted unless reasoning_effort='none' on models that support it).
        Apply the same validation used by the chat completions path.
        """
        params = dict(response_api_optional_params)

        if "max_output_tokens" in params:
            params["max_output_tokens"] = self._enforce_min_max_output_tokens(params.get("max_output_tokens"))

        if self._is_gpt_5_model(model=model):
            temperature = params.get("temperature")
            if temperature is not None and temperature != 1:
                reasoning = params.get("reasoning") or {}
                effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
                supports_none = self._supports_reasoning_effort_none(model=model)
                if supports_none and (effort == "none" or effort is None):
                    pass  # flexible temperature allowed
                elif drop_params or litellm.drop_params:
                    params.pop("temperature", None)
                else:
                    raise litellm.UnsupportedParamsError(
                        message=(
                            f"gpt-5 models don't support temperature={temperature}. "
                            "Only temperature=1 is supported. "
                            "For models like gpt-5.1/5.4, temperature is supported "
                            "when reasoning.effort='none' (or not specified). "
                            "To drop unsupported params set `litellm.drop_params = True`"
                        ),
                        status_code=400,
                    )

        return params

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Strip Anthropic-only `cache_control` markers before sending to OpenAI.

        OpenAI's Responses API rejects unknown fields on input content blocks
        with HTTP 400 ("Unknown parameter: 'input[0].content[0].cache_control'").
        Chat Completions strips these in
        `remove_cache_control_flag_from_messages_and_tools`; mirror that here.
        """

        if litellm_params.get("drop_metadata"):
            response_api_optional_request_params.pop("metadata", None)
        input = self._validate_input_param(input)
        input = self._flatten_encrypted_content_in_input(input=input, litellm_params=litellm_params)
        input = self._normalize_custom_tool_calls_for_third_party(input=input, litellm_params=litellm_params)
        input = self._sanitize_function_call_arguments_for_third_party(input=input, litellm_params=litellm_params)
        tools = response_api_optional_request_params.get("tools")
        input, tools = self.remove_cache_control_flag_from_input_and_tools(model=model, input=input, tools=tools)
        if tools is not None:
            response_api_optional_request_params["tools"] = tools
        final_request_params = dict(
            ResponsesAPIRequestParams(model=model, input=input, **response_api_optional_request_params)
        )

        return final_request_params

    @staticmethod
    def _speaks_native_codex_dialect(litellm_params: GenericLiteLLMParams) -> bool:
        """Whether this request goes to a backend that natively speaks Codex's dialect.

        Two do: OpenAI's own API, and the ChatGPT subscription endpoint
        (``chatgpt.com/backend-api/codex``) that Codex itself talks to when it
        runs against a subscription. Everything in this class labelled "for
        third party" exists to paper over endpoints that choke on Codex's input
        shapes — vLLM among them — and both of those handle the shapes natively.
        Rewriting them there is at best a loss and at worst a hard 400.

        The check cannot be ``api_base`` alone. A deployment that leaves
        ``api_base`` unset sees ``None`` here rather than the provider's default,
        so an ``api_base``-only guard silently failed open for the subscription
        backend: its ``custom_tool_call`` items were rewritten to
        ``function_call`` while keeping their ``ctc_`` ids, and the backend
        answered::

            Invalid 'input[9].id': 'ctc_07c5...'. Expected an ID that begins with 'fc'.

        ``chatgpt/`` in the model string is the reliable signal for the
        subscription route — it is the prefix the provider is always configured
        with.
        """
        if isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")
            model = litellm_params.get("model")
        else:
            api_base = getattr(litellm_params, "api_base", None)
            model = getattr(litellm_params, "model", None)

        # The subscription route names itself in the model string; it never sets
        # its own `api_base`, so this is the only place the prefix is visible.
        if isinstance(model, str) and model.startswith("chatgpt/"):
            return True

        # No `api_base` at all means "the provider's own default", and for this
        # class that default is OpenAI. Treating it as third-party was the
        # fail-open that produced the 400 above, and it is the wrong guess in
        # general: an unset base is not evidence of an unsupported endpoint.
        if api_base is None:
            return True

        # An explicit base is the one reliable third-party signal — every
        # non-OpenAI deployment in this gateway sets one (the DGX vLLM at
        # 127.0.0.1:8888, SCNet, the OpenLux/ofapp/sssaicode proxies).
        return isinstance(api_base, str) and "api.openai.com" in api_base

    def _flatten_encrypted_content_in_input(
        self,
        input: str | ResponseInputParam,
        litellm_params: GenericLiteLLMParams,
    ) -> str | ResponseInputParam:
        """Normalize Codex inter-agent input for third-party Responses endpoints.

        Codex delivers inter-agent task payloads as `agent_message` input items
        whose content blocks are `encrypted_content`; only OpenAI's backend can
        decrypt them. Third-party Responses endpoints reject the encrypted block
        with a 400 (unknown variant) and ignore the `agent_message` item type,
        leaving the sub-agent with an empty task payload. Rewrite such items to
        plain `message` items with `input_text` content whenever the upstream is
        not OpenAI's official API.
        """
        if self._speaks_native_codex_dialect(litellm_params):
            return input
        # Per-deployment opt-out: forward the encrypted blocks verbatim so the
        # upstream can try to decrypt them itself.
        if isinstance(litellm_params, dict):
            passthrough = litellm_params.get("encrypted_content_passthrough")
        else:
            passthrough = getattr(litellm_params, "encrypted_content_passthrough", None)
        if passthrough is True:
            return input
        if not isinstance(input, list):
            return input
        for item in input:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            flattened_content: list[Any] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "encrypted_content"
                    and block.get("encrypted_content") is not None
                ):
                    flattened_content.append({"type": "input_text", "text": block["encrypted_content"]})
                else:
                    flattened_content.append(block)
            item["content"] = flattened_content
            if item.get("type") == "agent_message":
                item["type"] = "message"
                item["role"] = "user"
                item.pop("author", None)
                item.pop("recipient", None)
        return input

    def _normalize_custom_tool_calls_for_third_party(
        self,
        input: str | ResponseInputParam,
        litellm_params: GenericLiteLLMParams,
    ) -> str | ResponseInputParam:
        """Convert Codex ``custom_tool_call`` history items to ``function_call``.

        Codex sub-agents and tools (e.g. apply_patch) emit Responses API input
        items with ``type="custom_tool_call"``. Several third-party Responses
        endpoints (vLLM included) fail to parse those items and return 500
        (``'ResponseCustomToolCall' object has no attribute 'get'``). For any
        non-OpenAI upstream we rewrite them to the standard ``function_call``
        shape so the client-facing Responses API keeps working end-to-end.
        """
        if self._speaks_native_codex_dialect(litellm_params):
            return input
        if not isinstance(input, list):
            return input
        for item in input:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "custom_tool_call":
                raw_input = item.pop("input", "")
                item["type"] = "function_call"
                item["arguments"] = json.dumps({"input": raw_input}, ensure_ascii=False)
            elif item.get("type") == "custom_tool_call_output":
                item["type"] = "function_call_output"
        return input

    def _sanitize_function_call_arguments_for_third_party(
        self,
        input: str | ResponseInputParam,
        litellm_params: GenericLiteLLMParams,
    ) -> str | ResponseInputParam:
        """Replace invalid ``function_call`` arguments with ``{}`` before forwarding.

        Some backends (e.g. vLLM) reject conversation history whose
        ``function_call`` items carry truncated/non-JSON ``arguments`` (the model
        may have been cut off mid-string). OpenAI's API tolerates them as opaque
        strings, but stricter third-party endpoints return 400 and abort the
        whole turn. Normalizing invalid JSON to ``{}`` keeps the request flowing;
        the accompanying ``function_call_output`` error text still gives the
        model the context it needs.
        """
        if self._speaks_native_codex_dialect(litellm_params):
            return input
        if not isinstance(input, list):
            return input
        for item in input:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                item["arguments"] = "{}"
        return input

    def remove_cache_control_flag_from_input_and_tools(
        self,
        model: str,  # allows overrides to selectively run this
        input: str | ResponseInputParam,
        tools: List[ALL_RESPONSES_API_TOOL_PARAMS] | None = None,
    ) -> Tuple[
        str | ResponseInputParam,
        List[ALL_RESPONSES_API_TOOL_PARAMS] | None,
    ]:
        """Sibling of `remove_cache_control_flag_from_messages_and_tools` on
        the chat path. Strips Anthropic-only `cache_control` markers from
        Responses API input content blocks and tools.

        `filter_value_from_dict` mutates each dict in place, so the same
        objects are returned.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            filter_value_from_dict,
        )

        if isinstance(input, list):
            for item in input:
                if isinstance(item, dict):
                    filter_value_from_dict(cast(dict, item), "cache_control")

        if tools is not None:
            for tool in tools:
                if isinstance(tool, dict):
                    filter_value_from_dict(cast(dict, tool), "cache_control")

        return input, tools

    def _validate_input_param(self, input: str | ResponseInputParam) -> str | ResponseInputParam:
        """
        Ensure all input fields if pydantic are converted to dict

        OpenAI API Fails when we try to JSON dumps specific input pydantic fields.
        This function ensures all input fields are converted to dict.
        """
        if isinstance(input, list):
            validated_input = []
            for item in input:
                # if it's pydantic, convert to dict
                if isinstance(item, BaseModel):
                    validated_input.append(item.model_dump(exclude_none=True))
                elif isinstance(item, dict):
                    # Handle reasoning items specifically to filter out status=None
                    if item.get("type") == "reasoning":
                        verbose_logger.debug(f"Handling reasoning item: {item}")
                        # Type assertion since we know it's a dict at this point
                        dict_item = cast(dict[str, Any], item)
                        filtered_item = self._handle_reasoning_item(dict_item)
                    else:
                        # For other dict items, just pass through
                        filtered_item = cast(dict[str, Any], item)
                    validated_input.append(filtered_item)
                else:
                    validated_input.append(item)
            return validated_input  # type: ignore
        # Input is expected to be either str or List, no single BaseModel expected
        return input

    def _handle_reasoning_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Handle reasoning items specifically to filter out status=None using OpenAI's model.
        Issue: https://github.com/BerriAI/litellm/issues/13484
        OpenAI API does not accept ReasoningItem(status=None), so we need to:
        1. Check if the item is a reasoning type
        2. Create a ResponseReasoningItem object with the item data
        3. Convert it back to dict with exclude_none=True to filter None values
        """
        if item.get("type") == "reasoning":
            try:
                # Ensure required fields are present for ResponseReasoningItem
                item_data = dict(item)
                if "summary" not in item_data:
                    item_data["summary"] = (
                        item_data.get("reasoning_content", "")[:100] + "..."
                        if len(item_data.get("reasoning_content", "")) > 100
                        else item_data.get("reasoning_content", "")
                    )

                # Create ResponseReasoningItem object from the item data
                reasoning_item = ResponseReasoningItem(**item_data)

                # Convert back to dict with exclude_none=True to exclude None fields
                dict_reasoning_item = reasoning_item.model_dump(exclude_none=True)

                return dict_reasoning_item
            except Exception as e:
                verbose_logger.debug(f"Failed to create ResponseReasoningItem, falling back to manual filtering: {e}")
                # Fallback: manually filter out known None fields
                filtered_item = {
                    k: v
                    for k, v in item.items()
                    if v is not None or k not in {"status", "content", "encrypted_content"}
                }
                return filtered_item
        return item

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_tool_names: set[str] | None = None,
    ) -> ResponsesAPIResponse:
        """No transform applied since outputs are in OpenAI spec already"""
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json = raw_response.json()
            raw_response_json["created_at"] = _safe_convert_created_field(raw_response_json["created_at"])
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)
        if custom_tool_names:
            output = raw_response_json.get("output")
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        normalize_custom_tool_call_item(item, custom_tool_names)
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)
        try:
            response = ResponsesAPIResponse.model_validate(raw_response_json)
        except Exception:
            verbose_logger.debug(f"Error constructing ResponsesAPIResponse: {raw_response_json}, using model_construct")
            response = ResponsesAPIResponse.model_construct(**raw_response_json)

        # Store processed headers in additional_headers so they get returned to the client
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
        headers.setdefault("Content-Type", "application/json")
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the endpoint for OpenAI responses API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk into a ResponsesAPIStreamingResponse
        """
        # Convert the dictionary to a properly typed ResponsesAPIStreamingResponse
        verbose_logger.debug("Raw OpenAI Chunk=%s", parsed_chunk)
        event_type = str(parsed_chunk.get("type"))
        event_pydantic_model = OpenAIResponsesAPIConfig.get_event_model_class(event_type=event_type)
        # Some OpenAI-compatible providers send error.code: null; coalesce so validation succeeds.
        try:
            error_obj = parsed_chunk.get("error")
            if isinstance(error_obj, dict) and error_obj.get("code") is None:
                parsed_chunk = dict(parsed_chunk)
                parsed_chunk["error"] = dict(error_obj)
                parsed_chunk["error"]["code"] = "unknown_error"
        except Exception:
            verbose_logger.debug("Failed to coalesce error.code in parsed_chunk")

        try:
            return event_pydantic_model(**parsed_chunk)
        except ValidationError:
            verbose_logger.debug(
                "Pydantic validation failed for %s with chunk %s, falling back to model_construct",
                event_pydantic_model.__name__,
                parsed_chunk,
            )
            return event_pydantic_model.model_construct(**parsed_chunk)

    @staticmethod
    def get_event_model_class(event_type: str) -> Any:
        """
        Returns the appropriate event model class based on the event type.

        Args:
            event_type (str): The type of event from the response chunk

        Returns:
            Any: The corresponding event model class

        Raises:
            ValueError: If the event type is unknown
        """
        event_models = {
            ResponsesAPIStreamEvents.RESPONSE_CREATED: ResponseCreatedEvent,
            ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS: ResponseInProgressEvent,
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED: ResponseCompletedEvent,
            ResponsesAPIStreamEvents.RESPONSE_FAILED: ResponseFailedEvent,
            ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE: ResponseIncompleteEvent,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED: OutputItemAddedEvent,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE: OutputItemDoneEvent,
            ResponsesAPIStreamEvents.CONTENT_PART_ADDED: ContentPartAddedEvent,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE: ContentPartDoneEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA: OutputTextDeltaEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED: OutputTextAnnotationAddedEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE: OutputTextDoneEvent,
            ResponsesAPIStreamEvents.REFUSAL_DELTA: RefusalDeltaEvent,
            ResponsesAPIStreamEvents.REFUSAL_DONE: RefusalDoneEvent,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA: FunctionCallArgumentsDeltaEvent,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE: FunctionCallArgumentsDoneEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_IN_PROGRESS: FileSearchCallInProgressEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_SEARCHING: FileSearchCallSearchingEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_COMPLETED: FileSearchCallCompletedEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_IN_PROGRESS: WebSearchCallInProgressEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_SEARCHING: WebSearchCallSearchingEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_COMPLETED: WebSearchCallCompletedEvent,
            ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS: MCPListToolsInProgressEvent,
            ResponsesAPIStreamEvents.MCP_LIST_TOOLS_COMPLETED: MCPListToolsCompletedEvent,
            ResponsesAPIStreamEvents.MCP_LIST_TOOLS_FAILED: MCPListToolsFailedEvent,
            ResponsesAPIStreamEvents.MCP_CALL_IN_PROGRESS: MCPCallInProgressEvent,
            ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA: MCPCallArgumentsDeltaEvent,
            ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE: MCPCallArgumentsDoneEvent,
            ResponsesAPIStreamEvents.MCP_CALL_COMPLETED: MCPCallCompletedEvent,
            ResponsesAPIStreamEvents.MCP_CALL_FAILED: MCPCallFailedEvent,
            ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE: ImageGenerationPartialImageEvent,
            ResponsesAPIStreamEvents.ERROR: ErrorEvent,
            # Shell tool events: passthrough as GenericEvent so payload is preserved
            ResponsesAPIStreamEvents.SHELL_CALL_IN_PROGRESS: GenericEvent,
            ResponsesAPIStreamEvents.SHELL_CALL_COMPLETED: GenericEvent,
            ResponsesAPIStreamEvents.SHELL_CALL_OUTPUT: GenericEvent,
        }

        model_class = event_models.get(cast(ResponsesAPIStreamEvents, event_type))
        if not model_class:
            return GenericEvent

        return model_class

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        if stream is not True:
            return False
        if model is not None:
            try:
                if (
                    litellm.utils.supports_native_streaming(
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                    )
                    is False
                ):
                    return True
            except Exception as e:
                verbose_logger.debug(f"Error getting model info in OpenAIResponsesAPIConfig: {e}")
        return False

    def supports_native_websocket(self) -> bool:
        """OpenAI supports native WebSocket for Responses API"""
        return True

    #########################################################
    ########## DELETE RESPONSE API TRANSFORMATION ##############
    #########################################################
    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, dict]:
        """
        Transform the delete response API request into a URL and data

        OpenAI API expects the following request
        - DELETE /v1/responses/{response_id}
        """
        encoded_response_id = encode_url_path_segment(response_id, field_name="response_id")
        url = f"{api_base}/{encoded_response_id}"
        data: dict = {}
        return url, data

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        """
        Transform the delete response API response into a DeleteResponseResult
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)
        return DeleteResponseResult(**raw_response_json)

    #########################################################
    ########## GET RESPONSE API TRANSFORMATION ###############
    #########################################################
    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, dict]:
        """
        Transform the get response API request into a URL and data

        OpenAI API expects the following request
        - GET /v1/responses/{response_id}
        """
        encoded_response_id = encode_url_path_segment(response_id, field_name="response_id")
        url = f"{api_base}/{encoded_response_id}"
        data: dict = {}
        return url, data

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform the get response API response into a ResponsesAPIResponse
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)
        response = ResponsesAPIResponse.model_validate(raw_response_json)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers

        return response

    #########################################################
    ########## LIST INPUT ITEMS TRANSFORMATION #############
    #########################################################
    def transform_list_input_items_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        before: str | None = None,
        include: List[str] | None = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> Tuple[str, dict]:
        encoded_response_id = encode_url_path_segment(response_id, field_name="response_id")
        url = f"{api_base}/{encoded_response_id}/input_items"
        params: dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if include:
            params["include"] = ",".join(include)
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        return url, params

    def transform_list_input_items_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> dict:
        try:
            return raw_response.json()
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)

    #########################################################
    ########## CANCEL RESPONSE API TRANSFORMATION ##########
    #########################################################
    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, dict]:
        """
        Transform the cancel response API request into a URL and data

        OpenAI API expects the following request
        - POST /v1/responses/{response_id}/cancel
        """
        encoded_response_id = encode_url_path_segment(response_id, field_name="response_id")
        url = f"{api_base}/{encoded_response_id}/cancel"
        data: dict = {}
        return url, data

    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform the cancel response API response into a ResponsesAPIResponse
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)

        response = ResponsesAPIResponse.model_validate(raw_response_json)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers

        return response

    #########################################################
    ########## COMPACT RESPONSE API TRANSFORMATION ##########
    #########################################################
    def transform_compact_response_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, dict]:
        """
        Transform the compact response API request into a URL and data

        OpenAI API expects the following request
        - POST /v1/responses/compact
        """
        # Preserve query params (e.g., api-version) while appending /compact.
        parsed_url = httpx.URL(api_base)
        compact_path = parsed_url.path.rstrip("/") + "/compact"
        url = str(parsed_url.copy_with(path=compact_path))

        input = self._validate_input_param(input)
        tools = response_api_optional_request_params.get("tools")
        input, tools = self.remove_cache_control_flag_from_input_and_tools(model=model, input=input, tools=tools)
        if tools is not None:
            response_api_optional_request_params["tools"] = tools
        data = dict(ResponsesAPIRequestParams(model=model, input=input, **response_api_optional_request_params))

        return url, data

    def transform_compact_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform the compact response API response into a ResponsesAPIResponse
        """
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json = raw_response.json()
            raw_response_json["created_at"] = _safe_convert_created_field(raw_response_json["created_at"])
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)

        try:
            response = ResponsesAPIResponse.model_validate(raw_response_json)
        except Exception:
            verbose_logger.debug(f"Error constructing ResponsesAPIResponse: {raw_response_json}, using model_construct")
            response = ResponsesAPIResponse.model_construct(**raw_response_json)

        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers

        return response
