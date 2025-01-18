from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    Union,
    TypedDict,
    TypeVar,
    cast,
)
from pydantic import BaseModel
import os

from openai.types import CompletionUsage as OpenAIUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage


logger = logging.getLogger("instructor")
R_co = TypeVar("R_co", covariant=True)
T_Model = TypeVar("T_Model", bound="Response")

from enum import Enum


class Response(Protocol):
    usage: OpenAIUsage | AnthropicUsage


class Provider(Enum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    DATABRICKS = "databricks"
    CEREBRAS = "cerebras"
    FIREWORKS = "fireworks"
    WRITER = "writer"
    UNKNOWN = "unknown"


def get_provider(base_url: str) -> Provider:
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "cerebras" in str(base_url):
        return Provider.CEREBRAS
    elif "fireworks" in str(base_url):
        return Provider.FIREWORKS
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    elif "writer" in str(base_url):
        return Provider.WRITER
    return Provider.UNKNOWN


def extract_json_from_codeblock(content: str) -> str:
    first_paren = content.find("{")
    last_paren = content.rfind("}")
    return content[first_paren : last_paren + 1]


def extract_json_from_stream(chunks: Iterable[str]) -> Generator[str, None, None]:
    capturing = False
    brace_count = 0
    for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    capturing = False
    brace_count = 0
    async for chunk in chunks:
        for char in chunk:
            if char == "{":
                capturing = True
                brace_count += 1
                yield char
            elif char == "}" and capturing:
                brace_count -= 1
                yield char
                if brace_count == 0:
                    capturing = False
                    break  # Cease yielding upon closing the current JSON object
            elif capturing:
                yield char


def update_total_usage(
    response: T_Model | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Model | ChatCompletion | None:
    if response is None:
        return None

    response_usage = getattr(response, "usage", None)
    if isinstance(response_usage, OpenAIUsage) and isinstance(total_usage, OpenAIUsage):
        total_usage.completion_tokens += response_usage.completion_tokens or 0
        total_usage.prompt_tokens += response_usage.prompt_tokens or 0
        total_usage.total_tokens += response_usage.total_tokens or 0
        if (rtd := response_usage.completion_tokens_details) and (
            ttd := total_usage.completion_tokens_details
        ):
            ttd.audio_tokens = (ttd.audio_tokens or 0) + (rtd.audio_tokens or 0)
            ttd.reasoning_tokens = (ttd.reasoning_tokens or 0) + (
                rtd.reasoning_tokens or 0
            )
            ttd.accepted_prediction_tokens = (ttd.accepted_prediction_tokens or 0) + (rtd.accepted_prediction_tokens or 0)
            ttd.rejected_prediction_tokens = (ttd.rejected_prediction_tokens or 0) + (rtd.rejected_prediction_tokens or 0)
        if (rpd := response_usage.prompt_tokens_details) and (
            tpd := total_usage.prompt_tokens_details
        ):
            tpd.audio_tokens = (tpd.audio_tokens or 0) + (rpd.audio_tokens or 0)
            tpd.cached_tokens = (tpd.cached_tokens or 0) + (rpd.cached_tokens or 0)
        response.usage = total_usage  # Replace each response usage with the total usage
        return response

    # Anthropic usage.
    try:
        from anthropic.types import Usage as AnthropicUsage

        if isinstance(response_usage, AnthropicUsage) and isinstance(
            total_usage, AnthropicUsage
        ):
            if not total_usage.cache_creation_input_tokens:
                total_usage.cache_creation_input_tokens = 0

            if not total_usage.cache_read_input_tokens:
                total_usage.cache_read_input_tokens = 0

            total_usage.input_tokens += response_usage.input_tokens or 0
            total_usage.output_tokens += response_usage.output_tokens or 0
            total_usage.cache_creation_input_tokens += (
                response_usage.cache_creation_input_tokens or 0
            )
            total_usage.cache_read_input_tokens += (
                response_usage.cache_read_input_tokens or 0
            )
            response.usage = total_usage
            return response
    except ImportError:
        pass

    logger.debug("No compatible response.usage found, token usage not updated.")
    return response


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if (
        hasattr(message, "function_call")
        and message.function_call is not None
        and ret["content"]
    ):
        if not isinstance(ret["content"], str):
            response_message: str = ""
            for content_message in ret["content"]:
                if "text" in content_message:
                    response_message += content_message["text"]
                elif "refusal" in content_message:
                    response_message += content_message["refusal"]
            ret["content"] = response_message
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def is_async(func: Callable[..., Any]) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(func)
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__  # type: ignore - dynamic
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(func)
    return is_coroutine


def merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # merge all consecutive user messages into a single message
    new_messages: list[dict[str, Any]] = []
    # Detect whether all messages have a flat content (i.e. all string)
    # Some providers require content to be a string, so we need to check that and behave accordingly
    flat_string = all(isinstance(m["content"], str) for m in messages)
    for message in messages:
        new_content = message["content"]
        if not flat_string and isinstance(new_content, str):
            # If content is not flat, transform it into a list of text
            new_content = [{"type": "text", "text": new_content}]

        if len(new_messages) > 0 and message["role"] == new_messages[-1]["role"]:
            if flat_string:
                # New content is a string
                new_messages[-1]["content"] += f"\n\n{new_content}"
            else:
                # New content is a list
                new_messages[-1]["content"].extend(new_content)
        else:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": new_content,
                }
            )

    return new_messages


class classproperty(Generic[R_co]):
    """Descriptor for class-level properties.

    Examples:
        >>> from instructor.utils import classproperty

        >>> class MyClass:
        ...     @classproperty
        ...     def my_property(cls):
        ...         return cls

        >>> assert MyClass.my_property
    """

    def __init__(self, method: Callable[[Any], R_co]) -> None:
        self.cproperty = method

    def __get__(self, instance: object, cls: type[Any]) -> R_co:
        return self.cproperty(cls)


def get_message_content(message: ChatCompletionMessageParam) -> list[Any]:
    content = message.get("content", "")
    try:
        if isinstance(content, list):
            return content
        else:
            return [content]
    except Exception as e:
        logging.debug(f"Error getting message content: {e}")
        return [content]


def transform_to_gemini_prompt(
    messages_chatgpt: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    messages_gemini: list[dict[str, Any]] = []
    system_prompt = ""
    for message in messages_chatgpt:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            messages_gemini.append(
                {"role": "user", "parts": get_message_content(message)}
            )
        elif message["role"] == "assistant":
            messages_gemini.append(
                {"role": "model", "parts": get_message_content(message)}
            )

    if system_prompt:
        if messages_gemini:
            messages_gemini[0]["parts"].insert(0, f"*{system_prompt}*")
        else:
            messages_gemini.append({"role": "user", "parts": [f"*{system_prompt}*"]})

    return messages_gemini


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Map OpenAPI schema to Gemini properties: gemini function call schemas

    Ref - https://ai.google.dev/api/python/google/generativeai/protos/Schema,
    Note that `enum` requires specific `format` setting
    """

    import jsonref

    class FunctionSchema(BaseModel):
        description: str | None = None
        enum: list[str] | None = None
        example: Any | None = None
        format: str | None = None
        nullable: bool | None = None
        items: FunctionSchema | None = None
        required: list[str] | None = None
        type: str
        properties: dict[str, FunctionSchema] | None = None

    schema: dict[str, Any] = jsonref.replace_refs(obj, lazy_load=False)  # type: ignore
    schema.pop("$defs", "")

    def add_enum_format(obj: dict[str, Any]) -> dict[str, Any]:
        if isinstance(obj, dict):
            new_dict: dict[str, Any] = {}
            for key, value in obj.items():
                new_dict[key] = add_enum_format(value)
                if key == "enum":
                    new_dict["format"] = "enum"
            return new_dict
        else:
            return obj

    schema = add_enum_format(schema)

    return FunctionSchema(**schema).model_dump(exclude_none=True, exclude_unset=True)


def update_gemini_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "generation_config" in kwargs:
        map_openai_args_to_gemini = {
            "max_tokens": "max_output_tokens",
            "temperature": "temperature",
            "n": "candidate_count",
            "top_p": "top_p",
            "stop": "stop_sequences",
        }

        # update gemini config if any params are set
        for k, v in map_openai_args_to_gemini.items():
            val = kwargs["generation_config"].pop(k, None)
            if val == None:
                continue
            kwargs["generation_config"][v] = val

    # gemini has a different prompt format and params from other providers
    kwargs["contents"] = transform_to_gemini_prompt(kwargs.pop("messages"))

    # minimize gemini safety related errors - model is highly prone to false alarms
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore

    kwargs["safety_settings"] = kwargs.get("safety_settings", {})

    fallback_safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # Update or add fallback settings, respecting stricter existing ones
    for category, fallback_threshold in fallback_safety_settings.items():
        current_threshold = kwargs["safety_settings"].get(category)
        if current_threshold is None or current_threshold < fallback_threshold:
            kwargs["safety_settings"][category] = fallback_threshold

    return kwargs


def disable_pydantic_error_url():
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: Union[str, list[SystemMessage], None],  # noqa: UP007
    new_system: Union[str, list[SystemMessage]],  # noqa: UP007
) -> Union[str, list[SystemMessage]]:  # noqa: UP007
    if existing_system is None:
        return new_system

    if isinstance(existing_system, str) and isinstance(new_system, str):
        return f"{existing_system}\n\n{new_system}"

    if isinstance(existing_system, list) and isinstance(new_system, list):
        return existing_system + new_system

    if isinstance(existing_system, str) and isinstance(new_system, list):
        return [SystemMessage(type="text", text=existing_system)] + new_system

    if isinstance(existing_system, list) and isinstance(new_system, str):
        return existing_system + [SystemMessage(type="text", text=new_system)]

    raise ValueError("Unsupported system message type combination")


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    def convert_message(content: Union[str, dict[str, Any]]) -> SystemMessage:  # noqa: UP007
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        elif isinstance(content, dict):
            return SystemMessage(**content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    result: list[SystemMessage] = []
    for m in messages:
        if m["role"] == "system":
            # System message must always be a string or list of dictionaries
            content = cast(Union[str, list[dict[str, Any]]], m["content"])  # noqa: UP007
            if isinstance(content, list):
                result.extend(convert_message(item) for item in content)
            else:
                result.append(convert_message(content))
    return result
