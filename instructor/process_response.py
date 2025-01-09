# type: ignore[all]
from __future__ import annotations

from collections.abc import Iterable
from textwrap import dedent
from typing import Any, TypeVar, get_args, get_origin
from typing_extensions import ParamSpec

import inspect
import json
import logging

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, create_model

# from instructor.client_bedrock import handle_bedrock_json
from instructor.mode import Mode
from instructor.dsl.iterable import IterableBase, IterableModel
from instructor.dsl.parallel import (
    ParallelBase,
    ParallelModel,
    handle_parallel_model,
    get_types_array,
    VertexAIParallelBase,
    VertexAIParallelModel,
)
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import (
    AdapterBase,
    ModelAdapter,
    is_simple_type,
)
from instructor.function_calls import OpenAISchema, openai_schema
from instructor.utils import (
    merge_consecutive_messages,
    extract_system_messages,
    combine_system_messages,
)
from instructor.multimodal import convert_messages

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """
    Asynchronously processes the response from the OpenAI API.

    Args:
        response (ChatCompletion): The raw response from the OpenAI API.
        response_model (type[T_Model | OpenAISchema | BaseModel] | None): The expected model type for the response.
        stream (bool): Whether the response is streamed.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to apply strict validation.
        mode (Mode): The processing mode to use.

    Returns:
        T_Model | ChatCompletion: The processed response, either as the specified model type or the raw ChatCompletion.

    This function handles various response types, including streaming responses and different model bases.
    It applies the appropriate processing based on the response_model and mode provided.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | VertexAIParallelBase | None:
    """
    Process the response from the API call and convert it to the specified response model.

    Args:
        response (T_Model): The raw response from the API call.
        response_model (type[OpenAISchema | BaseModel] | None): The model to convert the response to.
        stream (bool): Whether the response is a streaming response.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to use strict validation.
        mode (Mode): The mode used for processing the response.

    Returns:
        The processed response, which could be:
        - The raw response if no response_model is specified
        - An instance of the response_model
        - A list of tasks if the model is an IterableBase
        - The content of the model if it's an AdapterBase

    This function handles various types of responses and models, including streaming
    responses, iterable models, parallel models, and adapter models. It also attaches
    the raw response to the processed model when applicable.
    """
    logger.debug(
        f"Instructor Raw Response: {response}",
    )

    # TODO: remove this
    print(f"instructor.process_response.py: response_model {response_model}")

    # TODO: remove this
    print(f"instructor.process_response.py: response {response}")

    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response

    # TODO: remove this
    print(f"instructor.process_response.py: model {model}")

    return model


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def handle_parallel_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    assert (
        new_kwargs.get("stream", False) is False
    ), "stream=True is not supported when using PARALLEL_TOOLS mode"
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return ParallelModel(typehint=response_model), new_kwargs


def handle_functions(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    Mode.warn_mode_functions_deprecation()
    new_kwargs["functions"] = [response_model.openai_schema]
    new_kwargs["function_call"] = {
        "name": response_model.openai_schema["name"]
    }
    return response_model, new_kwargs


def handle_tools_strict(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    response_model_schema = pydantic_function_tool(response_model)
    response_model_schema["function"]["strict"] = True
    new_kwargs["tools"] = [response_model_schema]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model_schema["function"]["name"]},
    }
    return response_model, new_kwargs


def handle_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_mistral_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = "any"
    return response_model, new_kwargs


def handle_json_o1(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    roles = [message["role"] for message in new_kwargs.get("messages", [])]
    if "system" in roles:
        raise ValueError("System messages are not supported For the O1 models")

    message = dedent(
        f"""
        Understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["messages"].append(
        {
            "role": "user",
            "content": message,
        },
    )
    return response_model, new_kwargs


def handle_json_modes(
    response_model: type[T], new_kwargs: dict[str, Any], mode: Mode
) -> tuple[type[T], dict[str, Any]]:
    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if mode == Mode.JSON:
        new_kwargs["response_format"] = {"type": "json_object"}
    elif mode == Mode.JSON_SCHEMA:
        new_kwargs["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema(),
        }
    elif mode == Mode.MD_JSON:
        new_kwargs["messages"].append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(
            new_kwargs["messages"]
        )

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(
            0,
            {
                "role": "system",
                "content": message,
            },
        )
    elif isinstance(new_kwargs["messages"][0]["content"], str):
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"
    elif isinstance(new_kwargs["messages"][0]["content"], list):
        new_kwargs["messages"][0]["content"][0]["text"] += f"\n\n{message}"
    else:
        raise ValueError(
            "Invalid message format, must be a string or a list of messages"
        )

    return response_model, new_kwargs


def handle_anthropic_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    tool_descriptions = response_model.anthropic_schema
    new_kwargs["tools"] = [tool_descriptions]
    new_kwargs["tool_choice"] = {
        "type": "tool",
        "name": response_model.__name__,
    }

    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    return response_model, new_kwargs


def handle_anthropic_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    json_schema_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["system"] = combine_system_messages(
        new_kwargs.get("system"),
        [{"type": "text", "text": json_schema_message}],
    )

    return response_model, new_kwargs


def handle_cohere_modes(
    new_kwargs: dict[str, Any]
) -> tuple[None, dict[str, Any]]:
    messages = new_kwargs.pop("messages", [])
    chat_history = []
    for message in messages[:-1]:
        chat_history.append(  # type: ignore
            {
                "role": message["role"],
                "message": message["content"],
            }
        )
    new_kwargs["message"] = messages[-1]["content"]
    new_kwargs["chat_history"] = chat_history
    if "model_name" in new_kwargs and "model" not in new_kwargs:
        new_kwargs["model"] = new_kwargs.pop("model_name")
    new_kwargs.pop("strict", None)
    return None, new_kwargs


def handle_fireworks_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_fireworks_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False

    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    return response_model, new_kwargs


def handle_gemini_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    assert (
        "model" not in new_kwargs
    ), "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"

    from .utils import update_gemini_kwargs

    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(
            0, {"role": "system", "content": message}
        )
    else:
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"

    new_kwargs["generation_config"] = new_kwargs.get(
        "generation_config", {}
    ) | {"response_mime_type": "application/json"}

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_gemini_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    assert (
        "model" not in new_kwargs
    ), "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"

    from .utils import update_gemini_kwargs

    new_kwargs["tools"] = [response_model.gemini_schema]
    new_kwargs["tool_config"] = {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [response_model.__name__],
        },
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_vertexai_parallel_tools(
    response_model: type[Iterable[T]], new_kwargs: dict[str, Any]
) -> tuple[VertexAIParallelBase, dict[str, Any]]:
    assert (
        new_kwargs.get("stream", False) is False
    ), "stream=True is not supported when using PARALLEL_TOOLS mode"

    from instructor.client_vertexai import vertexai_process_response

    # Extract concrete types before passing to vertexai_process_response
    model_types = list(get_types_array(response_model))
    contents, tools, tool_config = vertexai_process_response(
        new_kwargs, model_types
    )

    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config

    return VertexAIParallelModel(typehint=response_model), new_kwargs


def handle_vertexai_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from instructor.client_vertexai import vertexai_process_response

    contents, tools, tool_config = vertexai_process_response(
        new_kwargs, response_model
    )

    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config
    return response_model, new_kwargs


def handle_vertexai_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from instructor.client_vertexai import vertexai_process_json_response

    contents, generation_config = vertexai_process_json_response(
        new_kwargs, response_model
    )

    new_kwargs["contents"] = contents
    new_kwargs["generation_config"] = generation_config
    return response_model, new_kwargs


def handle_bedrock_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    print(f"handle_bedrock_json: response_model {response_model}")
    print(f"handle_bedrock_json: new_kwargs {new_kwargs}")
    json_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself 
        and don't include any other text in the response apart from the json
        """
    )
    system_message = new_kwargs.pop("system", None)
    if not system_message:
        new_kwargs["system"] = [{"text": json_message}]
    else:
        if not isinstance(system_message, list):
            raise ValueError(
                """system must be a list of SystemMessage refer 
                https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
                """
            )
        system_message.append({"text": json_message})
    return response_model, new_kwargs


def handle_cohere_json_schema(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    _, new_kwargs = handle_cohere_modes(new_kwargs)

    return response_model, new_kwargs


def handle_cerebras_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if new_kwargs.get("stream", False):
        raise ValueError("Stream is not supported for Cerebras Tool Calling")
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_cerebras_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    instruction = f"""
You are a helpful assistant that excels at following instructions.Your task is to understand the content and provide the parsed objects in json that match the following json_schema:\n

Here is the relevant JSON schema to adhere to

<schema>
{response_model.model_json_schema()}
</schema>

Your response should consist only of a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""

    new_kwargs["messages"] = [
        {"role": "system", "content": instruction}
    ] + new_kwargs["messages"]
    return response_model, new_kwargs


def handle_cohere_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    _, new_kwargs = handle_cohere_modes(new_kwargs)

    instruction = f"""\
Extract a valid {response_model.__name__} object based on the chat history and the json schema below.
{response_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {response_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""
    new_kwargs["chat_history"] = [
        {"role": "user", "message": instruction}
    ] + new_kwargs["chat_history"]
    return response_model, new_kwargs


def handle_writer_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = "auto"
    return response_model, new_kwargs


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def prepare_response_model(response_model: type[T] | None) -> type[T] | None:
    """
    Prepares the response model for use in the API call.

    This function performs several transformations on the input response_model:
    1. If the response_model is None, it returns None.
    2. If it's a simple type, it wraps it in a ModelAdapter.
    3. If it's a TypedDict, it converts it to a Pydantic BaseModel.
    4. If it's an Iterable, it wraps the element type in an IterableModel.
    5. If it's not already a subclass of OpenAISchema, it applies the openai_schema decorator.

    Args:
        response_model (type[T] | None): The input response model to be prepared.

    Returns:
        type[T] | None: The prepared response model, or None if the input was None.
    """
    if response_model is None:
        return None

    if is_simple_type(response_model):
        response_model = ModelAdapter[response_model]

    if is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    if get_origin(response_model) is Iterable:
        iterable_element_class = get_args(response_model)[0]
        response_model = IterableModel(iterable_element_class)

    if not issubclass(response_model, OpenAISchema):
        response_model = openai_schema(response_model)  # type: ignore

    return response_model


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T] | VertexAIParallelBase | None, dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.

    Args:
        response_model (type[T] | None): The response model to be used for parsing the API response.
        mode (Mode): The mode to use for handling the response model. Defaults to Mode.TOOLS.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        tuple[type[T] | None, dict[str, Any]]: A tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """

    new_kwargs = kwargs.copy()
    print(f"instructor.process_response.py: new_kwargs -> {new_kwargs}")
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    if response_model is None:
        if mode in {Mode.COHERE_JSON_SCHEMA, Mode.COHERE_TOOLS}:
            # This is cause cohere uses 'message' and 'chat_history' instead of 'messages'
            return handle_cohere_modes(new_kwargs)
        # Handle images without a response model
        if "messages" in new_kwargs:
            messages = convert_messages(
                new_kwargs["messages"],
                mode,
                autodetect_images=autodetect_images,
            )
            if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
                # Handle OpenAI style or Anthropic style messages
                new_kwargs["messages"] = [
                    m for m in messages if m["role"] != "system"
                ]
                if "system" not in new_kwargs:
                    system_message = extract_system_messages(messages)
                    if system_message:
                        new_kwargs["system"] = system_message
            else:
                new_kwargs["messages"] = messages
        return None, new_kwargs

    if mode in {Mode.PARALLEL_TOOLS}:
        return handle_parallel_tools(response_model, new_kwargs)
    elif mode in {Mode.VERTEXAI_PARALLEL_TOOLS}:
        return handle_vertexai_parallel_tools(response_model, new_kwargs)

    response_model = prepare_response_model(response_model)

    mode_handlers = {  # type: ignore
        Mode.FUNCTIONS: handle_functions,
        Mode.TOOLS_STRICT: handle_tools_strict,
        Mode.TOOLS: handle_tools,
        Mode.MISTRAL_TOOLS: handle_mistral_tools,
        Mode.JSON_O1: handle_json_o1,
        Mode.JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON),  # type: ignore
        Mode.MD_JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.MD_JSON),  # type: ignore
        Mode.JSON_SCHEMA: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON_SCHEMA),  # type: ignore
        Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
        Mode.ANTHROPIC_JSON: handle_anthropic_json,
        Mode.COHERE_JSON_SCHEMA: handle_cohere_json_schema,
        Mode.COHERE_TOOLS: handle_cohere_tools,
        Mode.GEMINI_JSON: handle_gemini_json,
        Mode.GEMINI_TOOLS: handle_gemini_tools,
        Mode.VERTEXAI_TOOLS: handle_vertexai_tools,
        Mode.VERTEXAI_JSON: handle_vertexai_json,
        Mode.CEREBRAS_JSON: handle_cerebras_json,
        Mode.CEREBRAS_TOOLS: handle_cerebras_tools,
        Mode.FIREWORKS_JSON: handle_fireworks_json,
        Mode.FIREWORKS_TOOLS: handle_fireworks_tools,
        Mode.WRITER_TOOLS: handle_writer_tools,
        Mode.BEDROCK_JSON: handle_bedrock_json,
    }

    if mode in mode_handlers:
        response_model, new_kwargs = mode_handlers[mode](
            response_model, new_kwargs
        )
    else:
        raise ValueError(f"Invalid patch mode: {mode}")

    if "messages" in new_kwargs:
        new_kwargs["messages"] = convert_messages(
            new_kwargs["messages"],
            mode,
            autodetect_images=autodetect_images,
        )
    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None
                and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": new_kwargs,
        },
    )
    return response_model, new_kwargs
