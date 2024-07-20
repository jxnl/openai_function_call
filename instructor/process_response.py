# type: ignore[all]
from __future__ import annotations

from collections.abc import Iterable
from textwrap import dedent
from instructor.dsl.iterable import IterableBase, IterableModel
from instructor.dsl.parallel import ParallelBase, ParallelModel, handle_parallel_model
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase, ModelAdapter, is_simple_type
from instructor.function_calls import OpenAISchema, openai_schema
from instructor.utils import merge_consecutive_messages
from instructor.decorators import AsyncValidationError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, create_model
import json
import inspect
import logging
from typing import (
    get_args,
    get_origin,
    TypeVar,
    Any,
)
from collections.abc import Generator
from typing_extensions import ParamSpec

from instructor.mode import Mode

from .utils import transform_to_gemini_prompt

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
    """Processes a OpenAI response with the response model, if available.
    It can use `validation_context` and `strict` to validate the response
    via the pydantic model

    Args:
        response (ChatCompletion): The response from OpenAI's API
        response_model (BaseModel): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (bool, optional): Whether to use strict json parsing. Defaults to None.
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

    print(response_model.schema_json())

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    if isinstance(model, OpenAISchema):
        validation_errors = await model.model_async_validate(validation_context)
        if validation_errors:
            raise AsyncValidationError(f"Validation errors: {validation_errors}")

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
    response_model: type[OpenAISchema | BaseModel],
    stream: bool,
    validation_context: dict | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | Generator[T_Model, None, None] | ChatCompletion:
    """Processes a OpenAI response with the response model, if available.

    Args:
        response (T): The response from OpenAI's API
        response_model (Type[T_Model]): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (_type_, optional): Whether to use strict json parsing. Defaults to None.
        mode (Mode, optional): The openai completion mode. Defaults to Mode.FUNCTIONS.

    Returns:
        Union[T_Model, T]: The parsed response, if a response model is available, otherwise the response as is from the SDK
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )

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
    return model


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T], dict[str, Any]]:
    """Prepare the response model type hint, and returns the response_model
    along with the new modified kwargs needed to be able to use the response_model
    parameter with the patch function.


    Args:
        response_model (T): The response model to use for parsing the response
        mode (Mode, optional): The openai completion mode. Defaults to Mode.TOOLS.

    Raises:
        NotImplementedError: When using stream=True with a non-iterable response_model
        ValueError: When using an invalid patch mode

    Returns:
        Union[Type[OpenAISchema], dict]: The response model to use for parsing the response
    """
    new_kwargs = kwargs.copy()
    if response_model is not None:
        # Handles the case where the response_model is a simple type
        # Literal, Annotated, Union, str, int, float, bool, Enum
        # We wrap the response_model in a ModelAdapter that sets 'content' as the response
        if is_simple_type(response_model):
            response_model = ModelAdapter[response_model]

        if is_typed_dict(response_model):
            response_model: BaseModel = create_model(
                response_model.__name__,
                **{k: (v, ...) for k, v in response_model.__annotations__.items()},
            )

        # This a special case for parallel tools
        if mode == Mode.PARALLEL_TOOLS:
            assert (
                new_kwargs.get("stream", False) is False
            ), "stream=True is not supported when using PARALLEL_TOOLS mode"
            new_kwargs["tools"] = handle_parallel_model(response_model)
            new_kwargs["tool_choice"] = "auto"

            # This is a special case for parallel models
            response_model = ParallelModel(typehint=response_model)
            return response_model, new_kwargs

        # This is for all other single model cases
        if get_origin(response_model) is Iterable:
            iterable_element_class = get_args(response_model)[0]
            response_model = IterableModel(iterable_element_class)
        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore

        if new_kwargs.get("stream", False) and not issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            raise NotImplementedError(
                "stream=True is not supported when using response_model parameter for non-iterables"
            )

        if mode == Mode.FUNCTIONS:
            new_kwargs["functions"] = [response_model.openai_schema]
            new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}
        elif mode in {Mode.TOOLS, Mode.MISTRAL_TOOLS}:
            new_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": response_model.openai_schema,
                }
            ]
            if mode == Mode.MISTRAL_TOOLS:
                new_kwargs["tool_choice"] = "any"
            else:
                new_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": response_model.openai_schema["name"]},
                }
        elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
            # If its a JSON Mode we need to massage the prompt a bit
            # in order to get the response we want in a json format
            message = dedent(
                f"""
                As a genius expert, your task is to understand the content and provide
                the parsed objects in json that match the following json_schema:\n

                {json.dumps(response_model.model_json_schema(), indent=2)}

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
                # For some providers, the messages array must be alternating roles of user and assistant, we must merge
                # consecutive user messages into a single message
                new_kwargs["messages"] = merge_consecutive_messages(
                    new_kwargs["messages"]
                )
            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )
            # if it is, system append the schema to the end
            else:
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        elif mode == Mode.ANTHROPIC_TOOLS:
            tool_descriptions = response_model.anthropic_schema
            new_kwargs["tools"] = [tool_descriptions]
            new_kwargs["tool_choice"] = {
                "type": "tool",
                "name": response_model.__name__,
            }

            system_messages = [
                m["content"] for m in new_kwargs["messages"] if m["role"] == "system"
            ]

            if "system" in kwargs and system_messages:
                raise ValueError(
                    "Only a single system message is supported - either set it as a message in the messages array or use the system parameter"
                )

            if not "system" in kwargs:
                new_kwargs["system"] = "\n\n".join(system_messages)

            new_kwargs["messages"] = [
                m for m in new_kwargs["messages"] if m["role"] != "system"
            ]

        elif mode == Mode.ANTHROPIC_JSON:
            # anthropic wants system message to be a string so we first extract out any system message
            openai_system_messages = [
                message["content"]
                for message in new_kwargs.get("messages", [])
                if message["role"] == "system"
            ]

            if "system" in kwargs and openai_system_messages:
                raise ValueError(
                    "Only a single System message is supported - either set it using the system parameter or in the list of messages"
                )

            if not "system" in kwargs:
                new_kwargs["system"] = (
                    new_kwargs.get("system", "")
                    + "\n\n"
                    + "\n\n".join(openai_system_messages)
                )

                new_kwargs["system"] += f"""
                You must only responsd in JSON format that adheres to the following schema:

                <JSON_SCHEMA>
                {json.dumps(response_model.model_json_schema(), indent=2)}
                </JSON_SCHEMA>
                """
                new_kwargs["system"] = dedent(new_kwargs["system"])
            else:
                new_kwargs["system"] += dedent(f"""
                You must only responsd in JSON format that adheres to the following schema:

                <JSON_SCHEMA>
                {json.dumps(response_model.model_json_schema(), indent=2)}
                </JSON_SCHEMA>
                """)

            new_kwargs["messages"] = [
                message
                for message in new_kwargs.get("messages", [])
                if message["role"] != "system"
            ]

            # the messages array must be alternating roles of user and assistant, we must merge
            # consecutive user messages into a single message
            new_kwargs["messages"] = merge_consecutive_messages(new_kwargs["messages"])

        elif mode == Mode.COHERE_TOOLS:
            instruction = f"""\
Extract a valid {response_model.__name__} object based on the chat history and the json schema below.
{response_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {response_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""
            messages = new_kwargs.pop("messages", [])
            chat_history = []
            for message in messages:
                # format in Cohere's ChatMessage format
                chat_history.append(
                    {
                        "role": message["role"],
                        "message": message["content"],
                    }
                )
            new_kwargs["message"] = instruction
            new_kwargs["chat_history"] = chat_history
        elif mode == Mode.GEMINI_JSON:
            assert (
                "model" not in new_kwargs
            ), "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
            message = dedent(
                f"""
                As a genius expert, your task is to understand the content and provide
                the parsed objects in json that match the following json_schema:\n

                {json.dumps(response_model.model_json_schema(), indent=2)}

                Make sure to return an instance of the JSON, not the schema itself
                """
            )
            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )
            # if it is, system append the schema to the end
            else:
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"

            # default to json response type
            new_kwargs["generation_config"] = new_kwargs.get(
                "generation_config", {}
            ) | {"response_mime_type": "application/json"}

            map_openai_args_to_gemini = {
                "max_tokens": "max_output_tokens",
                "temperature": "temperature",
                "n": "candidate_count",
                "top_p": "top_p",
                "stop": "stop_sequences",
            }

            # update gemini config if any params are set
            for k, v in map_openai_args_to_gemini.items():
                val = new_kwargs.pop(k, None)
                if val == None:
                    continue
                new_kwargs["generation_config"][v] = val

            # gemini has a different prompt format and params from other providers
            new_kwargs["contents"] = transform_to_gemini_prompt(
                new_kwargs.pop("messages")
            )

            # minimize gemini safety related errors - model is highly prone to false alarms
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            new_kwargs["safety_settings"] = new_kwargs.get("safety_settings", {}) | {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        elif mode == Mode.VERTEXAI_TOOLS:
            from instructor.client_vertexai import vertexai_process_response

            contents, tools, tool_config = vertexai_process_response(
                new_kwargs, response_model
            )

            new_kwargs["contents"] = contents
            new_kwargs["tools"] = tools
            new_kwargs["tool_config"] = tool_config
        elif mode == Mode.VERTEXAI_JSON:
            from instructor.client_vertexai import vertexai_process_json_response

            contents, generation_config = vertexai_process_json_response(
                new_kwargs, response_model
            )

            new_kwargs["contents"] = contents
            new_kwargs["generation_config"] = generation_config
        else:
            raise ValueError(f"Invalid patch mode: {mode}")

    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": new_kwargs,
        },
    )
    return response_model, new_kwargs
