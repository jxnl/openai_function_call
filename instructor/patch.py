import inspect
import json
import logging
from collections.abc import Iterable
from functools import wraps
from json import JSONDecodeError
from typing import Callable, Optional, Type, TypeVar, Union, get_args, get_origin

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from instructor.dsl.iterable import IterableModel, IterableBase
from instructor.dsl.parallel import ParallelBase, ParallelModel, handle_parallel_model
from instructor.dsl.partial import PartialBase

from .function_calls import Mode, OpenAISchema, openai_schema

logger = logging.getLogger("instructor")
T = TypeVar("T")

OVERRIDE_DOCS = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is. 

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model. The `_raw_response.usage` attribute is modified to reflect the token usage from the last successful response as well as from any previous unsuccessful attempts.

Parameters:
    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
    max_retries (int): The maximum number of retries to attempt if the response is not valid (default: 0)
    validation_context (dict): The validation context to use for validating the response (default: None)
"""


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
        ret["content"] += json.dumps(message.model_dump()["tool_calls"])
    if hasattr(message, "function_call") and message.function_call is not None:
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def handle_response_model(
    *,
    response_model: Type[BaseModel],
    kwargs,
    mode: Mode = Mode.FUNCTIONS,
):
    new_kwargs = kwargs.copy()
    if response_model is not None:
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
            new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
            new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}  # type: ignore
        elif mode == Mode.TOOLS:
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
        elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
            # If its a JSON Mode we need to massage the prompt a bit
            # in order to get the response we want in a json format
            message = f"""
                As a genius expert, your task is to understand the content and provide 
                the parsed objects in json that match the following json_schema:\n
                {response_model.model_json_schema()['properties']}
                """
            # Check for nested models
            if "$defs" in response_model.model_json_schema():
                message += f"\nHere are some more definitions to adhere too:\n{response_model.model_json_schema()['$defs']}"

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
                        "role": "assistant",
                        "content": "Here is the perfectly correctly formatted JSON\n```json",
                    },
                )
                new_kwargs["stop"] = "```"
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

            # if the first message is a system append the schema to the end
            if new_kwargs["messages"][0]["role"] == "system":
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        else:
            raise ValueError(f"Invalid patch mode: {mode}")
    return response_model, new_kwargs


def process_response(
    response,
    *,
    response_model: Type[BaseModel],
    stream: bool,
    validation_context: dict = None,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
):  # type: ignore
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
    if response_model is None:
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
    if isinstance(response_model, IterableBase):
        #! If the response model is a multitask, return the tasks
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


async def process_response_async(
    response,
    *,
    response_model: Type[BaseModel],
    stream: bool,
    validation_context: dict = None,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
):  # type: ignore
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
    if isinstance(response_model, IterableBase):
        #! If the response model is a multitask, return the tasks
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


async def retry_async(
    func,
    response_model,
    validation_context,
    args,
    kwargs,
    max_retries,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    retries = 0
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    while retries <= max_retries:
        try:
            response: ChatCompletion = await func(*args, **kwargs)
            stream = kwargs.get("stream", False)
            if isinstance(response, ChatCompletion) and response.usage is not None:
                total_usage.completion_tokens += response.usage.completion_tokens or 0
                total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                total_usage.total_tokens += response.usage.total_tokens or 0
                response.usage = (
                    total_usage  # Replace each response usage with the total usage
                )
            return await process_response_async(
                response,
                response_model=response_model,
                stream=stream,
                validation_context=validation_context,
                strict=strict,
                mode=mode,
            )
        except (ValidationError, JSONDecodeError) as e:
            logger.exception(f"Retrying, exception: {e}")
            logger.debug(f"Error response: {response}")
            if mode == Mode.TOOLS:
                kwargs["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": response.choices[0].message.tool_calls[0].id,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                        "content": "failure",
                    }
                )
            kwargs["messages"].append(dump_message(response.choices[0].message))  # type: ignore
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, fix the errors, exceptions found\n{e}",
                }
            )
            if mode == Mode.MD_JSON:
                kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": "```json",
                    },
                )
            retries += 1
            if retries > max_retries:
                raise e


def retry_sync(
    func,
    response_model,
    validation_context,
    args,
    kwargs,
    max_retries,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    retries = 0
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    while retries <= max_retries:
        # Excepts ValidationError, and JSONDecodeError
        try:
            response = func(*args, **kwargs)
            stream = kwargs.get("stream", False)
            if isinstance(response, ChatCompletion) and response.usage is not None:
                total_usage.completion_tokens += response.usage.completion_tokens or 0
                total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                total_usage.total_tokens += response.usage.total_tokens or 0
                response.usage = (
                    total_usage  # Replace each response usage with the total usage
                )
            return process_response(
                response,
                response_model=response_model,
                stream=stream,
                validation_context=validation_context,
                strict=strict,
                mode=mode,
            )
        except (ValidationError, JSONDecodeError) as e:
            logger.exception(f"Retrying, exception: {e}")
            logger.debug(f"Error response: {response}")
            if mode == Mode.TOOLS:
                kwargs["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": response.choices[0].message.tool_calls[0].id,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                        "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                    }
                )
            kwargs["messages"].append(dump_message(response.choices[0].message))
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                }
            )
            if mode == Mode.MD_JSON:
                kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": "```json",
                    },
                )
            retries += 1
            if retries > max_retries:
                logger.warning(f"Max retries reached, exception: {e}")
                raise e


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
        hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )


def wrap_chatcompletion(func: Callable, mode: Mode = Mode.FUNCTIONS) -> Callable:
    func_is_async = is_async(func)

    @wraps(func)
    async def new_chatcompletion_async(
        response_model=None,
        validation_context=None,
        max_retries=1,
        *args,
        **kwargs,
    ):
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, kwargs=kwargs, mode=mode
        )  # type: ignore
        response = await retry_async(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    @wraps(func)
    def new_chatcompletion_sync(
        response_model=None,
        validation_context=None,
        max_retries=1,
        *args,
        **kwargs,
    ):
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, kwargs=kwargs, mode=mode
        )  # type: ignore
        response = retry_sync(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    wrapper_function = (
        new_chatcompletion_async if func_is_async else new_chatcompletion_sync
    )
    wrapper_function.__doc__ = OVERRIDE_DOCS
    return wrapper_function


def patch(client: Union[OpenAI, AsyncOpenAI], mode: Mode = Mode.FUNCTIONS):
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")
    client.chat.completions.create = wrap_chatcompletion(
        client.chat.completions.create, mode=mode
    )
    return client


def apatch(client: AsyncOpenAI, mode: Mode = Mode.FUNCTIONS):
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    return patch(client, mode=mode)
