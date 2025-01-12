from __future__ import annotations
from functools import wraps
from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
    overload,
)
from collections.abc import Awaitable
from typing_extensions import ParamSpec

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from instructor.client import Creation
from instructor.process_response import handle_response_model
from instructor.retry import retry_async, retry_sync
from instructor.utils import is_async
from instructor.hooks import Hooks
from instructor.templating import handle_templating

from instructor.mode import Mode
import logging

from tenacity import (
    AsyncRetrying,
    Retrying,
)

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> Creation[T_Model]: ...


class AsyncInstructorChatCompletionCreate(Protocol):
    async def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> Creation[T_Model]: ...


def handle_context(
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Handle the context and validation_context parameters.
    If both are provided, raise an error.
    If validation_context is provided, issue a deprecation warning and use it as context.
    If neither is provided, return None.
    """
    if context is not None and validation_context is not None:
        raise ValueError(
            "Cannot provide both 'context' and 'validation_context'. Use 'context' instead."
        )
    if validation_context is not None and context is None:
        import warnings

        warnings.warn(
            "'validation_context' is deprecated. Use 'context' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        context = validation_context
    return context


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.TOOLS,
) -> OpenAI: ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
) -> AsyncOpenAI: ...


@overload
def patch(
    create: Callable[T_ParamSpec, T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


@overload
def patch(
    create: Awaitable[T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


def patch(  # type: ignore
    client: OpenAI | AsyncOpenAI | None = None,
    create: Callable[T_ParamSpec, T_Retval] | None = None,
    mode: Mode = Mode.TOOLS,
) -> OpenAI | AsyncOpenAI:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    - `hooks` parameter to hook into the completion process
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    func_is_async = is_async(func)

    @wraps(func)  # type: ignore
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> Creation[T_Model]:
        context = handle_context(context, validation_context)

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )  # type: ignore
        new_kwargs = handle_templating(new_kwargs, context)

        response = await retry_async(
            func=func,  # type:ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            mode=mode,
            hooks=hooks,
        )
        return response  # type: ignore

    @wraps(func)  # type: ignore
    def new_create_sync(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> Creation[T_Model]:
        context = handle_context(context, validation_context)

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )  # type: ignore

        new_kwargs = handle_templating(new_kwargs, context)

        response = retry_sync(
            func=func,  # type: ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            hooks=hooks,
            strict=strict,
            kwargs=new_kwargs,
            mode=mode,
        )
        return response  # type: ignore

    new_create = new_create_async if func_is_async else new_create_sync

    if client is not None:
        client.chat.completions.create = new_create  # type: ignore
        return client
    else:
        return new_create  # type: ignore


def apatch(client: AsyncOpenAI, mode: Mode = Mode.TOOLS) -> AsyncOpenAI:
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead", DeprecationWarning, stacklevel=2
    )
    return patch(client, mode=mode)
