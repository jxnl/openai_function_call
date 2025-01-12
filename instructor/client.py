from __future__ import annotations

import openai
import inspect
import instructor
from .utils import Provider, get_provider
from openai.types.chat import ChatCompletionMessageParam
from typing import (
    TypeVar,
    Callable,
    overload,
    Union,
    Literal,
    Generic,
    Any,
)
from tenacity import (
    AsyncRetrying,
    Retrying,
)
from collections.abc import Generator, Iterable, Awaitable, AsyncGenerator
from typing_extensions import Self
from pydantic import BaseModel
from instructor.dsl.partial import Partial
from instructor.hooks import Hooks, HookName


T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class Creation(Generic[T]):
    raw: Any # should be uniform completion type
    processed: T

    def __init__(self, raw: Any, processed: T) -> None:
        self.raw, self.processed = raw, processed


class Instructor:
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Creation[T]],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        if mode == instructor.Mode.FUNCTIONS:
            instructor.Mode.warn_mode_functions_deprecation()
        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

    def on(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.on(hook_name, handler)

    def off(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.off(hook_name, handler)

    def clear(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        )
        | None = None,
    ) -> None:
        self.hooks.clear(hook_name)

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    @overload
    def create(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[T]: ...

    @overload
    def create(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self: AsyncInstructor,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    @overload
    def create(
        self: Self,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Any | Awaitable[T] | Awaitable[Any]:
        kwargs = self.handle_kwargs(kwargs)

        creation: Creation[T] = self.create_fn(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed

    @overload
    def create_partial(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_partial(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True

        kwargs = self.handle_kwargs(kwargs)

        response_model = instructor.Partial[response_model]  # type: ignore
        creation: Creation[Generator[T, None, None]] = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed

    @overload
    def create_iterable(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_iterable(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_iterable(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True
        kwargs = self.handle_kwargs(kwargs)

        response_model = Iterable[response_model]  # type: ignore
        creation: Creation[Generator[T, None, None]] = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed

    @overload
    def create_with_completion(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Awaitable[tuple[T, Any]]: ...

    @overload
    def create_with_completion(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    def create_with_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any] | Awaitable[tuple[T, Any]]:
        kwargs = self.handle_kwargs(kwargs)
        creation: Creation[T] = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed, creation.raw

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle and process keyword arguments for the API call.

        This method merges the provided kwargs with the default kwargs stored in the instance.
        It ensures that any kwargs passed to the method call take precedence over the default ones.
        """
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs

    def __getattr__(self, attr: str) -> Any:
        if attr not in {"create", "chat", "messages"}:
            return getattr(self.client, attr)

        return getattr(self, attr)


class AsyncInstructor(Instructor):
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Creation[T]],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

    async def create(  # type: ignore[override]
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T | Any:
        kwargs = self.handle_kwargs(kwargs)
        creation: Creation[T] = await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed

    async def create_partial(  # type: ignore[override]
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True
        creation: Creation[AsyncGenerator[T, None]] = await self.create_fn( # type: ignore 
            response_model=instructor.Partial[response_model],  # type: ignore
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        async for item in creation.processed:
            yield item

    async def create_iterable(  # type: ignore[override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True
        async for creation in await self.create_fn(
            response_model=Iterable[response_model],
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        ):
            creation: Creation[T]
            item = creation.processed
            yield item

    async def create_with_completion(  # type: ignore[override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        kwargs = self.handle_kwargs(kwargs)
        creation: Creation[T] = await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=self.hooks,
            **kwargs,
        )
        return creation.processed, creation.raw


@overload
def from_openai(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor:
    pass


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor:
    pass


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        import warnings

        warnings.warn(
            "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. Unexpected behavior may occur with other client types.",
            stacklevel=2,
        )

    if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.MD_JSON,
        }

    if provider in {Provider.OPENAI, Provider.DATABRICKS}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.FUNCTIONS,
            instructor.Mode.PARALLEL_TOOLS,
            instructor.Mode.MD_JSON,
            instructor.Mode.TOOLS_STRICT,
            instructor.Mode.JSON_O1,
        }

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            mode=mode,
            provider=provider,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            mode=mode,
            provider=provider,
            **kwargs,
        )


@overload
def from_litellm(
    completion: Callable[..., Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_litellm(
    completion: Awaitable[Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor:
    pass


def from_litellm(
    completion: Callable[..., Any] | Awaitable[Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    is_async = inspect.iscoroutinefunction(completion)

    if not is_async:
        return Instructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )
