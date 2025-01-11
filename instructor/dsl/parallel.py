import sys
import json
from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from collections.abc import Generator
from pydantic import BaseModel
from instructor.function_calls import OpenAISchema, openai_schema
from collections.abc import Iterable

from instructor.mode import Mode

T = TypeVar("T", bound=OpenAISchema)

class MissingToolCall(Exception):
    def __init__(self, response: Any, *args: Any) -> None:
        super().__init__(*args)
        self.response = response


class ParallelBase:
    def __init__(self, *models: type[OpenAISchema]):
        # Note that for everything else we've created a class, but for parallel base it is an instance
        assert len(models) > 0, "At least one model is required"
        self.models = models
        self.registry = {
            model.__name__ if hasattr(model, "__name__") else str(model): model
            for model in models
        }

    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[BaseModel, None, None]:
        #! We expect this from the OpenAISchema class, We should address
        #! this with a protocol or an abstract class... @jxnlco
        assert mode == Mode.PARALLEL_TOOLS, "Mode must be PARALLEL_TOOLS"
        if not response.choices[0].message.tool_calls:
            raise MissingToolCall(response)

        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            yield self.registry[name].model_validate_json(
                arguments, context=validation_context, strict=strict
            )


class VertexAIParallelBase(ParallelBase):
    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[BaseModel, None, None]:
        assert (
            mode == Mode.VERTEXAI_PARALLEL_TOOLS
        ), "Mode must be VERTEXAI_PARALLEL_TOOLS"

        if not response or not response.candidates:
            return

        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call is not None:
                    name = part.function_call.name
                    arguments = part.function_call.args

                    if name in self.registry:
                        # Convert dict to JSON string before validation
                        json_str = json.dumps(arguments)
                        yield self.registry[name].model_validate_json(
                            json_str, context=validation_context, strict=strict
                        )


if sys.version_info >= (3, 10):
    from types import UnionType

    def is_union_type(typehint: type[Iterable[T]]) -> bool:
        return get_origin(get_args(typehint)[0]) in (Union, UnionType)
else:

    def is_union_type(typehint: type[Iterable[T]]) -> bool:
        return get_origin(get_args(typehint)[0]) is Union


def get_types_array(typehint: type[Iterable[T]]) -> tuple[type[T], ...]:
    should_be_iterable = get_origin(typehint)

    if should_be_iterable is not Iterable:
        raise TypeError(f"Model should be with Iterable instead of {typehint}")

    if is_union_type(typehint):
        # works for Iterable[Union[int, str]], Iterable[int | str]
        the_types = get_args(get_args(typehint)[0])
        return the_types

    # works for Iterable[int]
    return get_args(typehint)


def handle_parallel_model(typehint: type[Iterable[T]]) -> list[dict[str, Any]]:
    the_types = get_types_array(typehint)
    return [
        {"type": "function", "function": openai_schema(model).openai_schema}
        for model in the_types
    ]


def ParallelModel(typehint: type[Iterable[T]]) -> ParallelBase:
    the_types = get_types_array(typehint)
    return ParallelBase(*[model for model in the_types])


def VertexAIParallelModel(typehint: type[Iterable[T]]) -> VertexAIParallelBase:
    the_types = get_types_array(typehint)
    return VertexAIParallelBase(*[model for model in the_types])
