# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload, TypeVar

from mistralai.client import MistralClient

import instructor
from instructor.mode import Mode
from instructor.utils import Provider

T = TypeVar("T")

@overload
def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.AsyncInstructor | instructor.Instructor:
    """Create a patched Mistral client."""
    assert mode in {
        Mode.MISTRAL_TOOLS,
        Mode.MISTRAL_JSON,
    }, f"Mode must be one of {Mode.MISTRAL_TOOLS}, {Mode.MISTRAL_JSON}"

    assert isinstance(
        client, MistralClient
    ), "Client must be an instance of mistralai.MistralClient"

    if use_async:
        create = client.chat.create_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )

    create = client.chat.create
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=Provider.MISTRAL,
        mode=mode,
        **kwargs,
    )
