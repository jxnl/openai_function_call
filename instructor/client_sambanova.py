from __future__ import annotations

from typing import overload, Any

import openai
import instructor


@overload
def from_sambanova(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...

def from_sambanova(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor:
    assert mode in {
        instructor.Mode.TOOLS,
    }, "Mode be one of { instructor.Mode.TOOLS }"

    assert isinstance(
        client, (openai.OpenAI)
    ), "Client must be an instance of openai.OpenAI"

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=client.chat.completions.create, mode=mode),
        provider=instructor.Provider.SAMBANOVA,
        mode=mode,
        **kwargs,
    )
