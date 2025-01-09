from __future__ import annotations

from typing import Any, overload
import boto3
from botocore.client import BaseClient
import instructor
from instructor.client import AsyncInstructor, Instructor


@overload
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def handle_bedrock_json(
    response_model: Any,
    new_kwargs: Any,
) -> tuple[Any, Any]:
    print(f"handle_bedrock_json: response_model {response_model}")
    print(f"handle_bedrock_json: new_kwargs {new_kwargs}")
    return response_model, new_kwargs


def from_bedrock(
    client: BaseClient,
    mode: instructor.Mode = instructor.Mode.BEDROCK_JSON,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert mode in {
        instructor.Mode.BEDROCK_TOOLS,
        instructor.Mode.BEDROCK_JSON,
    }, "Mode must be one of {instructor.Mode.BEDROCK_TOOLS, instructor.Mode.BEDROCK_JSON}"
    assert isinstance(
        client,
        BaseClient,
    ), "Client must be an instance of boto3.client"
    create = client.converse  # Example method, replace with actual method

    return Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.BEDROCK,
        mode=mode,
        **kwargs,
    )
