from typing import Iterable, Literal, Union
from pydantic import BaseModel

import pytest
import instructor


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


def test_sync_parallel_tools(client):
    client = instructor.patch(client, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3


@pytest.mark.asyncio
async def test_async_parallel_tools(aclient):
    client = instructor.patch(aclient, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3
