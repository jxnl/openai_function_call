# type: ignore[all]
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError, validator
from typing import Optional, Union, Literal, Annotated
from instructor.dsl.partial import Partial, PartialLiteralMixin
import pytest
import instructor
from openai import OpenAI, AsyncOpenAI

models = ["gpt-4o-mini"]
modes = [
    instructor.Mode.TOOLS,
]


class SampleNestedPartial(BaseModel):
    b: int


class SamplePartial(BaseModel):
    a: int
    b: SampleNestedPartial


class NestedA(BaseModel):
    a: str
    b: Optional[str]
    c: Optional[Annotated[datetime, Partial]]


class NestedB(BaseModel):
    c: str
    d: str
    e: list[Union[str, int]]
    f: str


class UnionWithNested(BaseModel):
    a: list[Union[NestedA, NestedB]]
    b: list[NestedA]
    c: NestedB


SampleEnum = Literal["a_value", "b_value", "c_value"]
SampleMixedEnum = Literal["a_value", "b_value", "c_value", 1, 2, 3]


class PartialEnums(BaseModel):
    a: Annotated[Literal["a_value"], Partial]
    b: Annotated[SampleEnum, Partial]
    c: Annotated[SampleMixedEnum, Partial]
    d: Annotated[Literal["a_value", 10], Partial]
    e: Annotated[Literal["a_value"], Partial]
    f: Literal["a_value"]
    g: Annotated[UUID, Partial]
    h: Optional[Annotated[datetime, Partial]]
    i: Optional[NestedA]


def test_partial():
    partial = Partial[SamplePartial]
    assert partial.model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {"b": {"title": "B", "type": "integer"}},
                "required": ["b"],
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"$ref": "#/$defs/PartialSampleNestedPartial"},
        },
        "required": ["a", "b"],
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Wrapped model JSON schema has changed"
    assert partial.get_partial_model().model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "B",
                    }
                },
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "A",
            },
            "b": {
                "anyOf": [
                    {"$ref": "#/$defs/PartialSampleNestedPartial"},
                    {"type": "null"},
                ],
                "default": {},
            },
        },
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Partial model JSON schema has changed"


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]
    # Handle any leading whitespace from the model
    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    for model, expected_dict in zip(
        partial.model_from_chunks(["\n", "\t", " ", '{"b": {"b": 1}}']),
        expected_model_dicts,
    ):
        assert model.model_dump() == expected_dict

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model
    async def async_generator():
        for chunk in ["\n", "\t", " ", '{"b": {"b": 1}}']:
            yield chunk

    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    i = 0
    async for model in partial.model_from_chunks_async(async_generator()):
        assert model.model_dump() == expected_model_dicts[i]
        i += 1

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


def test_summary_extraction():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = OpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = AsyncOpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


def test_union_with_nested():
    partial = Partial[UnionWithNested]
    partial.get_partial_model().model_validate_json(
        '{"a": [{"b": "b"}, {"d": "d"}], "b": [{"b": "b"}], "c": {"d": "d"}, "e": [1, "a"]}'
    )


def test_partial_enums():
    # Test that we can annotate enum values with `Partial` and support parsing
    # partial values with the partial model
    partial = Partial[PartialEnums]
    partial_results = (
        '{"a": "a_", "b": "b_", "c": "c_v", "d": 1, "e": "a_", "f": "a_value", "g": "1", "h": "", "i": {"c": ""}}'
    )
    partial_validated = partial.get_partial_model().model_validate_json(partial_results)

    assert partial_validated.a is None
    assert partial_validated.b is None
    assert partial_validated.c is None
    assert partial_validated.d is None
    assert partial_validated.e is None
    assert partial_validated.f == "a_value"
    assert partial_validated.g is None
    assert partial_validated.h is None
    assert partial_validated.i is not None
    assert partial_validated.i.c is None


    with pytest.raises(ValidationError):
        partial.model_validate_json(partial_results)

    with pytest.raises(ValidationError):
        # "f" is not marked as a partil enum
        partial.get_partial_model().model_validate_json('{"f": "a_"}')

    resolved_enum_partial_results = (
        '{"a": "a_value", "b": "b_value", "c": "c_v", "d": 10, "g": "123e4567-e89b-12d3-a456-426655440000", "h": "2024-01-01T00:00:00"}'
    )
    resolved_enum_partial_validated = partial.get_partial_model().model_validate_json(
        resolved_enum_partial_results
    )
    assert resolved_enum_partial_validated.a == "a_value"
    assert resolved_enum_partial_validated.b == "b_value"
    # this value still isn't fully resolved
    assert resolved_enum_partial_validated.c is None
    assert resolved_enum_partial_validated.d == 10
    assert resolved_enum_partial_validated.g == UUID("123e4567-e89b-12d3-a456-426655440000")
    assert resolved_enum_partial_validated.h == datetime(2024, 1, 1)
