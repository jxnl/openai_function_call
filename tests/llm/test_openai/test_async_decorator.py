from itertools import product
from pydantic import BaseModel, ValidationInfo
import pytest
import instructor
from instructor.decorators import async_field_validator, async_model_validator
from openai import AsyncOpenAI
from instructor import from_openai
from .util import models, modes
from instructor.function_calls import OpenAISchema


class UserExtractValidated(OpenAISchema):
    name: str
    age: int

    @async_field_validator("name")
    async def validate_name(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_simple_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"


class ValidationResult(OpenAISchema):
    chain_of_thought: str
    is_valid: bool


class ExtractedContent(OpenAISchema):
    relevant_question: str

    @async_field_validator("relevant_question")
    async def validate_relevant_question(cls, v: str, info: ValidationInfo) -> str:
        client = from_openai(AsyncOpenAI())
        if info.context and "content" in info.context:
            original_source = info.context["content"]
            evaluation = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Evaluate and determine if the question is a valid question well supported from the text",
                    },
                    {
                        "role": "user",
                        "content": f"The question is {v} and the source is {original_source}",
                    },
                ],
                response_model=ValidationResult,
            )
            if not evaluation.is_valid:
                raise ValueError(f"{v} is an invalid question!")
            return v

        raise ValueError("Invalid Response!")


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_async_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    content = """
    From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. 

    Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland. 

    In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight. 

    Let each of us here tonight in this Chamber send an unmistakable signal to Ukraine and to the world. 

    Please rise if you are able and show that, Yes, we the United States of America stand with the Ukrainian people. 

    Throughout our history we’ve learned this lesson when dictators do not pay a price for their aggression they cause more chaos.   

    They keep moving.
    """
    model = await aclient.chat.completions.create(
        model=model,
        response_model=ExtractedContent,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": f"Generate a question from the context of {content}",
            },
        ],
        validation_context={"content": content},
    )
    assert isinstance(
        model, ExtractedContent
    ), "Should be instance of Extracted Content"


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_nested_model(model, mode, aclient):
    class Users(OpenAISchema):
        users: list[UserExtractValidated]

    aclient = instructor.from_openai(aclient, mode=mode)
    resp = await aclient.chat.completions.create(
        model=model,
        response_model=Users,
        messages=[
            {
                "role": "user",
                "content": f"Extract users from this sentence - Tom is 22 and lives with his roomate Jack who is 24",
            }
        ],
    )

    assert isinstance(resp, Users)
    for user in resp.users:
        assert user.name.isupper()


@pytest.mark.asyncio
async def test_field_validator():
    class User(OpenAISchema):
        name: str
        label: str

        @async_field_validator("name", "label")
        def validate_user(self, v: str):
            if not v.isupper():
                raise ValueError(f"Uppercase response required for {v}")

    exceptions = await User(name="tom", label="active").model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "Uppercase response required for tom",
        "Uppercase response required for active",
    ]


@pytest.mark.asyncio
async def test_model_validator():
    class User(OpenAISchema):
        name: str
        label: str

        @async_model_validator()
        def validate_user(self):
            if not self.name.isupper() or not self.label.isupper():
                raise ValueError(f"Uppercase response required")

    exceptions = await User(name="tom", label="active").model_async_validate()

    assert len(exceptions) == 1
    assert [str(item) for item in exceptions] == [
        "Uppercase response required",
    ]


@pytest.mark.asyncio
async def test_parsing_nested_field():
    class Users(OpenAISchema):
        users: list[UserExtractValidated]

    exceptions = await Users(
        users=[
            UserExtractValidated(name="thomas", age=22),
            UserExtractValidated(name="vincent", age=24),
        ]
    ).model_async_validate()

    assert len(exceptions) == 2
    assert [str(item) for item in exceptions] == [
        "All Letters in the name must be uppercased. thomas is not a valid response. Eg JASON, TOM not jason, tom",
        "All Letters in the name must be uppercased. vincent is not a valid response. Eg JASON, TOM not jason, tom",
    ]


@pytest.mark.asyncio
async def test_context_passing_in_nested_model():
    class ModelValidationCheck(OpenAISchema):
        user_names: list[str]

        @async_model_validator()
        def validate_model(self, info: ValidationInfo):
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(OpenAISchema):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        model=ModelValidationCheck(user_names=["Jack", "Thomas", "Ben"])
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert str(res[0]) == "Invalid Error but with {'abcdef': '123'}!"


@pytest.mark.asyncio
async def test_context_passing_in_nested_field_validator():
    class ModelValidationCheck(OpenAISchema):
        user_names: list[str]

        @async_field_validator("user_names")
        def validate_model(self, v: list[str], info: ValidationInfo):
            raise ValueError(f"Invalid Error but with {info.context}!")

    class ModelValidationWrapper(OpenAISchema):
        model: ModelValidationCheck

    res = await ModelValidationWrapper(
        model=ModelValidationCheck(user_names=["Jack", "Thomas", "Ben"])
    ).model_async_validate(validation_context={"abcdef": "123"})

    assert len(res) == 1
    assert str(res[0]) == "Invalid Error but with {'abcdef': '123'}!"
