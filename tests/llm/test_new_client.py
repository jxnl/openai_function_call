import cohere
import os
import openai
import instructor
import anthropic
import pytest
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    age: int


def test_client_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_messages_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_chat_completions_create_with_response():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user, completion = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10

    from openai.types.chat import ChatCompletion

    assert isinstance(completion, ChatCompletion)


def test_client_chat_completions_create():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_chat_completions_create_partial():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    for user in client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)


def test_client_chat_completions_create_iterable():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-3.5-turbo")

    users = [
        user
        for user in client.chat.completions.create_iterable(
            response_model=User,
            messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
            temperature=0,
        )
    ]
    assert len(users) == 2


@pytest.mark.asyncio
async def test_async_client_chat_completions_create():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    user = await instructor_client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_partial():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    async for user in instructor_client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_iterable():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    async for user in instructor_client.chat.completions.create_iterable(
        response_model=User,
        messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
        temperature=0,
    ):
        assert isinstance(user, User)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_with_response():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-3.5-turbo")

    user, response = await instructor_client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    from openai.types.chat import ChatCompletion

    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, ChatCompletion)


def test_client_from_anthropic_with_response():
    client = instructor.from_anthropic(
        anthropic.Anthropic(),
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user, response = client.messages.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, anthropic.types.Message)


def test_client_anthropic_response():
    client = anthropic.Anthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_anthropic_bedrock_response():
    client = anthropic.AnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_anthropic_response():
    client = anthropic.AsyncAnthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-3-haiku-20240307",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_anthropic_bedrock_response():
    client = anthropic.AsyncAnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
def test_client_cohere_response():
    client = cohere.Client()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-r-plus",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
def test_client_cohere_response_with_nested_classes():
    client = cohere.Client()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-r-plus",
    )

    class Person(BaseModel):
        name: str = Field(description="name of the person")
        country_of_origin: str = Field(description="country of origin of the person")

    class Group(BaseModel):
        group_name: str = Field(description="name of the group")
        members: list[Person] = Field(description="list of members in the group")

    task = """\
    Given the following text, create a Group object for 'The Beatles' band

    Text:
    The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
    """
    group = instructor_client.messages.create(
        response_model=Group,
        messages=[{"role": "user", "content": task}],
        temperature=0,
    )
    assert group.group_name == "The Beatles"
    assert len(group.members) == 4
    assert group.members[0].name == "John Lennon"
    assert group.members[1].name == "Paul McCartney"
    assert group.members[2].name == "George Harrison"
    assert group.members[3].name == "Ringo Starr"


@pytest.mark.skip(reason="Skipping if Cohere API is not available")
@pytest.mark.asyncio
async def test_client_cohere_async():
    client = cohere.AsyncClient()
    instructor_client = instructor.from_cohere(
        client,
        max_tokens=1000,
        model="command-r-plus",
    )

    class Person(BaseModel):
        name: str = Field(description="name of the person")
        country_of_origin: str = Field(description="country of origin of the person")

    class Group(BaseModel):
        group_name: str = Field(description="name of the group")
        members: list[Person] = Field(description="list of members in the group")

    task = """\
    Given the following text, create a Group object for 'The Beatles' band

    Text:
    The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
    """
    group = await instructor_client.messages.create(
        response_model=Group,
        messages=[{"role": "user", "content": task}],
        temperature=0,
    )
    assert group.group_name == "The Beatles"
    assert len(group.members) == 4
    assert group.members[0].name == "John Lennon"
    assert group.members[1].name == "Paul McCartney"
    assert group.members[2].name == "George Harrison"
    assert group.members[3].name == "Ringo Starr"


@pytest.mark.skip(reason="Skip for now")
def test_client_from_mistral_with_response():
    import mistralai.client as mistralaicli

    client = instructor.from_mistral(
        mistralaicli.MistralClient(),
        max_tokens=1000,
        model="mistral-large-latest",
    )

    user, response = client.messages.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skip for now")
def test_client_mistral_response():
    import mistralai.client as mistralaicli

    client = mistralaicli.MistralClient()
    instructor_client = instructor.from_mistral(
        client, max_tokens=1000, model="mistral-large-latest"
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
