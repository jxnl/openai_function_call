# Instructor, The Most Popular Library for Simple Structured Outputs

Instructor is the most popular Python library for working with structured outputs from large language models (LLMs), boasting over 600,000 monthly downloads. Built on top of Pydantic, it provides a simple, transparent, and user-friendly API to manage validation, retries, and streaming responses. Get ready to supercharge your LLM workflows with the community's top choice!

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/bD9YE9JArw)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)

## Want your logo on our website?

If your company uses Instructor a lot, we'd love to have your logo on our website! Please fill out [this form](https://q7gjsgfstrp.typeform.com/to/wluQlVVQ)

## Key Features

- **Response Models**: Specify Pydantic models to define the structure of your LLM outputs
- **Retry Management**: Easily configure the number of retry attempts for your requests
- **Validation**: Ensure LLM responses conform to your expectations with Pydantic validation
- **Streaming Support**: Work with Lists and Partial responses effortlessly
- **Flexible Backends**: Seamlessly integrate with various LLM providers beyond OpenAI
- **Support in many Languages**: We support many languages including [Python](https://python.useinstructor.com), [TypeScript](https://js.useinstructor.com), [Ruby](https://ruby.useinstructor.com), [Go](https://go.useinstructor.com), and [Elixir](https://hex.pm/packages/instructor)

## Get Started in Minutes

Install Instructor with a single command:

```bash
pip install -U instructor
```

Now, let's see Instructor in action with a simple example:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30
```

### Using Hooks

Instructor provides a powerful hooks system that allows you to intercept and log various stages of the LLM interaction process. Here's a simple example demonstrating how to use hooks:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


# Initialize the OpenAI client with Instructor
client = instructor.from_openai(OpenAI())


# Define hook functions
def log_kwargs(**kwargs):
    print(f"Function called with kwargs: {kwargs}")


def log_exception(exception: Exception):
    print(f"An exception occurred: {str(exception)}")


client.on("completion:kwargs", log_kwargs)
client.on("completion:error", log_exception)

user_info = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "Extract the user name: 'John is 20 years old'"}
    ],
)

"""
{
        'args': (),
        'kwargs': {
            'messages': [
                {
                    'role': 'user',
                    'content': "Extract the user name: 'John is 20 years old'",
                }
            ],
            'model': 'gpt-4o-mini',
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'UserInfo',
                        'description': 'Correctly extracted `UserInfo` with all the required parameters with correct types',
                        'parameters': {
                            'properties': {
                                'name': {'title': 'Name', 'type': 'string'},
                                'age': {'title': 'Age', 'type': 'integer'},
                            },
                            'required': ['age', 'name'],
                            'type': 'object',
                        },
                    },
                }
            ],
            'tool_choice': {'type': 'function', 'function': {'name': 'UserInfo'}},
        },
    }
"""

print(f"Name: {user_info.name}, Age: {user_info.age}")
#> Name: John, Age: 20
``` 

This example demonstrates:
1. A pre-execution hook that logs all kwargs passed to the function.
2. An exception hook that logs any exceptions that occur during execution.

The hooks provide valuable insights into the function's inputs and any errors,
enhancing debugging and monitoring capabilities.

### Using Anthropic Models

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_anthropic(Anthropic())

# note that client.chat.completions.create will also work
resp = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    system="You are a world class AI that excels at extracting user data from a sentence",
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Cohere Models

Make sure to install `cohere` and set your system environment variable with `export CO_API_KEY=<YOUR_COHERE_API_KEY>`.

```
pip install cohere
```

```python
import instructor
import cohere
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_cohere(cohere.Client())

# note that client.chat.completions.create will also work
resp = client.chat.completions.create(
    model="command-r-plus",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Gemini Models

Make sure you [install](https://ai.google.dev/api/python/google/generativeai#setup) the Google AI Python SDK. You should set a `GOOGLE_API_KEY` environment variable with your API key.
Gemini tool calling also requires `jsonref` to be installed.

```
pip install google-generativeai jsonref
```

```python
import instructor
import google.generativeai as genai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# genai.configure(api_key=os.environ["API_KEY"]) # alternative API key configuration
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",  # model defaults to "gemini-pro"
    ),
    mode=instructor.Mode.GEMINI_JSON,
)
```

Alternatively, you can [call Gemini from the OpenAI client](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library#python). You'll have to setup [`gcloud`](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev), get setup on Vertex AI, and install the Google Auth library.

```sh
pip install google-auth
```

```python
import google.auth
import google.auth.transport.requests
import instructor
from openai import OpenAI
from pydantic import BaseModel

creds, project = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

# Pass the Vertex endpoint and authentication to the OpenAI SDK
PROJECT = 'PROJECT_ID'
LOCATION = (
    'LOCATION'  # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
)
base_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi'

client = instructor.from_openai(
    OpenAI(base_url=base_url, api_key=creds.token), mode=instructor.Mode.JSON
)


# JSON mode is req'd
class User(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="google/gemini-1.5-flash-001",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Litellm

```python
import instructor
from litellm import completion
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_litellm(completion)

resp = client.chat.completions.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

### Using Mistral Models with Multimodal Support

Make sure to install `mistralai` and set your system environment variable with `export MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>`.

```bash
pip install mistralai
```

```python
import instructor
from mistralai import MistralClient
from instructor.multimodal import Image
from pydantic import BaseModel, Field


class ImageAnalysis(BaseModel):
    description: str = Field(..., description="A detailed description of the image")
    objects: list[str] = Field(..., description="List of objects identified in the image")
    colors: list[str] = Field(..., description="List of dominant colors in the image")


# Initialize the Mistral client with Instructor
client = instructor.from_mistral(
    MistralClient(api_key="your-api-key"),
    mode=instructor.Mode.MISTRAL_JSON
)

# Analyze an image using Pixtral model
analysis = client.chat.completions.create(
    model="pixtral-12b-2409",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? List the objects and colors."},
                Image.from_url("https://example.com/image.jpg")  # You can also use Image.from_path()
            ]
        }
    ],
    response_model=ImageAnalysis,
)

print(f"Description: {analysis.description}")
print(f"Objects: {', '.join(analysis.objects)}")
print(f"Colors: {', '.join(analysis.colors)}")

# Example with multiple images
images = [
    Image.from_url("https://example.com/image1.jpg"),
    Image.from_url("https://example.com/image2.jpg"),
]

analysis = client.chat.completions.create(
    model="pixtral-12b-2409",
    messages=[
        {
            "role": "user",
            "content": ["Describe these images"] + images,
        }
    ],
    response_model=ImageAnalysis,
)
```

## Types are inferred correctly

This was the dream of Instructor but due to the patching of OpenAI, it wasn't possible for me to get typing to work well. Now, with the new client, we can get typing to work well! We've also added a few `create_*` methods to make it easier to create iterables and partials, and to access the original completion.

### Calling `create`

```python
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(openai.OpenAI())

user = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

Now if you use an IDE, you can see the type is correctly inferred.

![type](./docs/blog/posts/img/type.png)

### Handling async: `await create`

This will also work correctly with asynchronous clients.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.AsyncOpenAI())


class User(BaseModel):
    name: str
    age: int


async def extract():
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=User,
    )
```

Notice that simply because we return the `create` method, the `extract()` function will return the correct user type.

![async](./docs/blog/posts/img/async_type.png)

### Returning the original completion: `create_with_completion`

You can also return the original completion object

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user, completion = client.chat.completions.create_with_completion(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

![with_completion](./docs/blog/posts/img/with_completion.png)

### Streaming Partial Objects: `create_partial`

In order to handle streams, we still support `Iterable[T]` and `Partial[T]` but to simplify the type inference, we've added `create_iterable` and `create_partial` methods as well!

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user_stream = client.chat.completions.create_partial(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)

for user in user_stream:
    print(user)
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name='John Doe' age=None
    #> name='John Doe' age=None
    #> name='John Doe' age=None
    #> name='John Doe' age=30
    #> name='John Doe' age=30
    # name=None age=None
    # name='' age=None
    # name='John' age=None
    # name='John Doe' age=None
    # name='John Doe' age=30
```

Notice now that the type inferred is `Generator[User, None]`

![generator](./docs/blog/posts/img/generator.png)

### Streaming Iterables: `create_iterable`

We get an iterable of objects when we want to extract multiple objects.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


users = client.chat.completions.create_iterable(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create 2 users"},
    ],
    response_model=User,
)

for user in users:
    print(user)
    #> name='John Doe' age=30
    #> name='Jane Doe' age=28
    # User(name='John Doe', age=30)
    # User(name='Jane Smith', age=25)
```

![iterable](./docs/blog/posts/img/iterable.png)

## [Evals](https://github.com/jxnl/instructor/tree/main/tests/llm/test_openai/evals#how-to-contribute-writing-and-running-evaluation-tests)

We invite you to contribute to evals in `pytest` as a way to monitor the quality of the OpenAI models and the `instructor` library. To get started check out the evals for [Anthropic](https://github.com/jxnl/instructor/blob/main/tests/llm/test_anthropic/evals/test_simple.py) and [OpenAI](https://github.com/jxnl/instructor/tree/main/tests/llm/test_openai/evals#how-to-contribute-writing-and-running-evaluation-tests) and contribute your own evals in the form of pytest tests. These evals will be run once a week and the results will be posted.

## Contributing

If you want to help, checkout some of the issues marked as `good-first-issue` or `help-wanted` found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cookbook.

## CLI

We also provide some added CLI functionality for easy convenience:

- `instructor jobs` : This helps with the creation of fine-tuning jobs with OpenAI. Simple use `instructor jobs create-from-file --help` to get started creating your first fine-tuned GPT-3.5 model

- `instructor files` : Manage your uploaded files with ease. You'll be able to create, delete and upload files all from the command line

- `instructor usage` : Instead of heading to the OpenAI site each time, you can monitor your usage from the CLI and filter by date and time period. Note that usage often takes ~5-10 minutes to update from OpenAI's side

## License

This project is licensed under the terms of the MIT License.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/jxnl/instructor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxnl/instructor" />
</a>
