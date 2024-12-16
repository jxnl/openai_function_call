---
draft: False
date: 2024-02-26
title: "Structured outputs with Mistral, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Mistral. Learn how to generate structured, type-safe outputs with Mistral, including multimodal support with Pixtral."
slug: mistral
tags:
  - patching
  - multimodal
authors:
  - shanktt
---

# Structured outputs with Mistral, a complete guide w/ instructor

This guide demonstrates how to use Mistral with Instructor to generate structured outputs. You'll learn how to use function calling with Mistral Large to create type-safe responses, including support for multimodal inputs with Pixtral.

Mistral Large is the flagship model from Mistral AI, supporting 32k context windows and functional calling abilities. Mistral Large's addition of [function calling](https://docs.mistral.ai/guides/function-calling/) makes it possible to obtain structured outputs using JSON schema. With Pixtral, you can now also process images alongside text inputs.

By the end of this blog post, you will learn how to effectively utilize Instructor with Mistral Large and Pixtral for both text and image processing tasks.

## Text Processing with Mistral Large

```python
import os
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

instructor_client = from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=Mode.MISTRAL_TOOLS,
    max_tokens=1000,
)

resp = instructor_client.messages.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)
```

## Multimodal Processing with Pixtral

```python
import os
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode
from instructor.multimodal import Image

class ImageDescription(BaseModel):
    description: str
    objects: list[str]
    colors: list[str]

# Initialize the client with Pixtral model
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
instructor_client = from_mistral(
    client=client,
    model="pixtral",  # Use Pixtral for multimodal capabilities
    mode=Mode.MISTRAL_JSON,
    max_tokens=1000,
)

# Load and process an image
image = Image.from_path("path/to/your/image.jpg")
resp = instructor_client.messages.create(
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "Describe this image in detail, including the main objects and colors present.",
                image
            ]
        }
    ],
    temperature=0,
)

print(resp)
```

## Image Requirements and Validation

When working with images in Pixtral:
- Supported formats: JPEG, PNG, GIF, WEBP
- Maximum image size: 20MB
- Images larger than the size limit will be automatically resized
- Base64 and file paths are supported input formats

The `Image` class handles all validation and preprocessing automatically, ensuring your images meet Mistral's requirements.

## Async Implementation

```python
import os
from pydantic import BaseModel
from mistralai import AsyncMistral
from instructor import from_mistral, Mode

class UserDetails(BaseModel):
    name: str
    age: int

# Initialize async client
client = AsyncMistral(api_key=os.environ.get("MISTRAL_API_KEY"))
instructor_client = from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=Mode.MISTRAL_TOOLS,
    max_tokens=1000,
)

async def get_user_details(text: str) -> UserDetails:
    return await instructor_client.messages.create(
        response_model=UserDetails,
        messages=[{"role": "user", "content": text}],
        temperature=0,
    )

# Usage
import asyncio
user = asyncio.run(get_user_details("Jason is 10"))
print(user)
```

## Streaming Support

Mistral supports streaming responses, which can be useful for real-time processing:

```python
from typing import AsyncIterator
from pydantic import BaseModel

class PartialResponse(BaseModel):
    partial_text: str

async def stream_response(text: str) -> AsyncIterator[PartialResponse]:
    async for partial in instructor_client.messages.create(
        response_model=PartialResponse,
        messages=[{"role": "user", "content": text}],
        temperature=0,
        stream=True,
    ):
        yield partial

# Usage
async for chunk in stream_response("Describe the weather"):
    print(chunk.partial_text)
```

## Using Instructor Hooks

Hooks allow you to add custom processing logic:

```python
from instructor import patch

# Add a custom hook
@patch.register_hook
def log_response(response, **kwargs):
    print(f"Model response: {response}")
    return response

# The hook will be called automatically
result = instructor_client.messages.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)
```

## Best Practices

When working with Mistral and Instructor:

1. **API Key Management**
   - Use environment variables for API keys
   - Consider using a .env file for development

2. **Model Selection**
   - Use mistral-large-latest for complex tasks
   - Use mistral-medium or mistral-small for simpler tasks
   - Use pixtral for multimodal applications

3. **Error Handling**
   - Implement proper try-except blocks
   - Handle rate limits and token limits
   - Use validation_context to prevent hallucinations

4. **Performance Optimization**
   - Use async implementations for concurrent requests
   - Implement streaming for long responses
   - Cache responses when appropriate

## Related Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Instructor GitHub Repository](https://github.com/jxnl/instructor/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [AsyncIO in Python](https://docs.python.org/3/library/asyncio.html)
