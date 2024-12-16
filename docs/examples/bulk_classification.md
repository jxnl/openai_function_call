---
title: User-Provided Tag Classification Tutorial
description: Learn to classify user-provided tags effectively using async functions and FastAPI for parallel processing.
---

# Bulk Classification from User-Provided Tags.

This tutorial shows how to do classification from user provided tags. This is valuable when you want to provide services that allow users to do some kind of classification.

!!! tips "Motivation"

    Imagine allowing the user to upload documents as part of a RAG application. Oftentimes, we might want to allow the user to specify an existing set of tags, give descriptions, and do the classification for them.

## Defining the Structures

One of the easy things to do is to allow users to define a set of tags in some kind of schema and save that in a database. Here's an example of a schema that we might use:

| tag_id | name     | instructions         |
| ------ | -------- | -------------------- |
| 0      | personal | Personal information |
| 1      | phone    | Phone number         |
| 2      | email    | Email address        |
| 3      | address  | Address              |
| 4      | Other    | Other information    |

1. **tag_id** — The unique identifier for the tag.
2. **name** — The name of the tag.
3. **instructions** — A description of the tag, which can be used as a prompt to describe the tag.

## Implementing the Classification

In order to do this we'll do a couple of things:

0. We'll use the `instructor` library to patch the `openai` library to use the `AsyncOpenAI` client.
1. Implement a `Tag` model that will be used to validate the tags from the context. (This will allow us to avoid hallucinating tags that are not in the context.)
2. Helper models for the request and response.
3. An async function to do the classification.
4. A main function to run the classification using the `asyncio.gather` function to run the classification in parallel.

If you want to learn more about how to do bad computations, check out our post on AsyncIO [here](../blog/posts/learn-async.md).

```python
import openai
import instructor

client = instructor.from_openai(
    openai.AsyncOpenAI(),
)
```

First, we'll need to import all of our Pydantic and instructor code and use the AsyncOpenAI client. Then, we'll define the tag model along with the tag instructions to provide input and output.

This is very helpful because once we use something like FastAPI to create endpoints, the Pydantic functions will serve as multiple tools:

1. A description for the developer
2. Type hints for the IDE
3. OpenAPI documentation for the FastAPI endpoint
4. Schema and Response Model for the language model.

```python
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]


class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]
```

Let's delve deeper into what the `validate_ids` function does. Notice that its purpose is to extract tags from the context and ensure that each ID and name exists in the set of tags. This approach helps minimize hallucinations. If we mistakenly identify either the ID or the tag, an error will be thrown, and the instructor will prompt the language model to retry until the correct item is successfully extracted.

```python
from pydantic import model_validator, ValidationInfo


@model_validator(mode="after")
def validate_ids(self, info: ValidationInfo):
    context = info.context
    if context:
        tags: List[Tag] = context.get("tags")
        assert self.id in {
            tag.id for tag in tags
        }, f"Tag ID {self.id} not found in context"
        assert self.name in {
            tag.name for tag in tags
        }, f"Tag name {self.name} not found in context"
    return self
```

Now, let's implement the function to do the classification. This function will take a single text and a list of tags and return the predicted tag.

```python
# <%hide%>
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]


class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]


# <%hide%>
async def tag_single_request(text: str, tags: List[Tag]) -> Tag:
    allowed_tags = [(tag.id, tag.name) for tag in tags]
    allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])

    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class text tagging system.",
            },
            {"role": "user", "content": f"Describe the following text: `{text}`"},
            {
                "role": "user",
                "content": f"Here are the allowed tags: {allowed_tags_str}",
            },
        ],
        response_model=Tag,  # Minimizes the hallucination of tags that are not in the allowed tags.
        validation_context={"tags": tags},
    )


async def tag_request(request: TagRequest) -> TagResponse:
    predictions = await asyncio.gather(
        *[tag_single_request(text, request.tags) for text in request.texts]
    )
    return TagResponse(
        texts=request.texts,
        predictions=predictions,
    )
```

Notice that we first define a single async function that makes a prediction of a tag, and we pass it into the validation context in order to minimize hallucinations.

Finally, we'll implement the main function to run the classification using the `asyncio.gather` function to run the classification in parallel.

```python
import asyncio

# <%hide%>
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator
import instructor
import openai

client = instructor.from_openai(
    openai.AsyncOpenAI(),
)


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]


class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]


async def tag_single_request(text: str, tags: List[Tag]) -> Tag:
    allowed_tags = [(tag.id, tag.name) for tag in tags]
    allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])

    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class text tagging system.",
            },
            {"role": "user", "content": f"Describe the following text: `{text}`"},
            {
                "role": "user",
                "content": f"Here are the allowed tags: {allowed_tags_str}",
            },
        ],
        response_model=Tag,  # Minimizes the hallucination of tags that are not in the allowed tags.
        validation_context={"tags": tags},
    )


async def tag_request(request: TagRequest) -> TagResponse:
    predictions = await asyncio.gather(
        *[tag_single_request(text, request.tags) for text in request.texts]
    )
    return TagResponse(
        texts=request.texts,
        predictions=predictions,
    )

## working-with-dataframes

When working with large datasets, it's often convenient to use pandas DataFrames. Here's how you can integrate this classification system with pandas:

```python
import pandas as pd

async def classify_dataframe(df: pd.DataFrame, text_column: str, tags: List[TagWithInstructions]) -> pd.DataFrame:
    request = TagRequest(
        texts=df[text_column].tolist(),
        tags=tags
    )
    response = await tag_request(request)
    df['predicted_tag'] = [pred.name for pred in response.predictions]
    return df
```

## streaming-responses

For real-time processing, you can stream responses as they become available:

```python
async def stream_classifications(texts: List[str], tags: List[TagWithInstructions]):
    async def process_single(text: str):
        prediction = await tag_single_request(text, tags)
        return {"text": text, "prediction": prediction}

    tasks = [process_single(text) for text in texts]
    for completed in asyncio.as_completed(tasks):
        yield await completed
```

## Single-Label Classification

For simple classification tasks where each text belongs to exactly one category:

```python
async def classify_single_label(text: str, tags: List[TagWithInstructions]) -> Tag:
    return await tag_single_request(text, tags)
```

## Multi-Label Classification

For cases where texts might belong to multiple categories:

```python
class MultiLabelTag(BaseModel):
    tags: List[Tag]

    @model_validator(mode="after")
    def validate_tags(self, info: ValidationInfo):
        context = info.context
        if context and context.get("tags"):
            valid_tags = context["tags"]
            for tag in self.tags:
                assert tag.id in {t.id for t in valid_tags}, f"Tag ID {tag.id} not found"
                assert tag.name in {t.name for t in valid_tags}, f"Tag name {tag.name} not found"
        return self

async def classify_multi_label(text: str, tags: List[TagWithInstructions]) -> List[Tag]:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a multi-label classification system."},
            {"role": "user", "content": f"Classify this text into multiple categories: {text}"},
            {"role": "user", "content": f"Available categories: {', '.join(t.name for t in tags)}"},
        ],
        response_model=MultiLabelTag,
        validation_context={"tags": tags},
    )
    return response.tags
```

# Example Usage

```python
# PLACEHOLDER: existing example code
```


# <%hide%>
tags = [
    TagWithInstructions(id=0, name="personal", instructions="Personal information"),
    TagWithInstructions(id=1, name="phone", instructions="Phone number"),
    TagWithInstructions(id=2, name="email", instructions="Email address"),
    TagWithInstructions(id=3, name="address", instructions="Address"),
    TagWithInstructions(id=4, name="Other", instructions="Other information"),
]

# Texts will be a range of different questions.
# Such as "How much does it cost?", "What is your privacy policy?", etc.
texts = [
    "What is your phone number?",
    "What is your email address?",
    "What is your address?",
    "What is your privacy policy?",
]

# The request will contain the texts and the tags.
request = TagRequest(texts=texts, tags=tags)

# The response will contain the texts, the predicted tags, and the confidence.
response = asyncio.run(tag_request(request))
print(response.model_dump_json(indent=2))
"""
{
  "texts": [
    "What is your phone number?",
    "What is your email address?",
    "What is your address?",
    "What is your privacy policy?"
  ],
  "predictions": [
    {
      "id": 1,
      "name": "phone"
    },
    {
      "id": 2,
      "name": "email"
    },
    {
      "id": 3,
      "name": "address"
    },
    {
      "id": 4,
      "name": "Other"
    }
  ]
}
"""
```

Which would result in:

```json
{
  "texts": [
    "What is your phone number?",
    "What is your email address?",
    "What is your address?",
    "What is your privacy policy?"
  ],
  "predictions": [
    {
      "id": 1,
      "name": "phone"
    },
    {
      "id": 2,
      "name": "email"
    },
    {
      "id": 3,
      "name": "address"
    },
    {
      "id": 4,
      "name": "Other"
    }
  ]
}
```

## What happens in production?

If we were to use this in production, we might expect to have some kind of fast API endpoint.

```python
from fastapi import FastAPI

app = FastAPI()

# <%hide%>
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]


class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]


# <%hide%>
@app.post("/tag", response_model=TagResponse)
async def tag(request: TagRequest) -> TagResponse:
    return await tag_request(request)
```

Since everything is already annotated with Pydantic, this code is very simple to write!

!!! warning "Where do tags come from?"

    I just want to call out that here you can also imagine the tag spec IDs and names and instructions for example could come from a database or somewhere else. I'll leave this as an exercise to the reader, but I hope this gives us a clear understanding of how we can do something like user-defined classification.

## Improving the Model

There's a couple things we could do to make this system a little bit more robust.

1. Use confidence score:

```python
# <%hide%>
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator, Field


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


# <%hide%>
class TagWithConfidence(Tag):
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="The confidence of the prediction, 0 is low, 1 is high",
    )
```

2. Use multiclass classification:

Notice in the example we use Iterable[Tag] vs Tag. This is because we might want to use a multiclass classification model that returns multiple tag!

```python
import instructor
import openai
import asyncio
from typing import Iterable

client = instructor.from_openai(
    openai.AsyncOpenAI(),
)

# <%hide%>
from typing import List
from pydantic import BaseModel, ValidationInfo, model_validator


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {
                tag.id for tag in tags
            }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


# <%hide%>
tags = [
    Tag(id=0, name="personal"),
    Tag(id=1, name="phone"),
    Tag(id=2, name="email"),
    Tag(id=3, name="address"),
    Tag(id=4, name="Other"),
]

# Texts will be a range of different questions.
# Such as "How much does it cost?", "What is your privacy policy?", etc.
text = "What is your phone number?"


async def get_tags(text: List[str], tags: List[Tag]) -> List[Tag]:
    allowed_tags = [(tag.id, tag.name) for tag in tags]
    allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])

    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class text tagging system.",
            },
            {"role": "user", "content": f"Describe the following text: `{text}`"},
            {
                "role": "user",
                "content": f"Here are the allowed tags: {allowed_tags_str}",
            },
        ],
        response_model=Iterable[Tag],
        validation_context={"tags": tags},
    )


tag_results = asyncio.run(get_tags(text, tags))
for tag in tag_results:
    print(tag)
    #> id=1 name='phone'
