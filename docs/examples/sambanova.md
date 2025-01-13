---
title: Using SambaNova for Inference: Setup and Example
description: Learn how to use SambaNova Cloud for inference with the Llama3-405B model, including API setup and a practical Python example.
---

# Structured Outputs using SambaNova Cloud
Instead of using openai or antrophic you can now also use sambanova for inference by using from_sambanova.

The examples are using llama3-405b model.

## SambaNova Cloud API
To use SambaNova Cloud you need to obtain a SambaNova Cloud API key.
Goto [SambaNova Cloud](https://cloud.sambanova.ai/) and login. Select APIs from the left menu and then create an account. After that, go again to APIs and generate your API key.

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor pydantic openai
```
You need to export the SambaNova Cloud API key and URL:
```
export SAMBANOVA_API_KEY="your-sambanova-cloud-api-key"
export SAMBANOVA_URL="sambanova-cloud-url"
```

An example:
```python
import os
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import instructor


class Character(BaseModel):
    name: str
    fact: List[str] = Field(..., description="A list of facts about the subject")


client = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url=os.getenv("SAMBANOVA_URL"),
)

client = instructor.from_sambanova(client, mode=instructor.Mode.TOOLS)

resp = client.chat.completions.create(
    model="llama3-405b",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the company SambaNova",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "SambaNova",
  "fact": [
    "SambaNova is a company that specializes in artificial intelligence and machine learning.",
    "They are known for their work in natural language processing and computer vision.",
    "SambaNova has received significant funding from investors and has partnered with several major companies to develop and implement their technology."
  ]
}
"""
```