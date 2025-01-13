---
title: Structured Outputs with SambaNova AI and Pydantic
description: Learn how to use SambaNova Cloud for structured outputs with Pydantic in Python and enhance API interactions.
---

# Structured Outputs with SambaNova Cloud

This guide demonstrates how to use SambaNova Cloud with Instructor to generate structured outputs. You'll learn how to use SambaNova Cloud's LLM models to create type-safe responses.

you'll need to sign up for an account and get an API key. You can do that [here](https://cloud.sambanova.ai/).

```bash
export SAMBANOVA_API_KEY="your-sambanova-cloud-api-key"
export SAMBANOVA_URL="sambanova-cloud-url"
pip install "instructor openai pydantic"
```

## SambaNova Cloud

SambaNova Cloud supports structured outputs with multiple models like the big `llama3-405b` model.

### Sync Example

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url=os.getenv("SAMBANOVA_URL"),
)

# Enable instructor patches for proxy OpenAI client
client = instructor.from_sambanova(client)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.chat.completions.create(
    model="llama3-405b",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Async Example

Not supported yet. Available soon!
