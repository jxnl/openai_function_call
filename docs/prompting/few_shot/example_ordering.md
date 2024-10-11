---
title: "Example Ordering"
description: "LLMs can be sensitive to the order of examples in prompts."
---

Does the order of in-context examples affect your task's output? If so, which ordering provides the best output?

LLMs can be sensitive to the order of examples in prompts<sup><a href="https://arxiv.org/abs/2104.08786">1</a><a href="https://arxiv.org/abs/2106.01751">2</a><a href="https://arxiv.org/abs/2101.06804">3</a><a href="https://aclanthology.org/2022.naacl-main.191/">4</a></sup>. The script below uses `instructor` to test different example permutations and see how the output changes.

## Implementation

```python
from pydantic import BaseModel
import instructor
from openai import OpenAI
from itertools import permutations

client = instructor.from_openai(OpenAI())


class Example(BaseModel):  # (1)!
    input: str
    output: str


class Response(BaseModel):
    response: str


def inference(examples, query):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"{examples} {query}",  # (2)!
            }
        ],
    ).response


if __name__ == "__main__":
    examples = [
        Example(input="The movie was so good", output="positive"),
        Example(input="The movie was somewhat good", output="negative"),
    ]
    query = "The movie was okay"

    permutations = list(permutations(examples))
    results = [inference(permutation, query) for permutation in permutations]
    print(permutations)
    """
    [
        (
            Example(input='The movie was so good', output='positive'),
            Example(input='The movie was somewhat good', output='negative'),
        ),
        (
            Example(input='The movie was somewhat good', output='negative'),
            Example(input='The movie was so good', output='positive'),
        ),
    ]
    """
    print(results)
    #> ['negative', 'positive']
```

1. This class can be customized to a specific task
2. This prompt can be customized to a specific task

!!! info
    For scenarios with a large number of examples, check out example selection techniques ([KNN](https://python.useinstructor.com/prompting/few_shot/exemplar_selection/knn/), [Vote-K](https://python.useinstructor.com/prompting/few_shot/exemplar_selection/vote_k/)).

## References

<sup id="ref-1">1</sup>: [Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://arxiv.org/abs/2104.08786)

<sup id="ref-2">2</sup>: [Reordering Examples Helps during Priming-based Few-Shot Learning](https://arxiv.org/abs/2106.01751)

<sup id="ref-2">3</sup>: [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)

<sup id="ref-3">4</sup>: [Learning To Retrieve Prompts for In-Context Learning](https://aclanthology.org/2022.naacl-main.191/)
