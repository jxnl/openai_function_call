---
title: "Generate In-Context Examples"
description: "If we do not have examples for our task, we can utilize *self-generated* in-context learning (SG-ICL), where we use a model to generate in-context examples."
---

How can we generate examples of our task to improve model outputs?

In-context learning is a prompting technique where examples are provided in the prompt for the model to learn from at inference time. If we do not already have examples for our task, we can utilize *self-generated* in-context learning (SG-ICL), where we use a model to generate these in-context examples.

## Implementation

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel


class GeneratedExample(BaseModel):
    input: str
    output: str


class Response(BaseModel):
    output: str


client = instructor.from_openai(OpenAI())


def generate_example(task, input, case):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=GeneratedExample,
        messages=[
            {
                "role": "user",
                "content": f"""
                           Generate an example for this task
                           {task}
                           that has this output
                           {case}
                           similar to this input
                           {input}
                           """,
            }
        ],
    )


def inference(examples, task, input):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"""
                           {examples}
                           {task}
                           {input}
                           """,
            }
        ],
    )


if __name__ == "__main__":
    task = "Predict the sentiment of the following text:"
    input = "This movie was a rollercoaster of emotions, keeping me engaged throughout."
    example_cases = ["positive", "negative"]

    examples = [
        generate_example(task, input, case)
        for case in example_cases
        for _ in range(2)
    ]  # Generate 2 examples per case

    for example in examples:
        print(example)
        """
        input='The performance of the lead actor was stellar, leaving a lasting impression.' output='positive'
        """
        """
        input="The weather today has been absolutely wonderful, lifting everyone's spirits." output='positive'
        """
        """
        input='The meal was overpriced and underwhelming, not worth the hype.' output='negative'
        """
        """
        input='The customer service experience was frustrating and disappointing.' output='negative'
        """

    print(inference(examples, task, input))
    #> output='positive'
```

## References

<sup id="ref-1">1</sup>: [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://arxiv.org/abs/2206.08082)

