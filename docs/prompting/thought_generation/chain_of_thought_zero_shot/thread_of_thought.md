---
title: "Thread of Thought Unraveling Chaotic Contexts"
description: "Thread of Thought helps models ignore irrelevant context in their prompt, improving overall response quality and relevance"
---

# Thread Of Thought unravelling chaotic contexts

Thread Of Thought <sup><a href="https://arxiv.org/pdf/2311.08734">1</a></sup> aims to help language models deal with irrelevant context, thus improving reasoning perfomance.

We can implement Thread Of Thought as seen below.

!!! example "Thread Of Thought template"

    **[ Input Prompt ]**

    Proceed through the context systematically, zeroing in on areas that could provide the answers we’re seeking

We can implement this in Instructor as seen below

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class Response(BaseModel):
    analysis: list[str] = Field(
        ...,
        description="This is a list that should contain a single \
            explanation for each source about its relevance and content.",
    )
    answer: str = Field(..., description="This is the answer to\
         the user's question")


response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "system",
            "content": """
                You are an expert Question Answerer.

                Here are all of the sources that you should refer to for context
                - The price of a house was $100,000 in 2024
                - The Great Wall of China is not visible from space \
                    with the naked eye
                - Honey never spoils; archaeologists have found pots of \
                    honey in ancient Egyptian tombs that are over 3,000 years old
                -The world's oldest known living tree is over 5,000 years \
                    old and is located in California
                - The price of a house was $80,000 in 2023
            """.strip(),
        },
        {
            "role": "user",
            "content": "What is the increase in the price of a home in \
                absolute figures in the US?",
        },
        {
            "role": "assistant",
            "content": "Proceed through the context systematically, zeroing in \
                on areas that could provide the answers we’re seeking",
        },
    ],
)

print(response)
# The increase in the price of a home from 2023 to
# 2024 is $20,000 in absolute figures.
```

## Useful Tips

Here are some alternative phrases that you can add to your prompt to generate a thread of thought before your model generates a response.

1. In a step-by-step manner, go through the context, surfacing important information that could be useful.
2. Walk me through this lengthy document segment by segment, focusing on each part's significance.
3. Guide me through the context part by part, providing insights along the way.
4. Divide the document into manageable parts and guide me through each one, providing insights as we move along.
5. Let's go through this document piece by piece, paying close attention to each section.
6. Take me through the context bit by bit, making sure we capture all important aspects.
7. Examine the document in chunks, evaluating each part critically before moving to the next.
8. Analyze the context by breaking it down into sections, summarizing each as we move forward.
9. Navigate through the context incrementally, identifying and summarizing relevant portions.
10. Proceed through the context systematically, zeroing in on areas that could provide the answers we're seeking.
11. Take me through this long document step-by-step, making sure not to miss any important details.
12. Analyze this extensive document in sections, summarizing each one and noting any key points.
13. Navigate through this long document by breaking it into smaller parts and summarizing each, so we don't miss anything.
14. Let's navigate through the context section by section, identifying key elements in each part.
15. Let's dissect the context into smaller pieces, reviewing each one for its importance and relevance.
16. Carefully analyze the context piece by piece, highlighting relevant points for each question.
17. Read the context in sections, concentrating on gathering insights that answer the question at hand.
18. Let's read through the document section by section, analyzing each part carefully as we go.
19. Let's dissect this document bit by bit, making sure to understand the nuances of each section.
20. Systematically work through this document, summarizing and analyzing each portion as we go.
21. Let's explore the context step-by-step, carefully examining each segment.
22. Systematically go through the context, focusing on each part individually.
23. Methodically examine the context, focusing on key segments that may answer the query.
24. Progressively sift through the context, ensuring we capture all pertinent details.
25. Take a modular approach to the context, summarizing each part before drawing any conclusions.
26. Examine each segment of the context meticulously, and let's discuss the findings.
27. Approach the context incrementally, taking the time to understand each portion fully.
28. Let's scrutinize the context in chunks, keeping an eye out for information that answers our queries.
29. Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.
30. Let's take a segmented approach to the context, carefully evaluating each part for its relevance to the questions posed.

### References

<sup id="ref-1">1</sup>: [Thread of Thought Unraveling Chaotic Contexts](https://arxiv.org/pdf/2311.08734)