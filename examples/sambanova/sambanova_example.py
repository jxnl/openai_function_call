import os
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor


class Character(BaseModel):
    name: str
    fact: list[str] = Field(..., description="A list of facts about the subject")


client = OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
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
