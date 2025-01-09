import importlib.util

from .mode import Mode
from .process_response import handle_response_model
from .distil import FinetuneFormat, Instructions
from .multimodal import Image, Audio
from .dsl import (
    CitationMixin,
    Maybe,
    Partial,
    IterableModel,
    llm_validator,
    openai_moderation,
)
from .function_calls import OpenAISchema, openai_schema
from .patch import apatch, patch
from .process_response import handle_parallel_model
from .client import (
    Instructor,
    AsyncInstructor,
    from_openai,
    from_litellm,
    Provider,
)

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_litellm",
    "AsyncInstructor",
    "Provider",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "Mode",
    "patch",
    "apatch",
    "llm_validator",
    "openai_moderation",
    "FinetuneFormat",
    "Instructions",
    "handle_parallel_model",
    "handle_response_model",
]


if importlib.util.find_spec("anthropic") is not None:
    from .client_anthropic import from_anthropic

    __all__ += ["from_anthropic"]

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    from .client_gemini import from_gemini

    __all__ += ["from_gemini"]

if importlib.util.find_spec("fireworks") is not None:
    from .client_fireworks import from_fireworks

    __all__ += ["from_fireworks"]

if importlib.util.find_spec("cerebras") is not None:
    from .client_cerebras import from_cerebras

    __all__ += ["from_cerebras"]

if importlib.util.find_spec("groq") is not None:
    from .client_groq import from_groq

    __all__ += ["from_groq"]

if importlib.util.find_spec("mistralai") is not None:
    from .client_mistral import from_mistral

    __all__ += ["from_mistral"]

if importlib.util.find_spec("cohere") is not None:
    from .client_cohere import from_cohere

    __all__ += ["from_cohere"]

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    from .client_vertexai import from_vertexai

    __all__ += ["from_vertexai"]

if importlib.util.find_spec("boto3") is not None:
    from .client_bedrock import from_bedrock

    __all__ += ["from_bedrock"]

if importlib.util.find_spec("writerai") is not None:
    from .client_writer import from_writer

    __all__ += ["from_writer"]
