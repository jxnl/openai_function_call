from __future__ import annotations
import importlib.util
from typing import Callable, Union, TypeVar

from .mode import Mode
from .process_response import handle_response_model
from .distil import FinetuneFormat, Instructions
from .multimodal import Image
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

T = TypeVar("T")

# Type aliases for client functions
ClientFunction = Union[
    Callable[..., Union[Instructor, AsyncInstructor]],
    None
]

__all__: list[str] = [
    "Instructor",
    "Image",
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

def _extend_all(new_items: list[str]) -> None:
    global __all__
    __all__ = __all__ + new_items

# Initialize optional client functions with explicit types
from_anthropic: ClientFunction = None
from_gemini: ClientFunction = None
from_fireworks: ClientFunction = None
from_cerebras: ClientFunction = None
from_groq: ClientFunction = None
from_mistral: ClientFunction = None
from_cohere: ClientFunction = None
from_vertexai: ClientFunction = None
from_writer: ClientFunction = None

# Import optional clients
if importlib.util.find_spec("anthropic") is not None:
    from .client_anthropic import from_anthropic as _from_anthropic
    globals()["from_anthropic"] = _from_anthropic
    _extend_all(["from_anthropic"])

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    from .client_gemini import from_gemini as _from_gemini
    globals()["from_gemini"] = _from_gemini
    _extend_all(["from_gemini"])

if importlib.util.find_spec("fireworks") is not None:
    from .client_fireworks import from_fireworks as _from_fireworks
    globals()["from_fireworks"] = _from_fireworks
    _extend_all(["from_fireworks"])

if importlib.util.find_spec("cerebras") is not None:
    from .client_cerebras import from_cerebras as _from_cerebras
    globals()["from_cerebras"] = _from_cerebras
    _extend_all(["from_cerebras"])

if importlib.util.find_spec("groq") is not None:
    from .client_groq import from_groq as _from_groq
    globals()["from_groq"] = _from_groq
    _extend_all(["from_groq"])

if importlib.util.find_spec("mistralai") is not None:
    from .client_mistral import from_mistral as _from_mistral
    globals()["from_mistral"] = _from_mistral
    _extend_all(["from_mistral"])

if importlib.util.find_spec("cohere") is not None:
    from .client_cohere import from_cohere as _from_cohere
    globals()["from_cohere"] = _from_cohere
    _extend_all(["from_cohere"])

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    from .client_vertexai import from_vertexai as _from_vertexai
    globals()["from_vertexai"] = _from_vertexai
    _extend_all(["from_vertexai"])

if importlib.util.find_spec("writerai") is not None:
    from .client_writer import from_writer as _from_writer
    globals()["from_writer"] = _from_writer
    _extend_all(["from_writer"])
