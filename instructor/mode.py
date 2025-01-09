import enum
import warnings


class Mode(enum.Enum):
    """The mode to use for patching the client"""

    FUNCTIONS = "function_call"
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    MISTRAL_TOOLS = "mistral_tools"
    JSON = "json_mode"
    JSON_O1 = "json_o1"
    MD_JSON = "markdown_json_mode"
    JSON_SCHEMA = "json_schema_mode"
    ANTHROPIC_TOOLS = "anthropic_tools"
    ANTHROPIC_JSON = "anthropic_json"
    COHERE_TOOLS = "cohere_tools"
    VERTEXAI_TOOLS = "vertexai_tools"
    VERTEXAI_JSON = "vertexai_json"
    VERTEXAI_PARALLEL_TOOLS = "vertexai_parallel_tools"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    COHERE_JSON_SCHEMA = "json_object"
    TOOLS_STRICT = "tools_strict"
    CEREBRAS_TOOLS = "cerebras_tools"
    CEREBRAS_JSON = "cerebras_json"
    FIREWORKS_TOOLS = "fireworks_tools"
    FIREWORKS_JSON = "fireworks_json"
    WRITER_TOOLS = "writer_tools"
    BEDROCK_TOOLS = "bedrock_tools"
    BEDROCK_JSON = "bedrock_json"

    @classmethod
    def warn_mode_functions_deprecation(cls):
        warnings.warn(
            "The FUNCTIONS mode is deprecated and will be removed in future versions",
            DeprecationWarning,
            stacklevel=2,
        )
