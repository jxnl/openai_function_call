import pytest  # noqa: F401
from _pytest.config import Config

def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_openai: mark test as requiring OpenAI API credentials",
    )
    config.addinivalue_line(
        "markers",
        "requires_mistral: mark test as requiring Mistral API credentials",
    )
