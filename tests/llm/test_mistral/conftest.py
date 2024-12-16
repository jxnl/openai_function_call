"""Pytest configuration for Mistral tests."""
import os
import pytest
from mistralai.client import MistralClient
from collections.abc import Iterator

def pytest_collection_modifyitems(items: Iterator[pytest.Item]) -> None:
    """Mark tests requiring Mistral API key."""
    for item in items:
        if "test_mistral" in str(item.fspath):
            item.add_marker(pytest.mark.requires_mistral)

@pytest.fixture
def client():
    """Create a Mistral client for testing."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY environment variable not set")
    return MistralClient(api_key=api_key)
