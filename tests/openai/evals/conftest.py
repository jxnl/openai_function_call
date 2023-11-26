# conftest.py
from openai import AsyncOpenAI, OpenAI
import pytest
import os

try:
    import braintrust

    wrap_openai = braintrust.wrap_openai
except ImportError:

    def wrap_openai(x):
        return x

    pass


@pytest.fixture(scope="session")
def client():
    if "BRAINTRUST_API_KEY" in os.environ:
        yield wrap_openai(
            OpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://proxy.braintrustapi.com/v1",
            )
        )
    else:
        yield OpenAI()


@pytest.fixture(scope="session")
def aclient():
    if "BRAINTRUST_API_KEY" in os.environ:
        yield wrap_openai(
            AsyncOpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url="https://proxy.braintrustapi.com/v1",
            )
        )
    else:
        yield AsyncOpenAI()
