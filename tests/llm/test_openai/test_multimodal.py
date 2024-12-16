import pytest
from instructor.multimodal import Image
import instructor
from instructor import Mode
from pydantic import Field, BaseModel
from itertools import product
from .util import models, modes
from typing import Any

image_url = "https://retail.degroot-inc.com/wp-content/uploads/2024/01/AS_Blueberry_Patriot_1-605x605.jpg"


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


@pytest.mark.requires_openai
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model: str, mode: Mode, client: Any) -> None:
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    Image.from_url(image_url),
                ],
            },
        ],
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0


@pytest.mark.requires_openai
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect(
    model: str, mode: Mode, client: Any
) -> None:
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,  # Ensure this is a vision-capable model
        response_model=ImageDescription,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images",
            },
            {
                "role": "user",
                "content": [
                    "What is this?",
                    image_url,
                ],
            },
        ],
        autodetect_images=True,
    )

    # Assertions to validate the response
    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0


@pytest.mark.requires_openai
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description_autodetect_no_response_model(
    model: str, mode: Mode, client: Any
) -> None:
    client = instructor.from_openai(client, mode=mode)
    response = client.chat.completions.create(
        response_model=None,
        model=model,  # Ensure this is a vision-capable model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can describe images. "
                "If looking at an image, reply with 'This is an image' and nothing else.",
            },
            {
                "role": "user",
                "content": image_url,
            },
        ],
        max_tokens=1000,
        temperature=1,
        autodetect_images=True,
    )

    assert response.choices[0].message.content.startswith("This is an image")
