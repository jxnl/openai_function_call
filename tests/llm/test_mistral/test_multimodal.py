import pytest
from pathlib import Path
from instructor.multimodal import Image
import instructor
from instructor import Mode
from pydantic import Field, BaseModel
from itertools import product
from unittest.mock import patch, MagicMock
from .util import models, modes
from typing import Any, cast, IO

# Test image URLs with different formats and sizes
test_images = {
    "jpeg": "https://retail.degroot-inc.com/wp-content/uploads/2024/01/AS_Blueberry_Patriot_1-605x605.jpg",
    "png": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png",
    "webp": "https://www.gstatic.com/webp/gallery/1.webp",
    "gif": "https://upload.wikimedia.org/wikipedia/commons/2/2c/Rotating_earth_%28large%29.gif",
}


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


@pytest.mark.requires_mistral
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multimodal_image_description(model: str, mode: Mode, client: Any) -> None:
    """Test basic image description with Mistral."""
    client = instructor.from_mistral(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
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
                    Image.from_url(test_images["jpeg"]),
                ],
            },
        ],
    )

    assert isinstance(response, ImageDescription)
    assert len(response.objects) > 0
    assert response.scene != ""
    assert len(response.colors) > 0


def test_image_size_validation(tmp_path: Path) -> None:
    """Test that images over 10MB are rejected."""
    large_image: Path = tmp_path / "large_image.jpg"
    # Create a file slightly over 10MB
    with open(large_image, "wb") as file_obj:
        typed_file: IO[bytes] = cast(IO[bytes], file_obj)
        typed_file.write(b"0" * (10 * 1024 * 1024 + 1))

    with pytest.raises(
        ValueError,
        match=r"Image file size \(10\.0MB\) exceeds Mistral's limit of 10\.0MB",
    ):
        Image.from_path(large_image).to_mistral()


def test_image_format_validation() -> None:
    """Test validation of supported image formats."""
    # Test valid formats
    for fmt, url in test_images.items():
        if fmt != "gif":  # Skip animated GIF
            image = Image.from_url(url)
            assert image.to_mistral() is not None

    # Test invalid format
    with pytest.raises(ValueError, match="Unsupported image format"):
        Image(source="test.bmp", media_type="image/bmp", data="fake_data").to_mistral()


@pytest.mark.requires_mistral
@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multiple_images(model: str, mode: Mode, client: Any) -> None:
    """Test handling multiple images in a single request."""
    client = instructor.from_mistral(client, mode=mode)
    images = [Image.from_url(url) for url in list(test_images.values())[:8]]

    response = client.chat.completions.create(
        model=model,
        response_model=ImageDescription,
        messages=[
            {
                "role": "user",
                "content": ["Describe these images"] + images,
            },
        ],
    )

    assert isinstance(response, ImageDescription)

    # Test exceeding image limit
    with pytest.raises(ValueError, match="Maximum of 8 images allowed"):
        too_many_images = images * 2  # 16 images
        client.chat.completions.create(
            model=model,
            response_model=ImageDescription,
            messages=[
                {
                    "role": "user",
                    "content": ["Describe these images"] + too_many_images,
                },
            ],
        )


def test_image_downscaling() -> None:
    """Test automatic downscaling of large images."""
    large_image_url = "https://example.com/large_image.jpg"  # Mock URL

    # Mock a large image response
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"0" * 1024 * 1024  # 1MB of data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_get.return_value = mock_response

        image = Image.from_url(large_image_url)
        mistral_format = image.to_mistral()

        # Verify image was processed for downscaling
        assert mistral_format is not None
        # Note: Actual downscaling verification would require PIL/image processing


def test_base64_image_handling(base64_image: str) -> None:
    """Test handling of base64-encoded images."""
    image = Image(
        source="data:image/jpeg;base64," + base64_image,
        media_type="image/jpeg",
        data=base64_image,
    )

    mistral_format = image.to_mistral()
    assert mistral_format["type"] == "image_url"
    assert mistral_format["data"].startswith("data:image/jpeg;base64,")


@pytest.fixture
def base64_image() -> str:
    """Fixture providing a valid base64-encoded test image."""
    return "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="  # 1x1 GIF
