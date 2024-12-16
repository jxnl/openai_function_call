from __future__ import annotations

import base64
import imghdr
import mimetypes
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Final, Literal, Optional,
    TypeVar, TypedDict, Union
)
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from .mode import Mode

ImgT = TypeVar('ImgT', bound='Image')

# Constants for Mistral image validation
VALID_MISTRAL_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_MISTRAL_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")  # For generic type hints

CacheControlType = Mapping[str, str]
OptionalCacheControlType = Optional[CacheControlType]

# Type hints for built-in functions and methods
GuessTypeResult = tuple[Optional[str], Optional[str]]
StrSplitResult = list[str]
StrSplitMethod = Callable[[str, Optional[int]], StrSplitResult]


class ImageParamsBase(TypedDict):
    type: Literal["image"]
    source: str


class ImageParams(ImageParamsBase, total=False):
    cache_control: CacheControlType


class Image(BaseModel):
    """Represents an image that can be loaded from a URL or file path."""

    VALID_MIME_TYPES: Final[frozenset[str]] = frozenset({
        "image/jpeg", "image/png", "image/gif", "image/webp"
    })
    VALID_MISTRAL_MIME_TYPES: Final[frozenset[str]] = frozenset({
        "image/jpeg", "image/png", "image/gif", "image/webp"
    })

    source: Union[str, Path] = Field(description="URL or file path of the image")
    media_type: str = Field(description="MIME type of the image")
    data: Optional[str] = Field(
        None, description="Base64 encoded image data", repr=False
    )
    @classmethod
    def autodetect(cls: type[ImgT], source: Union[str, Path]) -> Optional[ImgT]:
        """Attempt to autodetect an image from a source string or Path.

        Args:
            source: URL, file path, or base64 data

        Returns:
            Optional[Image]: An Image instance if detected, None if not a valid image

        Raises:
            ValueError: If unable to determine image type or unsupported format
        """
        try:
            if isinstance(source, str):
                if cls.is_base64(source):
                    result = cls.from_base64(source)
                    return result if isinstance(result, cls) else None
                elif urlparse(source).scheme in {"http", "https"}:
                    result = cls.from_url(source)
                    return result if isinstance(result, cls) else None
                elif Path(source).is_file():
                    result = cls.from_path(source)
                    return result if isinstance(result, cls) else None
                else:
                    result = cls.from_raw_base64(source)
                    return result if isinstance(result, cls) else None
            elif isinstance(source, Path):
                result = cls.from_path(source)
                return result if isinstance(result, cls) else None
            return None
        except Exception:
            return None

    @classmethod
    def autodetect_safely(cls: type[ImgT], source: Union[str, Path]) -> Union[str, ImgT]:
        """Safely attempt to autodetect an image from a source string or path.

        Args:
            source: URL, file path, or base64 data

        Returns:
            Union[str, Image]: An Image instance or the original string if not an image
        """
        try:
            result = cls.autodetect(source)
            return result if result is not None else str(source)
        except ValueError:
            return str(source)

    @classmethod
    def is_base64(cls: type[ImgT], s: str) -> bool:
        return bool(re.match(r"^data:image/[a-zA-Z]+;base64,", s))

    @classmethod
    def from_base64(cls: type[ImgT], data: str) -> ImgT:
        """Create an Image instance from base64 data."""
        if not cls.is_base64(data):
            raise ValueError("Invalid base64 data")

        # Split data URI into header and encoded parts
        parts: list[str] = data.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid base64 data URI format")
        header: str = parts[0]
        encoded: str = parts[1]

        # Extract media type from header
        type_parts: list[str] = header.split(":")
        if len(type_parts) != 2:
            raise ValueError("Invalid base64 data URI header")
        media_type: str = type_parts[1].split(";")[0]

        if media_type not in cls.VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(source=data, media_type=media_type, data=encoded)

    @classmethod
    def from_raw_base64(cls: type[ImgT], data: str) -> Optional[ImgT]:
        """Create an Image from raw base64 data.

        Args:
            data: Raw base64 encoded image data

        Returns:
            Optional[Image]: An Image instance or None if invalid
        """
        try:
            decoded: bytes = base64.b64decode(data)
            img_type: Optional[str] = imghdr.what(None, decoded)
            if img_type:
                media_type = f"image/{img_type}"
                if media_type in cls.VALID_MIME_TYPES:
                    return cls(source=data, media_type=media_type, data=data)
        except Exception:
            pass
        return None

    @classmethod
    @lru_cache
    def from_url(cls: type[ImgT], url: str) -> ImgT:
        """Create an Image instance from a URL.

        Args:
            url: The URL of the image

        Returns:
            Image: An Image instance

        Raises:
            ValueError: If unable to fetch image or unsupported format
        """
        if cls.is_base64(url):
            return cls.from_base64(url)
        parsed_url = urlparse(url)
        media_type: Optional[str] = mimetypes.guess_type(parsed_url.path)[0]

        if not media_type:
            try:
                response = requests.head(url, allow_redirects=True)
                media_type = response.headers.get("Content-Type")
            except requests.RequestException as e:
                raise ValueError("Failed to fetch image from URL") from e

        if media_type not in cls.VALID_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {media_type}")
        return cls(source=url, media_type=media_type, data=None)

    @classmethod
    @lru_cache
    def from_path(cls: type[ImgT], path: Union[str, Path]) -> ImgT:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError("Image file is empty")

        if path.stat().st_size > MAX_MISTRAL_IMAGE_SIZE:
            raise ValueError(
                f"Image file size ({path.stat().st_size / 1024 / 1024:.1f}MB) "
                f"exceeds Mistral's limit of {MAX_MISTRAL_IMAGE_SIZE / 1024 / 1024:.1f}MB"
            )
        media_type: Optional[str] = mimetypes.guess_type(str(path))[0]
        if media_type not in cls.VALID_MIME_TYPES:
            raise ValueError(
                f"Unsupported image format: {media_type}. "
                f"Supported formats are: {', '.join(cls.VALID_MIME_TYPES)}"
            )

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(source=path, media_type=media_type, data=data)

    @staticmethod
    @lru_cache
    def url_to_base64(url: str) -> str:
        """Cachable helper method for getting image url and encoding to base64."""
        response = requests.get(url)
        response.raise_for_status()
        data = base64.b64encode(response.content).decode("utf-8")
        return data

    def to_anthropic(self) -> dict[str, Any]:
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.data
        ):
            self.data = self.url_to_base64(self.source)

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }

    def to_openai(self) -> dict[str, Any]:
        if (
            isinstance(self.source, str)
            and self.source.startswith(("http://", "https://"))
            and not self.is_base64(self.source)
        ):
            return {"type": "image_url", "image_url": {"url": self.source}}
        elif self.data or self.is_base64(str(self.source)):
            data = self.data or str(self.source).split(",", 1)[1]
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{data}"},
            }
        else:
            raise ValueError("Image data is missing for base64 encoding.")

    def to_mistral(self) -> dict[str, Any]:
        """Convert the image to Mistral's format.

        Returns:
            dict[str, Any]: Image in Mistral's format

        Raises:
            ValueError: If image data is missing or format is unsupported
        """
        if not self.data:
            if urlparse(str(self.source)).scheme in {"http", "https"}:
                self.data = self.url_to_base64(str(self.source))
            elif Path(str(self.source)).is_file():
                source_path = Path(str(self.source))
                binary_data = source_path.read_bytes()
                self.data = base64.b64encode(binary_data).decode('utf-8')

        if not self.data:
            raise ValueError("No image data available")

        if self.media_type not in self.VALID_MISTRAL_MIME_TYPES:
            raise ValueError(f"Unsupported image format: {self.media_type}")

        # Ensure data is properly formatted as a data URL
        data_url = (
            self.data if self.data.startswith("data:")
            else f"data:{self.media_type};base64,{self.data}"
        )

        return {
            "type": "image_url",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": data_url
            }
        }

class Audio(BaseModel):
    """Represents an audio that can be loaded from a URL or file path."""

    source: Union[str, Path] = Field(description="URL or file path of the audio")
    data: Union[str, None] = Field(
        None, description="Base64 encoded audio data", repr=False
    )


class ImageWithCacheControl(Image):
    """Image with Anthropic prompt caching support."""

    cache_control: OptionalCacheControlType = Field(
        None, description="Optional Anthropic cache control image"
    )

    @classmethod
    def from_image_params(
        cls, source: Union[str, Path], image_params: dict[str, Any]
    ) -> Union[ImageWithCacheControl, None]:
        """Create an ImageWithCacheControl from image parameters.

        Args:
            source: The image source
            image_params: Dictionary containing image parameters

        Returns:
            Optional[ImageWithCacheControl]: An ImageWithCacheControl instance if valid
        """
        cache_control = image_params.get("cache_control")
        base_image = Image.autodetect(source)
        if base_image is None:
            return None

        return cls(
            source=base_image.source,
            media_type=base_image.media_type,
            data=base_image.data,
            cache_control=cache_control,
        )

    def to_anthropic(self) -> dict[str, Any]:
        """Override Anthropic return with cache_control."""
        result = super().to_anthropic()
        if self.cache_control:
            result["cache_control"] = self.cache_control
        return result


def convert_contents(
    contents: Union[
        str, Image, dict[str, Any], list[Union[str, Image, dict[str, Any]]]
    ],
    mode: Mode,
) -> Union[str, list[dict[str, Any]]]:
    """Convert contents to the appropriate format for the given mode."""
    # Handle single string case
    if isinstance(contents, str):
        return contents

    # Handle single image case
    if isinstance(contents, Image):
        if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
            return [contents.to_anthropic()]
        elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
            raise NotImplementedError("Gemini is not supported yet")
        elif mode in {Mode.MISTRAL_JSON, Mode.MISTRAL_TOOLS}:
            return [contents.to_mistral()]
        else:
            return [contents.to_openai()]

    # Handle single dict case
    if isinstance(contents, dict):
        return [contents]

    # Handle list case
    converted_contents: list[dict[str, Any]] = []
    for content in contents:
        if isinstance(content, str):
            converted_contents.append({"type": "text", "text": content})
        elif isinstance(content, Image):
            if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
                converted_contents.append(content.to_anthropic())
            elif mode in {Mode.GEMINI_JSON, Mode.GEMINI_TOOLS}:
                raise NotImplementedError("Gemini is not supported yet")
            elif mode in {Mode.MISTRAL_JSON, Mode.MISTRAL_TOOLS}:
                converted_contents.append(content.to_mistral())
            else:
                converted_contents.append(content.to_openai())
        elif isinstance(content, dict):
            converted_contents.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return converted_contents


def convert_messages(
    messages: list[dict[str, Any]],
    mode: Mode,
) -> list[dict[str, Any]]:
    """Convert messages to the appropriate format for the given mode.

    Args:
        messages: List of message dictionaries to convert
        mode: The mode to convert messages for (e.g. MISTRAL_JSON)

    Returns:
        List of converted message dictionaries
    """
    converted_messages: list[dict[str, Any]] = []
    for message in messages:
        converted_message = message.copy()
        content = message.get("content")

        # Handle string content
        if isinstance(content, str):
            converted_message["content"] = content
            converted_messages.append(converted_message)
            continue

        # Handle Image content
        if isinstance(content, Image):
            converted_message["content"] = convert_contents(content, mode)
            converted_messages.append(converted_message)
            continue

        # Handle list content
        if isinstance(content, list):
            # Explicitly type the content as Union[str, Image, dict[str, Any]]
            typed_content: list[Union[str, Image, dict[str, Any]]] = content
            converted_message["content"] = convert_contents(typed_content, mode)
            converted_messages.append(converted_message)
            continue

        # Handle other content types
        converted_messages.append(converted_message)

    return converted_messages
