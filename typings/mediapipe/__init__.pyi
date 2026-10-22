from enum import IntEnum
from typing import Any

from . import tasks as tasks


class ImageFormat(IntEnum):
    UNKNOWN: ImageFormat
    SRGB: ImageFormat
    SRGBA: ImageFormat
    GRAY8: ImageFormat
    GRAY16: ImageFormat
    SRGB48: ImageFormat
    SRGBA64: ImageFormat
    VEC32F1: ImageFormat
    VEC32F2: ImageFormat
    VEC32F4: ImageFormat


class Image:
    def __init__(self, image_format: ImageFormat, data: Any) -> None: ...
