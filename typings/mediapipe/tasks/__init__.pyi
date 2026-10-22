from enum import IntEnum
from typing import Optional

from . import vision as vision


class BaseOptions:
    class Delegate(IntEnum):
        CPU: Delegate
        GPU: Delegate

    model_asset_path: Optional[str]
    model_asset_buffer: Optional[bytes]
    delegate: Optional[Delegate]

    def __init__(
        self,
        model_asset_path: Optional[str] = ...,
        model_asset_buffer: Optional[bytes] = ...,
        delegate: Optional[Delegate] = ...,
    ) -> None: ...
