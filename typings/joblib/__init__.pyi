import os
from typing import Any


def load(
    filename: str | os.PathLike[str],
    mmap_mode: str | None = ...,
    ensure_native_byte_order: str = ...,
) -> Any: ...
