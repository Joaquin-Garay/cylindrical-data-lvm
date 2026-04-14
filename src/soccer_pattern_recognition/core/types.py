"""Shared typing aliases."""

import numpy.typing

try:  # Python >=3.10
    from typing import TypeAlias
except ImportError:  # pragma: no cover - Python <3.10 fallback
    TypeAlias = type  # type: ignore[assignment]

Array: TypeAlias = numpy.typing.NDArray
