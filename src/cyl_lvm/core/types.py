"""Shared typing aliases."""

from typing import Any, TypeAlias

import numpy.typing as npt

# Convention:
# - Array is the default annotation for NumPy arrays used by the library.
# - ArrayLike is reserved for boundary APIs whose contract explicitly accepts
#   lists, tuples, or other inputs that are coerced with np.asarray.
Array: TypeAlias = npt.NDArray[Any]
ArrayLike: TypeAlias = npt.ArrayLike
