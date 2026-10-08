"""
Shared ``to_dict()`` support for analysis result dataclasses.

Every result in :mod:`sidescanpy.analysis` is a dataclass mixing in
:class:`ResultMixin`. ``to_dict()`` returns a JSON-serializable dict:
numpy scalars become Python numbers, arrays and pandas objects become
lists (a DataFrame becomes ``{column: [values]}``), nested dataclasses
become dicts. Fields declared with :func:`array_field` (large per-ping
arrays, fit objects) are left out.
"""

import dataclasses
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SERIALIZE = "serialize"


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-serializable Python types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_dict(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return {str(c): to_jsonable(value[c].to_numpy()) for c in value.columns}
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_numpy())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {
        f.name: to_jsonable(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.metadata.get(_SERIALIZE, True)
    }


def array_field(default: Any = None) -> Any:
    """A dataclass field excluded from ``to_dict()`` and ``repr``."""
    return dataclasses.field(default=default, repr=False, compare=False, metadata={_SERIALIZE: False})


class ResultMixin:
    """Adds ``to_dict()`` to a result dataclass."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict of this result's fields."""
        return _dataclass_to_dict(self)
