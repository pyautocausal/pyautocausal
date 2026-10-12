"""Conversion to interoperable JSON (non-finite numbers are rejected)."""
from typing import Any
from datetime import date, datetime, timedelta
import json
import math
import numpy as np
import pandas as pd
from ..persistence.output_types import OutputType


def jsonify(obj: Any) -> Any:
    """Convert containers, NumPy values and model summaries to JSON values.

    NaN and infinities are rejected instead of writing non-standard JSON or
    silently replacing an undefined estimate. Callers may explicitly use None
    for unavailable values and record the reason in accompanying metadata.
    Dates and durations use ISO strings. Pandas objects retain the existing
    column/index mapping format and must have unique labels in that format.
    """
    if obj is pd.NA or obj is pd.NaT:
        raise ValueError("Missing pandas values cannot be saved as JSON; use None with an explicit reason")
    if isinstance(obj, np.datetime64):
        return jsonify(pd.Timestamp(obj))
    if isinstance(obj, np.timedelta64):
        return jsonify(pd.Timedelta(obj))
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, timedelta)):
        return pd.Timedelta(obj).isoformat()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if not obj.index.is_unique:
            raise ValueError("JSON pandas output requires a unique index to avoid losing rows")
        if isinstance(obj, pd.DataFrame) and not obj.columns.is_unique:
            raise ValueError("JSON DataFrame output requires unique columns to avoid losing values")
        return jsonify(obj.to_dict())
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'Mm':
            # tolist() turns nanosecond timestamps into unlabeled integers.
            return jsonify(obj[()]) if obj.ndim == 0 else [jsonify(value) for value in obj]
        return jsonify(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonify(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("Non-finite numbers cannot be saved as JSON; use None with an explicit reason")
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        result = {}
        serialized_keys = set()
        for key, value in obj.items():
            if isinstance(key, (datetime, date, np.datetime64)):
                key = jsonify(key)
            if isinstance(key, np.generic):
                key = key.item()
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(f"Unsupported JSON key type: {type(key).__name__}")
            if isinstance(key, float) and not math.isfinite(key):
                raise ValueError("Non-finite JSON dictionary key")
            # JSON object keys are strings. Distinct Python keys such as 1 and
            # '1' must not overwrite one another after the file is read back.
            serialized_key = key if isinstance(key, str) else json.dumps(key, allow_nan=False)
            if serialized_key in serialized_keys:
                raise ValueError(f"Dictionary keys collide in JSON: {key!r}")
            serialized_keys.add(serialized_key)
            result[key] = jsonify(value)
        return result
    if isinstance(obj, (list, tuple)):
        return [jsonify(item) for item in obj]
    if hasattr(obj, 'to_dict'):
        return jsonify(obj.to_dict())
    if hasattr(obj, '__dict__'):
        return jsonify(obj.__dict__)
    raise TypeError(f"Unsupported JSON value type: {type(obj).__name__}")


def prepare_output_for_saving(output: Any, output_type: OutputType) -> Any:
    return jsonify(output) if output_type == OutputType.JSON else output
