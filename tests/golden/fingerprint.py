"""Stable hashing of plain observation data. Never change this module: the committed trace was
recorded with it."""
import hashlib
import json
import math


def _canonical(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    if hasattr(value, "item"):
        return _canonical(value.item())
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def digest(value):
    text = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def fingerprint(observation):
    """{field: digest} for one observation dict."""
    return {field: digest(value) for field, value in sorted(observation.items())}
