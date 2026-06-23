import json
from typing import Any


try:
    import orjson
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    orjson = None


def loads(value: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(value)
    return json.loads(value)


def load_file(path: str) -> Any:
    if orjson is not None:
        with open(path, "rb") as handle:
            return orjson.loads(handle.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dumps(value: Any, *, ensure_ascii: bool = False, indent: int | None = None, sort_keys: bool = False) -> str:
    if orjson is not None:
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS
        return orjson.dumps(value, option=option).decode("utf-8")
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)


def dump_file(value: Any, path: str, *, ensure_ascii: bool = False, indent: int | None = None) -> None:
    if orjson is not None:
        option = orjson.OPT_INDENT_2 if indent else 0
        with open(path, "wb") as handle:
            handle.write(orjson.dumps(value, option=option))
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=ensure_ascii, indent=indent)
