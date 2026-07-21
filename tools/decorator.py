from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, Union
import types


_PYTHON_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(get_args(annotation)) - len(args) == 1:
            return True, args[0] if len(args) == 1 else Union[tuple(args)]
    if get_origin(annotation) is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(get_args(annotation)) - len(args) == 1:
            return True, args[0] if len(args) == 1 else Union[tuple(args)]
    return False, annotation


def _annotation_to_json_type(annotation: Any) -> str:
    optional, inner = _is_optional(annotation)
    annotation = inner

    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return _PYTHON_TO_JSON.get(annotation, "string")


@dataclass
class ToolSchema:
    name: str
    description: str
    kind: str
    parameters: dict
    fn: Callable
    source_file: str = ""   # absolute path to the .py file containing fn


def tool(name: str, description: str, kind: str = "action") -> Callable:
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                json_type = "string"
                is_opt = False
            else:
                is_opt, _ = _is_optional(annotation)
                json_type = _annotation_to_json_type(annotation)

            properties[param_name] = {"type": json_type}

            has_default = param.default is not inspect.Parameter.empty
            if not is_opt and not has_default:
                required.append(param_name)

        schema_dict: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        import os
        source = os.path.abspath(inspect.getfile(fn)) if hasattr(fn, "__code__") else ""
        fn.schema = ToolSchema(
            name=name,
            description=description,
            kind=kind,
            parameters=schema_dict,
            fn=fn,
            source_file=source,
        )
        return fn

    return decorator


def auto_schema(fn: Callable, *, kind: str = "action") -> ToolSchema:
    """Auto-generate a ToolSchema for an undecorated function using its name and type hints."""
    import os

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            json_type = "string"
            is_opt = False
        else:
            is_opt, _ = _is_optional(annotation)
            json_type = _annotation_to_json_type(annotation)
        properties[param_name] = {"type": json_type}
        if not is_opt and param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema_dict: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (fn.__doc__ or fn.__name__).split("\n")[0].strip(),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
    source = os.path.abspath(inspect.getfile(fn)) if hasattr(fn, "__code__") else ""
    return ToolSchema(
        name=fn.__name__,
        description=(fn.__doc__ or fn.__name__).split("\n")[0].strip(),
        kind=kind,
        parameters=schema_dict,
        fn=fn,
        source_file=source,
    )
