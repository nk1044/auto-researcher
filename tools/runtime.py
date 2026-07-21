from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tools.decorator import ToolSchema

logger = logging.getLogger(__name__)

_RESTRICTED_KINDS = frozenset({"test", "save"})


@dataclass
class ToolResult:
    success: bool
    value: Any = None
    error: Optional[str] = None


class ToolRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._tools: dict[str, ToolSchema] = {}

    def register(self, fn: Callable) -> None:
        schema: ToolSchema = getattr(fn, "schema", None)
        if schema is None:
            raise ValueError(f"{fn!r} is not decorated with @tool")
        self._tools[schema.name] = schema
        logger.debug("registered tool %r (kind=%s)", schema.name, schema.kind)

    def auto_discover(self, directory: str) -> None:
        from tools.decorator import auto_schema

        base = Path(directory)
        for py_file in sorted(base.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"_auto_discover_{py_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # needed for sandbox reimport
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                logger.warning("could not import %s: %s", py_file, exc)
                continue
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if not callable(obj) or attr_name.startswith("_"):
                    continue
                if getattr(obj, "__module__", None) != module_name:
                    continue  # skip imported names
                if hasattr(obj, "schema") and isinstance(obj.schema, ToolSchema):
                    self.register(obj)
                elif callable(obj) and getattr(obj, "__code__", None) is not None:
                    # Auto-generate schema for undecorated public functions
                    obj.schema = auto_schema(obj, kind="action")
                    self.register(obj)

    def get_schemas(self, kinds: Optional[list[str]] = None) -> list[dict]:
        schemas = []
        for schema in self._tools.values():
            if kinds is None or schema.kind in kinds:
                schemas.append(schema.parameters)
        return schemas

    async def call(
        self,
        name: str,
        caller: str = "coordinator",
        **kwargs: Any,
    ) -> ToolResult:
        schema = self._tools.get(name)
        if schema is None:
            return ToolResult(success=False, error=f"unknown tool: {name!r}")

        if schema.kind in _RESTRICTED_KINDS and caller != "coordinator":
            return ToolResult(
                success=False,
                error="Restricted: test/save are coordinator-only",
            )

        fn = schema.fn
        source_file = schema.source_file
        fn_name = fn.__qualname__

        script = self._build_sandbox_script(name, source_file, fn_name)

        cpu_limit = self._config.get("sandbox", {}).get("max_cpu_seconds", 120)
        mem_limit_mb = self._config.get("sandbox", {}).get("max_memory_mb", 512)
        timeout = self._config.get("sandbox", {}).get("timeout_seconds", 300)

        env_inject = json.dumps(
            {
                "cpu": cpu_limit,
                "mem_mb": mem_limit_mb,
                "kwargs": kwargs,
            }
        )

        try:
            result = await asyncio.wait_for(
                self._run_subprocess(script, env_inject, fn),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"tool {name!r} timed out after {timeout}s")
        except Exception as exc:
            logger.exception("unexpected error running tool %r", name)
            return ToolResult(success=False, error=str(exc))

    async def _run_subprocess(
        self,
        script: str,
        payload: str,
        fn: Callable,
    ) -> ToolResult:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=payload.encode())
        if stderr:
            logger.debug("sandbox stderr: %s", stderr.decode().strip())
        if not stdout.strip():
            return ToolResult(success=False, error="sandbox produced no output")
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"sandbox output not valid JSON: {exc}")
        return ToolResult(
            success=data.get("success", False),
            value=data.get("value"),
            error=data.get("error"),
        )

    def _build_sandbox_script(self, name: str, source_file: str, fn_name: str) -> str:
        project_root = str(Path(__file__).parent.parent)
        return textwrap.dedent(
            f"""\
            import json, sys, resource, importlib.util, asyncio

            sys.path.insert(0, {project_root!r})

            raw = sys.stdin.read()
            payload = json.loads(raw)

            cpu_limit = int(payload.get("cpu", 120))
            mem_mb = int(payload.get("mem_mb", 512))
            kwargs = payload.get("kwargs", {{}})

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            mem_bytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

            try:
                source_file = {source_file!r}
                fn_qualname = {fn_name!r}

                spec = importlib.util.spec_from_file_location("_tool_module", source_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                parts = fn_qualname.split(".")
                obj = mod
                for part in parts:
                    obj = getattr(obj, part)
                fn = obj

                if asyncio.iscoroutinefunction(fn):
                    result = asyncio.run(fn(**kwargs))
                else:
                    result = fn(**kwargs)

                print(json.dumps({{"success": True, "value": result}}))
            except Exception as exc:
                print(json.dumps({{"success": False, "error": str(exc)}}))
            """
        )
