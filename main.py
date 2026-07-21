"""Entry point: load config, initialise all subsystems, start the server."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _available_ram_mb() -> int:
    """Return available RAM in MB, or 0 if it cannot be determined."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024  # kB → MB
    except OSError:
        pass
    return 0


def _cap_max_subagents(config: dict[str, Any]) -> None:
    """Warn and auto-cap max_subagents if it exceeds what this machine can safely handle."""
    requested: int = config.get("max_subagents", 4)
    cpu_cores: int = os.cpu_count() or 1
    ram_mb: int = _available_ram_mb()

    # Leave 1 core free for the coordinator + OS overhead.
    cpu_limit: int = max(1, cpu_cores - 1)

    # Each subagent spawns subprocess tool calls; budget ~256 MB of headroom per agent.
    # If RAM is unknown (0), skip the RAM cap.
    ram_limit: int = max(1, ram_mb // 256) if ram_mb > 0 else requested

    safe: int = min(cpu_limit, ram_limit)

    ram_display = f"{ram_mb} MB" if ram_mb > 0 else "unknown"
    if requested > safe:
        logger.warning(
            "max_subagents=%d may exceed this machine's capacity "
            "(CPU cores=%d, available RAM=%s). "
            "Auto-capping to %d to prevent resource exhaustion.",
            requested, cpu_cores, ram_display, safe,
        )
        config["max_subagents"] = safe
    else:
        logger.info(
            "Resource check OK — max_subagents=%d, CPU cores=%d, available RAM=%s",
            requested, cpu_cores, ram_display,
        )


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.error("config.yaml not found at %s", cfg_path.absolute())
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    target = cfg.get("target_repo", "")
    if not target or target == "/path/to/your/repo":
        logger.error(
            "target_repo is not set in config.yaml. "
            "Edit config.yaml and set target_repo to the absolute path of the repo to improve."
        )
        sys.exit(1)
    if not Path(target).exists():
        logger.error("target_repo path does not exist: %s", target)
        sys.exit(1)

    models = cfg.get("models", {})
    if not models.get("coordinator"):
        logger.error("models.coordinator is required in config.yaml")
        sys.exit(1)

    _cap_max_subagents(cfg)
    return cfg


async def startup(config: dict[str, Any]) -> None:
    """Initialise all subsystems and register with the FastAPI app."""
    from memory import Memory
    from models.client import OllamaClient
    from models.registry import ModelRegistry
    from models.router import ModelRouter
    from tools.runtime import ToolRuntime
    from coordinator.coordinator import Coordinator
    from server.app import app, set_coordinator

    ollama_host: str = config.get("ollama_host", "http://localhost:11434")
    data_dir: str = config.get("data_dir", "./data")
    embed_model: str = config.get("models", {}).get("embed", "nomic-embed-text")

    # ── Memory ────────────────────────────────────────────────────────────────
    logger.info("Initialising memory (data_dir=%s)...", data_dir)
    memory = Memory(
        data_dir=data_dir,
        ollama_host=ollama_host,
        embed_model=embed_model,
        dup_threshold=config.get("dup_threshold", 0.92),
    )
    await memory.init()

    # ── Model registry ────────────────────────────────────────────────────────
    logger.info("Loading model registry...")
    registry = ModelRegistry(config)
    logger.info("Checking model availability in Ollama at %s...", ollama_host)
    try:
        await registry.validate(ollama_host)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info(
        "Models OK — coordinator=%s  default=%s  workers=%d",
        registry.coordinator.name,
        registry.default.name,
        len(registry.workers),
    )

    router = ModelRouter(registry)
    client = OllamaClient(base_url=ollama_host, registry=registry)

    # ── Tool runtime ──────────────────────────────────────────────────────────
    logger.info("Loading tools...")
    from tools.save_tool import save_to_github

    tools = ToolRuntime(config)
    tools.register(save_to_github)   # built-in save tool
    tools.auto_discover("user_tools")
    logger.info("Registered tools: %s", list(tools._tools.keys()))

    # ── Coordinator ───────────────────────────────────────────────────────────
    coordinator = Coordinator(
        config=config,
        memory=memory,
        client=client,
        router=router,
        tools=tools,
    )
    set_coordinator(coordinator)
    logger.info("Coordinator ready. Visit http://%s:%s to open the dashboard.",
                config.get("server", {}).get("host", "0.0.0.0"),
                config.get("server", {}).get("port", 8000))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Researcher autonomous improvement agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--autostart", action="store_true",
                        help="Start the improvement loop automatically on launch")
    args = parser.parse_args()

    config = load_config(args.config)

    # Run FastAPI with uvicorn; startup hook initialises subsystems
    from server.app import app

    @app.on_event("startup")
    async def _startup() -> None:
        await startup(config)
        if args.autostart:
            import server.app as server_app
            if server_app._coordinator is not None:
                asyncio.create_task(server_app._coordinator.run())

    server_cfg = config.get("server", {})
    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
