"""Task loader: create and run tasks from YAML configuration files.

Usage:
    python -m agent.run_task                          # use default config
    python -m agent.run_task config/my_tasks.yaml     # use custom config
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.services.task_engine import run_task
from agent.utils.file_utils import extract_operator_name_from_file as _extract_operator_name

logger = logging.getLogger(__name__)


def load_task_config(config_path: str | Path) -> dict:
    """Load and validate a YAML task configuration file.

    Returns the parsed config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config structure is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    return config


async def create_and_run_from_config(
    config_path: str | Path,
) -> list[dict]:
    """Load tasks from YAML config, create them in DB, and start execution.

    Returns a list of {"task_id", "name", "total_count"} for each created task.
    """
    config = load_task_config(config_path)
    mcp = MCPClient()

    # Apply parallel settings from config (override settings if specified)
    parallel_cfg = config.get("parallel", {})
    max_workers = parallel_cfg.get("max_workers", settings.task_max_workers)
    logger.info("Parallel config: max_workers=%d", max_workers)

    tasks_cfg = config.get("tasks", [])
    if not tasks_cfg:
        logger.warning("No tasks defined in config file")
        return []

    ops_base = Path(settings.operators_dir).parent  # project root
    created_tasks: list[dict] = []

    for task_cfg in tasks_cfg:
        task_name = task_cfg.get("name", f"config-task-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        file_paths = task_cfg.get("file_paths", [])

        if not file_paths:
            logger.warning("Task '%s' has no file_paths, skipping", task_name)
            continue

        # Create upload directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir = Path(f"uploads/{timestamp}")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Copy files and build items
        items = []
        for seq, fp in enumerate(file_paths, start=1):
            src = ops_base / fp
            if not src.exists():
                logger.warning("File not found: %s", src)
                continue
            dst = upload_dir / src.name
            shutil.copy2(str(src), str(dst))
            operator_name = _extract_operator_name(src)
            items.append({
                "seq": seq,
                "operator_name": operator_name,
                "file_path": str(upload_dir / src.name),
            })

        if not items:
            logger.warning("Task '%s': no valid files found, skipping", task_name)
            continue

        # Create task in DB
        result = await mcp.create_task(task_name, len(items), str(upload_dir))
        task_id = result["task_id"]
        await mcp.create_task_items(task_id, items)

        logger.info(
            "Created task %d (%s) with %d items from config",
            task_id, task_name, len(items),
        )

        # Start background execution
        asyncio.create_task(run_task(task_id))

        created_tasks.append({
            "task_id": task_id,
            "name": task_name,
            "total_count": len(items),
        })

    return created_tasks


async def main(config_path: str | None = None) -> None:
    """CLI entry point: load config, create tasks, wait for completion."""
    path = config_path or settings.task_config_file
    logger.info("Loading task config from: %s", path)

    created = await create_and_run_from_config(path)
    if not created:
        logger.warning("No tasks created from config")
        return

    logger.info("Created %d task(s):", len(created))
    for t in created:
        logger.info("  Task %d: %s (%d operators)", t["task_id"], t["name"], t["total_count"])

    # Wait for all tasks to complete
    mcp = MCPClient()
    while True:
        all_done = True
        for t in created:
            task = await mcp.get_task(t["task_id"])
            if task and task["status"] == "running":
                all_done = False
                break
        if all_done:
            break
        await asyncio.sleep(5)

    logger.info("All tasks completed")
    for t in created:
        task = await mcp.get_task(t["task_id"])
        if task:
            logger.info(
                "  Task %d (%s): %s — completed=%d, failed=%d",
                t["task_id"], t["name"], task["status"],
                task["completed_count"], task["failed_count"],
            )
