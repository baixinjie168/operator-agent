"""CLI entry point for running tasks from YAML configuration.

Usage:
    python -m agent.run_task                          # use default config
    python -m agent.run_task config/my_tasks.yaml     # use custom config
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.services.task_loader import main


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(config_path))
