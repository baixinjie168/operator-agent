"""EventBus — asyncio.Queue pub/sub per run_id.

The single exit point for all RuntimeEvents.  SSE consumes from here.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class EventBus:
    def __init__(self, max_queue_size: int = 256) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._max_size = max_queue_size

    def publish(self, run_id: str, data: dict) -> None:
        for q in self._subscribers.get(run_id, []):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_size)
        self._subscribers[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        lst = self._subscribers.get(run_id, [])
        try:
            lst.remove(q)
        except ValueError:
            pass
        if not lst:
            self._subscribers.pop(run_id, None)
