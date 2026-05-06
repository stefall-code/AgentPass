from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi.websockets import WebSocket

logger = logging.getLogger("agent_system")


class ConnectionManager:
    def __init__(self):
        self._connections: dict[WebSocket, Optional[str]] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=1024)
        self._consumer_task: asyncio.Task | None = None
        self.last_activity: float = 0.0

    async def add(self, websocket: WebSocket, agent_id: Optional[str] = None):
        async with self._lock:
            self._connections[websocket] = agent_id
        self.last_activity = time.time()

    async def remove(self, websocket: WebSocket):
        async with self._lock:
            self._connections.pop(websocket, None)

    def touch(self):
        self.last_activity = time.time()

    async def broadcast(self, message: str):
        async with self._lock:
            dead = []
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections.pop(ws, None)

    async def broadcast_json(self, data: dict):
        await self.broadcast(json.dumps(data, ensure_ascii=False))

    def emit_audit(self, event: dict):
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("audit event queue full, dropping event: action=%s agent=%s", event.get("action"), event.get("agent_id"))
        except Exception as e:
            logger.warning("emit_audit failed: %s", e)

    async def broadcast_audit(self, event: dict):
        await self.broadcast(json.dumps(event, ensure_ascii=False))

    async def _consume_loop(self):
        while True:
            try:
                event = await self._queue.get()
                if event is None:
                    break
                try:
                    await self.broadcast(json.dumps(event, ensure_ascii=False))
                except Exception:
                    logger.exception("failed to broadcast audit event")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("audit consume loop error, continuing")
                await asyncio.sleep(0.1)

    def start_consumer(self):
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.ensure_future(self._consume_loop())
            logger.info("WebSocket audit consumer started")

    async def stop_consumer(self):
        await self._queue.put(None)
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

    @property
    def count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()
