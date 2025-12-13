import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator


# Basic Op/Event types to mirror codex-rs pattern
@dataclass
class Op:
    session_id: Optional[str]
    kind: str  # e.g., "user_message"
    payload: Dict[str, Any]


@dataclass
class Event:
    session_id: str
    kind: str  # e.g., "reply", "status", "tool_call", "error"
    payload: Dict[str, Any]


class Dispatcher:
    def __init__(self, handler):
        self.handler = handler  # async function (op) -> List[Event]
        self.queue: asyncio.Queue[Op] = asyncio.Queue()
        self.listeners: Dict[str, List[asyncio.Queue[Event]]] = {}
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(Exception):
                await self.task
            self.task = None

    async def submit(self, op: Op):
        await self.queue.put(op)

    async def events(self, session_id: str) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        self.listeners.setdefault(session_id, []).append(q)
        try:
            while True:
                evt = await q.get()
                yield evt
        finally:
            self.listeners[session_id].remove(q)
            if not self.listeners[session_id]:
                del self.listeners[session_id]

    async def _run(self):
        while True:
            op = await self.queue.get()
            try:
                events = await self.handler(op)
                for evt in events:
                    for q in self.listeners.get(evt.session_id, []):
                        await q.put(evt)
            except Exception as exc:
                if op.session_id:
                    err_evt = Event(op.session_id, "error", {"message": str(exc)})
                    for q in self.listeners.get(op.session_id, []):
                        await q.put(err_evt)

