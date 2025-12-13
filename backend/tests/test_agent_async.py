import asyncio
import json
import types
import sys
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test")

from agent import run_agent_turn_async  # type: ignore
from session import SessionState  # type: ignore


class FakeResponse:
    def __init__(self, output, input_tokens=10, output_tokens=5):
        self.output = output
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class FakeAsyncClient:
    def __init__(self, response):
        self._response = response
        self.responses = self

    async def responses_create(self, **kwargs):
        # Not used; kept for compatibility if called directly
        return self._response

    async def responses(self):
        return self

    async def create(self, **kwargs):
        # Simulate streaming if requested
        if kwargs.get("stream"):
            async def agen():
                chunk_running = types.SimpleNamespace(
                    model_dump=lambda: {
                        "status": "in_progress",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "text", "text": "partial"}],
                            }
                        ],
                    }
                )
                chunk_done = self._response
                yield chunk_running
                yield chunk_done

            return agen()
        return self._response


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_run_agent_turn_async_message_only(monkeypatch):
    output = [
        {"type": "message", "content": [{"type": "text", "text": "hello"}]},
    ]
    fake_resp = FakeResponse(output)
    fake_client = FakeAsyncClient(fake_resp)
    monkeypatch.setattr("agent.async_client", fake_client)

    session = SessionState()
    events = [ev async for ev in run_agent_turn_async(session, "hi")]

    reply_events = [e for e in events if e.get("type") == "reply"]
    assert reply_events, "Expected a reply event"
    assert reply_events[0]["text"] == "hello"


os.environ.setdefault("OPENAI_API_KEY", "test")
