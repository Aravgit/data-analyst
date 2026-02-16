import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test")

import main  # type: ignore
from session import SessionStore, SessionState  # type: ignore
from agent import system_prompt  # type: ignore


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client(tmp_path, monkeypatch):
    # Isolate each test from global process state.
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path, raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "store", SessionStore(max_sessions=50, ttl_seconds=3600), raising=False)
    monkeypatch.setattr(main, "SESSION_DATA_BYTES_LIMIT", 2_147_483_648, raising=False)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _upload_csv(
    client: httpx.AsyncClient,
    filename: str,
    content: bytes,
    session_id: str | None = None,
    mode: str | None = None,
):
    data = {}
    if session_id:
        data["session_id"] = session_id
    if mode:
        data["mode"] = mode
    files = {"file": (filename, content, "text/csv")}
    return await client.post("/upload", data=data, files=files)


async def test_upload_append_default_and_dataset_collision(api_client):
    first = await _upload_csv(api_client, "sales.csv", b"date,revenue\n2026-01-01,10\n")
    assert first.status_code == 200
    first_body = first.json()
    sid = first_body["session_id"]
    assert first_body["csv_name"] == "sales"
    assert first_body["datasets"] == ["sales"]

    # Default mode is append. Same filename should be collision-resolved.
    second = await _upload_csv(api_client, "sales.csv", b"date,revenue\n2026-01-02,20\n", session_id=sid)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["session_id"] == sid
    assert second_body["csv_name"] == "sales_2"
    assert second_body["mode"] == "append"
    assert second_body["session_reset"] is False
    assert second_body["datasets"] == ["sales", "sales_2"]
    assert second_body["active_dataset"] == "sales_2"

    listed = await api_client.get(f"/session/{sid}/datasets")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["datasets"] == ["sales", "sales_2"]
    assert listed_body["active_dataset"] == "sales_2"


async def test_upload_replace_session_keeps_only_new_dataset(api_client):
    first = await _upload_csv(api_client, "sales.csv", b"date,revenue\n2026-01-01,10\n")
    sid = first.json()["session_id"]

    second = await _upload_csv(
        api_client,
        "ops.csv",
        b"date,unit,mw\n2026-01-01,U1,70\n",
        session_id=sid,
        mode="replace_session",
    )
    assert second.status_code == 200
    body = second.json()
    assert body["session_id"] == sid
    assert body["mode"] == "replace_session"
    assert body["session_reset"] is True
    assert body["datasets"] == ["ops"]
    assert body["active_dataset"] == "ops"

    listed = await api_client.get(f"/session/{sid}/datasets")
    assert listed.status_code == 200
    assert listed.json()["datasets"] == ["ops"]


async def test_upload_batch_creates_metadata_for_each_file(api_client):
    files = [
        ("files", ("sales.csv", b"date,revenue\n2026-01-01,10\n", "text/csv")),
        ("files", ("ops.csv", b"date,unit,mw\n2026-01-01,U1,70\n", "text/csv")),
    ]
    res = await api_client.post("/upload/batch", data={"mode": "append"}, files=files)
    assert res.status_code == 200
    body = res.json()
    sid = body["session_id"]
    assert body["uploaded_count"] == 2
    assert sorted(body["datasets"]) == ["ops", "sales"]
    assert body["active_dataset"] == "ops"
    assert len(body["uploaded"]) == 2
    assert body["uploaded"][0]["csv_name"] == "sales"
    assert body["uploaded"][1]["csv_name"] == "ops"

    listed = await api_client.get(f"/session/{sid}/datasets")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert sorted(listed_body["datasets"]) == ["ops", "sales"]


async def test_upload_session_limit_exceeded_returns_structured_413(api_client, monkeypatch):
    # Tiny cap to force failure.
    monkeypatch.setattr(main, "SESSION_DATA_BYTES_LIMIT", 64, raising=False)
    huge_csv = b"c1,c2\n" + b"a,b\n" * 200
    res = await _upload_csv(api_client, "big.csv", huge_csv)
    assert res.status_code == 413
    detail = res.json().get("detail", {})
    assert detail.get("code") == "session_limit_exceeded"
    assert "limit_bytes" in detail
    assert "current_bytes" in detail
    assert "session_id" in detail


async def test_list_session_datasets_404_for_unknown_session(api_client):
    missing = await api_client.get("/session/does-not-exist/datasets")
    assert missing.status_code == 404


def test_system_prompt_contains_dataset_inventory_context():
    session = SessionState()
    session.csv_registry = {
        "sales_2024": {"row_count": 100, "columns": ["date", "revenue", "region"]},
        "ops_daily": {"row_count": 50, "columns": ["date", "unit", "mw"]},
    }
    session.dataset_profiles = {
        "sales_2024": {"row_count": 100, "columns": ["date", "revenue", "region"]},
        "ops_daily": {"row_count": 50, "columns": ["date", "unit", "mw"]},
    }
    session.active_dataset = "ops_daily"

    prompt = system_prompt(session)
    assert "DATASET INVENTORY" in prompt
    assert "sales_2024" in prompt
    assert "ops_daily (active)" in prompt
    assert "cols=[date, unit, mw]" in prompt
