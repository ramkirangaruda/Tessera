"""Tests for the daemon's FastAPI/websocket surface — pure asyncio via TestClient, no real
dongle or serial device required."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from daemon.server import DaemonState, app, on_domain_switch, on_dongle_connect, on_dongle_eject, state


@dataclass
class FakeSessionContext:
    profile_manifest: dict


def setup_function():
    # reset module-level singleton between tests
    fresh = DaemonState()
    for k, v in fresh.to_dict().items():
        setattr(state, k, v)


def test_initial_state_is_wiped():
    client = TestClient(app)
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["wiped"] is True
    assert body["connected"] is False


def test_websocket_receives_initial_state_snapshot():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["wiped"] is True


def test_dongle_connect_broadcasts_and_updates_state():
    client = TestClient(app)
    manifest = {
        "domain": "code",
        "profile_id": "code@1100MB",
        "measured_bytes": 1149203968,
        "allocation": {"token_embd": 6},
    }
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial snapshot
        import asyncio

        asyncio.run(on_dongle_connect("laptop", FakeSessionContext(manifest), {"ram_free": 8_000_000_000}))
        msg = ws.receive_json()
        assert msg["event"] == "connected"
        assert msg["state"]["domain"] == "code"
        assert msg["state"]["profile_id"] == "code@1100MB"
        assert msg["state"]["wiped"] is False

    resp = client.get("/state")
    assert resp.json()["connected"] is True


def test_dongle_eject_wipes_and_broadcasts():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        import asyncio

        bundle = bytearray(b"secret transcript data")
        asyncio.run(on_dongle_eject(bundle))
        msg = ws.receive_json()
        assert msg["event"] == "wiped"
        assert msg["state"]["wiped"] is True
        assert bundle == bytearray(len(bundle))  # zeroized in place


def test_domain_switch_broadcasts_new_manifest():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        import asyncio

        asyncio.run(on_domain_switch("math", {"profile_id": "math@1100MB", "measured_bytes": 900, "allocation": {}}))
        msg = ws.receive_json()
        assert msg["event"] == "domain_switch"
        assert msg["state"]["domain"] == "math"
