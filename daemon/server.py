"""Host daemon: FastAPI + websockets, feeds the dashboard (spec §7 "Host daemon", §8).

Responsibilities (spec §7): device detect, handshake, decrypt to memory, hand the manifest to
the runtime, watch for eject, push state back, wipe (zeroize + visible confirmation).

This process owns the one DaemonState instance and broadcasts every state change to all
connected dashboard websockets. Device polling / real serial detection needs pyserial + an
attached dongle and isn't exercised by tests; the websocket broadcast plumbing and REST surface
are pure-asyncio and are (see test_server.py).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from daemon.crypto import wipe_bytes


@dataclass
class DaemonState:
    connected: bool = False
    device_name: Optional[str] = None
    domain: Optional[str] = None
    profile_id: Optional[str] = None
    footprint_bytes: Optional[int] = None
    ram_free_bytes: Optional[int] = None
    allocation: dict = field(default_factory=dict)
    wiped: bool = True  # starts wiped: nothing decrypted until a dongle connects

    def to_dict(self) -> dict:
        return asdict(self)


class ConnectionManager:
    def __init__(self) -> None:
        self._sockets: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._sockets.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._sockets:
            self._sockets.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self._sockets:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


state = DaemonState()
manager = ConnectionManager()
app = FastAPI(title="tessera-daemon")


@app.get("/state")
async def get_state() -> dict:
    return state.to_dict()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    await ws.send_text(json.dumps(state.to_dict()))
    try:
        while True:
            await ws.receive_text()  # dashboard is read-mostly; ignore inbound for now
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def on_dongle_connect(device_name: str, session_ctx, caps: dict) -> None:
    """Call after run_host_handshake succeeds (usb.py). Updates state + broadcasts, but does
    NOT keep session_ctx.bundle_plaintext around beyond what the runtime needs — the daemon
    process holding it in a local (not global) scope is what makes `wipe` meaningful."""
    manifest = session_ctx.profile_manifest
    state.connected = True
    state.device_name = device_name
    state.domain = manifest.get("domain")
    state.profile_id = manifest.get("profile_id")
    state.footprint_bytes = manifest.get("measured_bytes")
    state.ram_free_bytes = caps.get("ram_free")
    state.allocation = manifest.get("allocation", {})
    state.wiped = False
    await manager.broadcast({"event": "connected", "state": state.to_dict()})


async def on_dongle_eject(bundle_plaintext: bytearray) -> None:
    """Wipe must be visible (spec §7, §12 step 2) — broadcast the zeroize, not just perform it."""
    wipe_bytes(bundle_plaintext)
    state.connected = False
    state.wiped = True
    await manager.broadcast({"event": "wiped", "state": state.to_dict()})


async def on_domain_switch(domain: str, manifest: dict) -> None:
    """Stretch goal 1 (spec §9): automatic domain detection hot-swaps the manifest mid-session."""
    state.domain = domain
    state.profile_id = manifest.get("profile_id")
    state.footprint_bytes = manifest.get("measured_bytes")
    state.allocation = manifest.get("allocation", {})
    await manager.broadcast({"event": "domain_switch", "state": state.to_dict()})


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8420)


if __name__ == "__main__":
    main()
