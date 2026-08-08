"""End-to-end handshake test: host daemon logic vs. DongleSimulator (software stand-in for
firmware/) over an in-memory LoopbackTransport. No hardware required — this is exactly the mode
spec §10 calls out for parallel M5/M6 development ("develop against a mocked bundle")."""
from __future__ import annotations

import json
import threading

from daemon.crypto import SoftwareSecureElement, wipe_bytes
from daemon.usb import DongleSimulator, LoopbackTransport, push_state_and_wipe, run_host_handshake

MANIFEST = {
    "profile_id": "code@1100MB",
    "schema_version": 1,
    "model": "qwen2.5-1.5b-instruct",
    "domain": "code",
    "budget_bytes": 1153433600,
    "measured_bytes": 1149203968,
    "allocation": {"token_embd": 6, "blk.0.attn_q": 4},
    "eval": {"wikitext_ppl": 11.24},
}

BUNDLE = json.dumps({
    "meta": {"session_id": "abc123", "turn_count": 4},
    "transcript": [{"role": "user", "content": "hello"}],
}).encode()

CAPS = {"ram_free": 4_000_000_000, "has_gpu": False, "backend": "cpu", "mem_bw_mbps": 8000, "thermal_ok": True}


def test_full_handshake_and_bundle_transfer():
    host_transport, dongle_transport = LoopbackTransport.pair()
    host_se = SoftwareSecureElement()
    dongle_se = SoftwareSecureElement()
    dongle = DongleSimulator(dongle_transport, dongle_se, MANIFEST, BUNDLE)

    results = {}

    def run_dongle():
        dongle.serve_one_session()

    def run_host():
        results["session"] = run_host_handshake(host_transport, host_se, CAPS)

    t_dongle = threading.Thread(target=run_dongle, daemon=True)
    t_host = threading.Thread(target=run_host, daemon=True)
    t_dongle.start()
    t_host.start()
    t_dongle.join(timeout=5)
    t_host.join(timeout=5)

    session = results["session"]
    assert session.profile_manifest == MANIFEST
    assert session.bundle_plaintext == BUNDLE
    assert len(session.session_key) == 32


def test_state_push_and_wipe_ack():
    host_transport, dongle_transport = LoopbackTransport.pair()
    host_se = SoftwareSecureElement()
    dongle_se = SoftwareSecureElement()
    dongle = DongleSimulator(dongle_transport, dongle_se, MANIFEST, BUNDLE)

    results = {}
    t_dongle = threading.Thread(target=lambda: dongle.serve_one_session(), daemon=True)
    t_host = threading.Thread(target=lambda: results.__setitem__("session", run_host_handshake(host_transport, host_se, CAPS)), daemon=True)
    t_dongle.start(); t_host.start()
    t_dongle.join(timeout=5); t_host.join(timeout=5)
    session = results["session"]

    updated_bundle = json.dumps({"meta": {"session_id": "abc123", "turn_count": 5}}).encode()
    dongle_results = {}
    t_dongle2 = threading.Thread(target=lambda: dongle_results.__setitem__("updated", dongle.serve_state_push_and_wipe()), daemon=True)
    t_host2 = threading.Thread(target=lambda: push_state_and_wipe(host_transport, session, updated_bundle), daemon=True)
    t_dongle2.start(); t_host2.start()
    t_dongle2.join(timeout=5); t_host2.join(timeout=5)

    assert dongle_results["updated"] == updated_bundle

    # host-side wipe: zeroize its copy of the plaintext bundle after pushing state
    buf = bytearray(session.bundle_plaintext)
    wipe_bytes(buf)
    assert buf == bytearray(len(buf))
