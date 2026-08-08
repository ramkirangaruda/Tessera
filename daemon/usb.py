"""USB CDC transport + host-side handshake orchestration (spec §7).

`Transport` is the abstraction the handshake runs over. `SerialTransport` wraps pyserial for the
real dongle; `LoopbackTransport` is an in-memory pair used by tests and by dev work against a
software-simulated dongle (`DongleSimulator` below) so the daemon doesn't have to block on
firmware (spec §10: "M5/M6 can develop in parallel against a mocked bundle from day one").
"""
from __future__ import annotations

import json
import queue
from dataclasses import dataclass
from typing import Optional, Protocol

from daemon.crypto import (
    SecureElement,
    derive_session_key,
    new_challenge_nonce,
    open_bundle,
    seal_bundle,
    verify_signature,
    SealedBundle,
)
from daemon.protocol import DecodedFrame, FrameStream, MsgType, encode_frame

PROTOCOL_VERSION = 1


class Transport(Protocol):
    def write(self, data: bytes) -> None: ...
    def read(self, timeout_s: float = 1.0) -> bytes: ...


class SerialTransport:
    """Real dongle over USB CDC. Requires pyserial + an attached device; not exercised by the
    test suite (see LoopbackTransport for the same interface under test)."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        import serial  # pyserial

        self._ser = serial.Serial(port, baudrate, timeout=1.0)

    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def read(self, timeout_s: float = 1.0) -> bytes:
        self._ser.timeout = timeout_s
        return self._ser.read(4096)


class LoopbackTransport:
    """One end of an in-memory duplex pipe. `LoopbackTransport.pair()` returns two ends wired
    to each other — stand-in for the physical USB link in tests."""

    def __init__(self, inbox: "queue.Queue[bytes]", outbox: "queue.Queue[bytes]") -> None:
        self._inbox = inbox
        self._outbox = outbox

    @staticmethod
    def pair() -> tuple["LoopbackTransport", "LoopbackTransport"]:
        a_to_b: "queue.Queue[bytes]" = queue.Queue()
        b_to_a: "queue.Queue[bytes]" = queue.Queue()
        return LoopbackTransport(inbox=b_to_a, outbox=a_to_b), LoopbackTransport(inbox=a_to_b, outbox=b_to_a)

    def write(self, data: bytes) -> None:
        self._outbox.put(data)

    def read(self, timeout_s: float = 1.0) -> bytes:
        try:
            return self._inbox.get(timeout=timeout_s)
        except queue.Empty:
            return b""


def _send(transport: Transport, msg_type: MsgType, payload: bytes = b"") -> None:
    transport.write(encode_frame(msg_type, payload))


class HandshakeTimeout(RuntimeError):
    pass


def _recv_one(transport: Transport, stream: FrameStream, timeout_s: float = 2.0, max_wait_s: float = 10.0) -> DecodedFrame:
    waited = 0.0
    frames = stream.feed(transport.read(timeout_s))
    while not frames:
        waited += timeout_s
        if waited >= max_wait_s:
            raise HandshakeTimeout(f"no frame received within {max_wait_s}s")
        frames = stream.feed(transport.read(timeout_s))
    return frames[0]


@dataclass
class SessionContext:
    session_key: bytes
    profile_manifest: dict
    bundle_plaintext: Optional[bytes] = None


def run_host_handshake(
    transport: Transport, host_se: SecureElement, caps: dict
) -> SessionContext:
    """Host side of spec §7 messages 1-7. `caps` is
    {ram_free, has_gpu, backend, mem_bw_mbps, thermal_ok}."""
    stream = FrameStream()

    _send(transport, MsgType.HELLO, f"{PROTOCOL_VERSION}|".encode() + host_se.public_key_bytes())

    challenge = _recv_one(transport, stream)
    assert challenge.msg_type == MsgType.CHALLENGE
    # CHALLENGE payload is the 32-byte nonce followed by the dongle's pubkey (dev/mock path —
    # DongleSimulator rides its pubkey alongside the nonce since there's no out-of-band channel
    # to have learned it from beforehand; a real deployment may provision it differently).
    nonce, dongle_pubkey = challenge.payload[:32], challenge.payload[32:]

    sig = host_se.sign(nonce)
    _send(transport, MsgType.AUTH, sig)

    shared = host_se.ecdh(dongle_pubkey)
    session_key = derive_session_key(shared)

    _send(transport, MsgType.CAPS, json.dumps(caps).encode())

    profile_frame = _recv_one(transport, stream)
    assert profile_frame.msg_type == MsgType.PROFILE
    manifest = json.loads(profile_frame.payload.decode())

    begin = _recv_one(transport, stream)
    assert begin.msg_type == MsgType.SESSION_BEGIN
    chunks = bytearray()
    while True:
        frame = _recv_one(transport, stream)
        if frame.msg_type == MsgType.END:
            break
        assert frame.msg_type == MsgType.CHUNK
        chunks.extend(frame.payload)

    sealed = SealedBundle(nonce=bytes(chunks[:12]), ciphertext=bytes(chunks[12:]))
    plaintext = open_bundle(sealed, session_key)

    return SessionContext(session_key=session_key, profile_manifest=manifest, bundle_plaintext=plaintext)


def push_state_and_wipe(transport: Transport, session: SessionContext, updated_bundle: bytes) -> None:
    """Host side of spec §7 messages 8-9: push updated encrypted bundle on eject, wait for
    WIPE_ACK. Caller is responsible for zeroizing `session.bundle_plaintext` afterwards
    (daemon.crypto.wipe_bytes) — that's the visible-on-dashboard step of the demo (spec §12)."""
    sealed = seal_bundle(updated_bundle, session.session_key)
    _send(transport, MsgType.STATE_PUSH, sealed.nonce + sealed.ciphertext)

    stream = FrameStream()
    ack = _recv_one(transport, stream)
    if ack.msg_type != MsgType.WIPE_ACK:
        raise RuntimeError(f"expected WIPE_ACK, got {ack.msg_type}")


class DongleSimulator:
    """Software stand-in for the dongle side of the handshake — real dongle logic lives in
    firmware/ (C++/TinyUSB). This lets the host daemon's handshake be developed and tested
    without hardware (spec §10). Runs the dongle side of §7 messages 1-9 over a Transport."""

    def __init__(self, transport: Transport, dongle_se: SecureElement, manifest: dict, bundle_plaintext: bytes) -> None:
        self._t = transport
        self._se = dongle_se
        self._manifest = manifest
        self._bundle_plaintext = bundle_plaintext
        self._session_key: Optional[bytes] = None

    def serve_one_session(self) -> None:
        stream = FrameStream()
        hello = _recv_one(self._t, stream)
        assert hello.msg_type == MsgType.HELLO
        host_pubkey = hello.payload.split(b"|", 1)[1]

        nonce = new_challenge_nonce()
        _send(self._t, MsgType.CHALLENGE, nonce + self._se.public_key_bytes())

        auth = _recv_one(self._t, stream)
        assert auth.msg_type == MsgType.AUTH
        if not verify_signature(host_pubkey, nonce, auth.payload):
            raise PermissionError("host signature verification failed")

        shared = self._se.ecdh(host_pubkey)
        self._session_key = derive_session_key(shared)

        caps_frame = _recv_one(self._t, stream)
        assert caps_frame.msg_type == MsgType.CAPS
        # real dongle picks profile by matching caps["ram_free"] against manifest grid; the
        # simulator is handed a fixed manifest by the caller for determinism in tests.
        _send(self._t, MsgType.PROFILE, json.dumps(self._manifest).encode())

        sealed = seal_bundle(self._bundle_plaintext, self._session_key)
        _send(self._t, MsgType.SESSION_BEGIN)
        _send(self._t, MsgType.CHUNK, sealed.nonce + sealed.ciphertext)
        _send(self._t, MsgType.END)

    def serve_state_push_and_wipe(self) -> bytes:
        stream = FrameStream()
        push = _recv_one(self._t, stream)
        assert push.msg_type == MsgType.STATE_PUSH
        sealed = SealedBundle(nonce=push.payload[:12], ciphertext=push.payload[12:])
        updated = open_bundle(sealed, self._session_key)
        _send(self._t, MsgType.WIPE_ACK)
        return updated
