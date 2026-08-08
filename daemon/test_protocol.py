"""Unit tests for the frame protocol (spec §7) — no hardware required."""
from __future__ import annotations

import pytest

from daemon.protocol import FrameError, FrameStream, MsgType, decode_frame, encode_frame


def test_encode_decode_roundtrip():
    frame_bytes = encode_frame(MsgType.HELLO, b"hello payload")
    frame, rest = decode_frame(frame_bytes)
    assert frame.msg_type == MsgType.HELLO
    assert frame.payload == b"hello payload"
    assert rest == b""


def test_empty_payload():
    frame_bytes = encode_frame(MsgType.WIPE_ACK, b"")
    frame, rest = decode_frame(frame_bytes)
    assert frame.msg_type == MsgType.WIPE_ACK
    assert frame.payload == b""


def test_corrupted_crc_raises():
    frame_bytes = bytearray(encode_frame(MsgType.CAPS, b"{}"))
    frame_bytes[-1] ^= 0xFF  # flip a bit in the CRC
    with pytest.raises(FrameError):
        decode_frame(bytes(frame_bytes))


def test_corrupted_payload_raises():
    frame_bytes = bytearray(encode_frame(MsgType.CAPS, b"{\"ram_free\": 1}"))
    frame_bytes[5] ^= 0xFF  # flip a bit inside the payload
    with pytest.raises(FrameError):
        decode_frame(bytes(frame_bytes))


def test_multiple_frames_back_to_back():
    stream = FrameStream()
    blob = encode_frame(MsgType.HELLO, b"a") + encode_frame(MsgType.CHALLENGE, b"bb") + encode_frame(MsgType.AUTH, b"ccc")
    frames = stream.feed(blob)
    assert [f.msg_type for f in frames] == [MsgType.HELLO, MsgType.CHALLENGE, MsgType.AUTH]
    assert [f.payload for f in frames] == [b"a", b"bb", b"ccc"]


def test_frames_split_across_feeds():
    """Simulates a USB CDC stream delivering bytes in arbitrary chunks."""
    stream = FrameStream()
    blob = encode_frame(MsgType.PROFILE, b"profile-json-blob-of-some-length")
    assert stream.feed(blob[:3]) == []
    assert stream.feed(blob[3:10]) == []
    frames = stream.feed(blob[10:])
    assert len(frames) == 1
    assert frames[0].msg_type == MsgType.PROFILE
    assert frames[0].payload == b"profile-json-blob-of-some-length"


def test_full_handshake_message_sequence():
    """Encodes/decodes the full §7 handshake sequence end to end through one FrameStream."""
    stream = FrameStream()
    sequence = [
        (MsgType.HELLO, b"v1|hostpubkey"),
        (MsgType.CHALLENGE, b"\x00" * 32),
        (MsgType.AUTH, b"signature-bytes"),
        (MsgType.CAPS, b'{"ram_free": 4000000000, "has_gpu": false}'),
        (MsgType.PROFILE, b'{"profile_id": "code@1100MB"}'),
        (MsgType.SESSION_BEGIN, b""),
        (MsgType.CHUNK, b"encrypted-bundle-chunk-1"),
        (MsgType.END, b""),
        (MsgType.WIPE_ACK, b""),
    ]
    blob = b"".join(encode_frame(t, p) for t, p in sequence)
    frames = stream.feed(blob)
    assert [f.msg_type for f in frames] == [t for t, _ in sequence]
    assert [f.payload for f in frames] == [p for _, p in sequence]


def test_payload_too_large_rejected():
    with pytest.raises(FrameError):
        encode_frame(MsgType.CHUNK, b"x" * 70000)
