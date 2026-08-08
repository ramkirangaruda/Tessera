"""Wire protocol for the dongle <-> host handshake (spec §7).

Frame: [u16 len][u8 type][payload][u32 crc]  — length-prefixed, CRC32-checked, over the USB CDC
endpoint. `len` covers `payload` only. Pure stdlib (struct + zlib), no hardware or network
dependency, so this is fully unit-testable without a dongle attached.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum


class MsgType(IntEnum):
    HELLO = 1          # host -> dongle: protocol version, host pubkey
    CHALLENGE = 2       # dongle -> host: 32-byte nonce
    AUTH = 3             # host -> dongle: signature over nonce
    CAPS = 5              # host -> dongle: {ram_free, has_gpu, backend, mem_bw_mbps, thermal_ok}
    PROFILE = 6            # dongle -> host: chosen profile_id + full manifest
    SESSION_BEGIN = 7        # dongle -> host: encrypted bundle transfer starts
    CHUNK = 8                 # dongle -> host: encrypted bundle chunk
    END = 9                    # dongle -> host: encrypted bundle transfer ends
    STATE_PUSH = 10              # host -> dongle: updated encrypted bundle, on eject
    WIPE_ACK = 11                 # dongle -> host: wipe confirmed


HEADER_FMT = "<HB"   # u16 len, u8 type
CRC_FMT = "<I"        # u32 crc
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRC_SIZE = struct.calcsize(CRC_FMT)


class FrameError(ValueError):
    pass


def encode_frame(msg_type: "MsgType | int", payload: bytes = b"") -> bytes:
    if len(payload) > 0xFFFF:
        raise FrameError(f"payload too large: {len(payload)} bytes")
    header = struct.pack(HEADER_FMT, len(payload), int(msg_type))
    crc = zlib.crc32(header + payload)
    return header + payload + struct.pack(CRC_FMT, crc)


@dataclass
class DecodedFrame:
    msg_type: int
    payload: bytes


def decode_frame(buf: bytes) -> tuple[DecodedFrame, bytes]:
    """Decode exactly one frame off the front of `buf`. Returns (frame, remaining_bytes).
    Raises FrameError on a bad CRC. Raises NeedMoreData (a FrameError subclass) if `buf`
    doesn't yet contain a full frame — caller should buffer more bytes and retry."""
    if len(buf) < HEADER_SIZE:
        raise NeedMoreData()
    payload_len, msg_type = struct.unpack_from(HEADER_FMT, buf, 0)
    total = HEADER_SIZE + payload_len + CRC_SIZE
    if len(buf) < total:
        raise NeedMoreData()

    payload = buf[HEADER_SIZE:HEADER_SIZE + payload_len]
    (crc_received,) = struct.unpack_from(CRC_FMT, buf, HEADER_SIZE + payload_len)
    crc_expected = zlib.crc32(buf[:HEADER_SIZE + payload_len])
    if crc_received != crc_expected:
        raise FrameError(f"CRC mismatch: got {crc_received:#x}, expected {crc_expected:#x}")

    return DecodedFrame(msg_type=msg_type, payload=payload), buf[total:]


class NeedMoreData(FrameError):
    """Raised when `buf` doesn't yet contain a complete frame."""


class FrameStream:
    """Accumulates bytes from a streaming transport (USB CDC has no message boundaries) and
    yields complete, CRC-verified frames as they become available."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[DecodedFrame]:
        self._buf.extend(data)
        frames = []
        while True:
            try:
                frame, rest = decode_frame(bytes(self._buf))
            except NeedMoreData:
                break
            frames.append(frame)
            self._buf = bytearray(rest)
        return frames
