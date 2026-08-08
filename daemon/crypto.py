"""Bundle sealing + the host side of the ATECC608A handshake (spec §6 encryption, §7 messages
2-4: CHALLENGE / AUTH / ECDH -> session key).

Real crypto (P-256 ECDH + HKDF + AES-256-GCM) via the `cryptography` package. The dongle side of
the secure element is real ATECC608A hardware (spec §7 BOM) that firmware/ talks to over I2C; on
the host/dev side there's nothing to talk to yet, so `SoftwareSecureElement` is a same-API
stand-in so the daemon's handshake logic can be built and tested against a mocked dongle before
firmware exists (spec §10: "M5/M6 can develop in parallel against a mocked bundle from day one").
Swap it for a real ATECC608A driver client without touching server.py/usb.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

NONCE_SIZE = 12  # 96-bit, standard for AES-GCM
KEY_SIZE = 32  # AES-256


class SecureElement(Protocol):
    """API a dongle-side key custodian must expose. Implemented today by
    SoftwareSecureElement (dev/testing); the real dongle firmware talks to an ATECC608A over I2C
    instead — the private key never leaves that chip (spec §6)."""

    def public_key_bytes(self) -> bytes: ...
    def sign(self, data: bytes) -> bytes: ...
    def ecdh(self, peer_public_key_bytes: bytes) -> bytes: ...


class SoftwareSecureElement:
    """Dev/testing stand-in for the ATECC608A. NOT what ships on the dongle — real firmware
    keeps the private key in hardware and never exposes it, whereas this class holds it in
    process memory for the sake of exercising the daemon's protocol logic without hardware."""

    def __init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())

    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )

    def sign(self, data: bytes) -> bytes:
        return self._private_key.sign(data, ec.ECDSA(hashes.SHA256()))

    def ecdh(self, peer_public_key_bytes: bytes) -> bytes:
        peer_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), peer_public_key_bytes)
        return self._private_key.exchange(ec.ECDH(), peer_key)


def verify_signature(public_key_bytes: bytes, data: bytes, signature: bytes) -> bool:
    public_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public_key_bytes)
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def derive_session_key(shared_secret: bytes, info: bytes = b"tessera-session-key") -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=KEY_SIZE, salt=None, info=info).derive(shared_secret)


def new_challenge_nonce() -> bytes:
    return os.urandom(32)  # spec §7 msg 2: 32-byte nonce


@dataclass
class SealedBundle:
    nonce: bytes
    ciphertext: bytes  # AESGCM ciphertext includes the 16-byte auth tag


def seal_bundle(plaintext: bytes, session_key: bytes) -> SealedBundle:
    nonce = os.urandom(NONCE_SIZE)
    ct = AESGCM(session_key).encrypt(nonce, plaintext, associated_data=None)
    return SealedBundle(nonce=nonce, ciphertext=ct)


def open_bundle(sealed: SealedBundle, session_key: bytes) -> bytes:
    """Raises cryptography.exceptions.InvalidTag if the bundle was tampered with or the key is
    wrong — never partially decrypts."""
    return AESGCM(session_key).decrypt(sealed.nonce, sealed.ciphertext, associated_data=None)


def wipe_bytes(buf: bytearray) -> None:
    """Zeroize an in-memory bundle (spec §7 daemon responsibilities: 'wipe'). Must be called on
    every mutable buffer holding decrypted session data before it's dropped, and the daemon must
    surface this visibly to the dashboard — the wipe is step 2 of the demo (spec §12)."""
    for i in range(len(buf)):
        buf[i] = 0
