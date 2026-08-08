"""Tests for the handshake crypto + bundle sealing (spec §6, §7 msgs 2-4). Runs entirely against
SoftwareSecureElement — no hardware required."""
from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from daemon.crypto import (
    SoftwareSecureElement,
    derive_session_key,
    new_challenge_nonce,
    open_bundle,
    seal_bundle,
    verify_signature,
    wipe_bytes,
)


def test_ecdh_key_agreement_matches_both_sides():
    host = SoftwareSecureElement()
    dongle = SoftwareSecureElement()

    shared_on_host = host.ecdh(dongle.public_key_bytes())
    shared_on_dongle = dongle.ecdh(host.public_key_bytes())
    assert shared_on_host == shared_on_dongle

    key_host = derive_session_key(shared_on_host)
    key_dongle = derive_session_key(shared_on_dongle)
    assert key_host == key_dongle
    assert len(key_host) == 32


def test_challenge_signature_roundtrip():
    dongle = SoftwareSecureElement()
    nonce = new_challenge_nonce()
    assert len(nonce) == 32

    sig = dongle.sign(nonce)
    assert verify_signature(dongle.public_key_bytes(), nonce, sig)


def test_signature_rejects_wrong_key_or_tampered_nonce():
    dongle = SoftwareSecureElement()
    impostor = SoftwareSecureElement()
    nonce = new_challenge_nonce()
    sig = dongle.sign(nonce)

    assert not verify_signature(impostor.public_key_bytes(), nonce, sig)
    tampered_nonce = bytes([nonce[0] ^ 0xFF]) + nonce[1:]
    assert not verify_signature(dongle.public_key_bytes(), tampered_nonce, sig)


def test_seal_open_bundle_roundtrip():
    host = SoftwareSecureElement()
    dongle = SoftwareSecureElement()
    key = derive_session_key(host.ecdh(dongle.public_key_bytes()))

    plaintext = b'{"session_id": "abc", "transcript": "..."}' * 100
    sealed = seal_bundle(plaintext, key)
    assert sealed.ciphertext != plaintext

    recovered = open_bundle(sealed, key)
    assert recovered == plaintext


def test_open_bundle_rejects_tampered_ciphertext():
    key = derive_session_key(SoftwareSecureElement().ecdh(SoftwareSecureElement().public_key_bytes()))
    sealed = seal_bundle(b"secret session data", key)
    tampered = bytearray(sealed.ciphertext)
    tampered[0] ^= 0xFF
    sealed.ciphertext = bytes(tampered)
    with pytest.raises(InvalidTag):
        open_bundle(sealed, key)


def test_open_bundle_rejects_wrong_key():
    host = SoftwareSecureElement()
    dongle = SoftwareSecureElement()
    key = derive_session_key(host.ecdh(dongle.public_key_bytes()))
    wrong_key = derive_session_key(SoftwareSecureElement().ecdh(SoftwareSecureElement().public_key_bytes()))

    sealed = seal_bundle(b"secret session data", key)
    with pytest.raises(InvalidTag):
        open_bundle(sealed, wrong_key)


def test_wipe_bytes_zeroizes_buffer():
    buf = bytearray(b"very secret transcript data")
    wipe_bytes(buf)
    assert buf == bytearray(len(buf))
