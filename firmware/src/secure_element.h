// ATECC608A driver interface (I2C secure element — spec §6/§7). Key storage, ECDH, ECDSA
// signing; the raw private key never leaves the chip. Backed by Microchip's cryptoauthlib
// (https://github.com/MicrochipTech/cryptoauthlib) — this header is the thin API surface
// firmware code above it (main.cpp) actually calls, so cryptoauthlib specifics stay contained.
//
// Mirrors the SecureElement Protocol in daemon/crypto.py so the same handshake logic
// (spec §7 messages 2-4) is implemented once, conceptually, on both ends.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace tessera {

constexpr size_t PUBKEY_SIZE = 65;   // uncompressed P-256 point (0x04 || X || Y)
constexpr size_t SIGNATURE_SIZE = 64;  // raw (r, s), 32 bytes each
constexpr size_t SHARED_SECRET_SIZE = 32;

class SecureElement {
 public:
  // Initializes I2C + verifies the ATECC608A is present and provisioned with a key in the
  // configured slot. Returns false on failure (missing/misconfigured chip) — callers should
  // treat this as fatal at boot, since there is no software fallback for key custody (spec §6:
  // "the raw private key never leaves the chip").
  bool begin();

  void public_key(uint8_t out[PUBKEY_SIZE]);
  void sign(const uint8_t* data, size_t len, uint8_t out_sig[SIGNATURE_SIZE]);
  void ecdh(const uint8_t peer_pubkey[PUBKEY_SIZE], uint8_t out_shared[SHARED_SECRET_SIZE]);
};

}  // namespace tessera
