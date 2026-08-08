// Tessera dongle firmware entrypoint. Arduino-Pico core + TinyUSB composite (MSC + CDC),
// spec §7. Drives the same handshake sequence daemon/usb.py::DongleSimulator implements in
// Python for host-side testing — keep the two in lockstep when either changes.
//
// TODO(hardware): this file is the module wiring/skeleton; the USB composite descriptor setup
// (TinyUSB CDC+MSC) is board/core-specific boilerplate not written here pending the RP2040 vs
// ESP32-S3 decision (spec §15 open decision 5).

#include <string.h>

#include "protocol.h"
#include "secure_element.h"
#include "display.h"
#include "status_led.h"
#include "storage.h"

using namespace tessera;

namespace {

SecureElement g_se;
Display g_display;
StatusLed g_led;
Storage g_storage;
FrameStream g_frame_stream;

enum class HandshakeState {
  IDLE,
  AWAIT_HELLO,
  SENT_CHALLENGE,
  AUTHENTICATED,
  SESSION_ACTIVE,
};

HandshakeState g_state = HandshakeState::IDLE;
uint8_t g_challenge_nonce[32];
uint8_t g_session_key[32];

void send_frame(MsgType type, const uint8_t* payload, size_t len) {
  static uint8_t out_buf[4096];
  size_t n = encode_frame(type, payload, len, out_buf, sizeof(out_buf));
  if (n == 0) return;  // TODO(hardware): log oversized/failed frame
  // TODO(hardware): write out_buf[0:n] to the CDC endpoint
}

void on_frame(MsgType type, const uint8_t* payload, size_t len, void* /*ctx*/) {
  switch (type) {
    case MsgType::HELLO: {
      if (g_state != HandshakeState::IDLE) return;
      g_led.set_state(LedState::HANDSHAKE);
      // TODO(hardware): fill g_challenge_nonce from a real RNG (RP2040 hardware RNG / ESP32
      // TRNG), not a placeholder.
      uint8_t challenge_payload[32 + PUBKEY_SIZE];
      memcpy(challenge_payload, g_challenge_nonce, 32);
      g_se.public_key(challenge_payload + 32);
      send_frame(MsgType::CHALLENGE, challenge_payload, sizeof(challenge_payload));
      g_state = HandshakeState::SENT_CHALLENGE;
      break;
    }
    case MsgType::AUTH: {
      if (g_state != HandshakeState::SENT_CHALLENGE) return;
      // TODO(hardware): verify `payload` (host signature over g_challenge_nonce) against the
      // host pubkey learned from HELLO; on failure, set LedState::ERROR and reset to IDLE.
      g_state = HandshakeState::AUTHENTICATED;
      break;
    }
    case MsgType::CAPS: {
      if (g_state != HandshakeState::AUTHENTICATED) return;
      // TODO: parse {ram_free, has_gpu, backend, mem_bw_mbps, thermal_ok} JSON payload, match
      // against the manifest grid stored alongside the bundle, filtered by the domain selected
      // via the button (or auto-detected by the host, spec §9 stretch 1).
      g_led.set_state(LedState::STREAMING);
      // TODO: send_frame(MsgType::PROFILE, manifest_json, manifest_len);
      // TODO: send_frame(MsgType::SESSION_BEGIN, ...); chunk the encrypted bundle; END
      g_state = HandshakeState::SESSION_ACTIVE;
      g_led.set_state(LedState::ACTIVE);
      break;
    }
    case MsgType::STATE_PUSH: {
      if (g_state != HandshakeState::SESSION_ACTIVE) return;
      g_storage.write_bundle(payload, len);  // no-ops if write-protect switch is engaged
      send_frame(MsgType::WIPE_ACK, nullptr, 0);
      break;
    }
    default:
      break;  // unexpected message for current state — ignore
  }
}

}  // namespace

void setup() {
  g_se.begin();
  g_display.begin();
  g_led.begin();
  g_storage.begin();
  g_display.show_status("idle");
  g_led.set_state(LedState::IDLE);
}

void loop() {
  // TODO(hardware): read available bytes from the TinyUSB CDC endpoint into `buf`
  uint8_t buf[256];
  size_t n = 0;
  if (n > 0) {
    g_frame_stream.feed(buf, n, on_frame, nullptr);
  }
  g_led.tick();
}
