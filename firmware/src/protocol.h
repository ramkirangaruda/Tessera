// Wire protocol — MUST mirror daemon/protocol.py exactly (frame format, message type values).
// Frame: [u16 len][u8 type][payload][u32 crc], little-endian, CRC32 over header+payload.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace tessera {

enum class MsgType : uint8_t {
  HELLO = 1,
  CHALLENGE = 2,
  AUTH = 3,
  CAPS = 5,
  PROFILE = 6,
  SESSION_BEGIN = 7,
  CHUNK = 8,
  END = 9,
  STATE_PUSH = 10,
  WIPE_ACK = 11,
};

constexpr size_t HEADER_SIZE = 3;  // u16 len + u8 type
constexpr size_t CRC_SIZE = 4;
constexpr size_t MAX_PAYLOAD = 0xFFFF;

// Encodes a frame into `out` (caller-provided buffer, must be >= payload_len + HEADER_SIZE +
// CRC_SIZE). Returns the total frame length, or 0 if payload_len > MAX_PAYLOAD.
size_t encode_frame(MsgType type, const uint8_t* payload, size_t payload_len, uint8_t* out, size_t out_cap);

// Streaming decoder — feed() accumulates bytes and calls `on_frame` for each complete,
// CRC-verified frame. Mirrors daemon/protocol.py's FrameStream. Frames with a bad CRC are
// dropped silently (logged); a real link (USB CDC) doesn't corrupt bytes in transit, but this
// guards against a torn read across handshake retries.
class FrameStream {
 public:
  using FrameCallback = void (*)(MsgType type, const uint8_t* payload, size_t len, void* ctx);

  void feed(const uint8_t* data, size_t len, FrameCallback cb, void* ctx);

 private:
  static constexpr size_t BUF_CAP = 4096;
  uint8_t buf_[BUF_CAP];
  size_t buf_len_ = 0;
};

uint32_t crc32(const uint8_t* data, size_t len);

}  // namespace tessera
