#include "protocol.h"

#include <string.h>

namespace tessera {

// Standard CRC-32 (polynomial 0xEDB88320), matches Python's zlib.crc32 used in
// daemon/protocol.py — required for the two implementations to interoperate.
uint32_t crc32(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; bit++) {
      uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

size_t encode_frame(MsgType type, const uint8_t* payload, size_t payload_len, uint8_t* out, size_t out_cap) {
  if (payload_len > MAX_PAYLOAD) return 0;
  size_t total = HEADER_SIZE + payload_len + CRC_SIZE;
  if (out_cap < total) return 0;

  out[0] = static_cast<uint8_t>(payload_len & 0xFF);
  out[1] = static_cast<uint8_t>((payload_len >> 8) & 0xFF);
  out[2] = static_cast<uint8_t>(type);
  if (payload_len > 0) {
    memcpy(out + HEADER_SIZE, payload, payload_len);
  }

  uint32_t crc = crc32(out, HEADER_SIZE + payload_len);
  size_t crc_off = HEADER_SIZE + payload_len;
  out[crc_off + 0] = static_cast<uint8_t>(crc & 0xFF);
  out[crc_off + 1] = static_cast<uint8_t>((crc >> 8) & 0xFF);
  out[crc_off + 2] = static_cast<uint8_t>((crc >> 16) & 0xFF);
  out[crc_off + 3] = static_cast<uint8_t>((crc >> 24) & 0xFF);

  return total;
}

void FrameStream::feed(const uint8_t* data, size_t len, FrameCallback cb, void* ctx) {
  // Append incoming bytes, draining complete frames off the front. BUF_CAP bounds worst-case
  // memory; a manifest/profile frame is a few KB (spec §4.2), well under it.
  size_t copy_len = len;
  if (buf_len_ + copy_len > BUF_CAP) {
    copy_len = BUF_CAP - buf_len_;  // drop overflow rather than corrupt the buffer
  }
  memcpy(buf_ + buf_len_, data, copy_len);
  buf_len_ += copy_len;

  size_t pos = 0;
  while (buf_len_ - pos >= HEADER_SIZE) {
    uint16_t payload_len = buf_[pos] | (static_cast<uint16_t>(buf_[pos + 1]) << 8);
    MsgType type = static_cast<MsgType>(buf_[pos + 2]);
    size_t total = HEADER_SIZE + payload_len + CRC_SIZE;
    if (buf_len_ - pos < total) break;  // incomplete frame, wait for more data

    uint32_t crc_received = buf_[pos + HEADER_SIZE + payload_len] |
                             (static_cast<uint32_t>(buf_[pos + HEADER_SIZE + payload_len + 1]) << 8) |
                             (static_cast<uint32_t>(buf_[pos + HEADER_SIZE + payload_len + 2]) << 16) |
                             (static_cast<uint32_t>(buf_[pos + HEADER_SIZE + payload_len + 3]) << 24);
    uint32_t crc_expected = crc32(buf_ + pos, HEADER_SIZE + payload_len);

    if (crc_received == crc_expected) {
      cb(type, buf_ + pos + HEADER_SIZE, payload_len, ctx);
    }
    pos += total;
  }

  // shift remaining unparsed bytes to the front
  size_t remaining = buf_len_ - pos;
  memmove(buf_, buf_ + pos, remaining);
  buf_len_ = remaining;
}

}  // namespace tessera
