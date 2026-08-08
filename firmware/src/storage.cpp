#include "storage.h"

// TODO(hardware): wire to the actual SPI/SDIO SD card driver or onboard flash filesystem
// (LittleFS on RP2040 flash, or SdFat over SPI for a microSD breakout — pick with the board,
// spec §15 open decision 5). GPIO pin for the write-protect switch is board-specific; define
// WRITE_PROTECT_PIN in board config once hardware is in hand.

namespace tessera {

bool Storage::begin() {
  // TODO: init SD/flash, mount filesystem, configure WRITE_PROTECT_PIN as INPUT_PULLUP
  return true;
}

bool Storage::write_protected() const {
  // TODO: return digitalRead(WRITE_PROTECT_PIN) == LOW (switch position -> read-only)
  return false;
}

bool Storage::write_bundle(const uint8_t* encrypted_bundle, size_t len) {
  if (write_protected()) {
    return false;  // physical guard — no override path, see storage.h
  }
  // TODO: write encrypted_bundle to bundle.tsra-session on the storage medium
  (void)encrypted_bundle;
  (void)len;
  return true;
}

size_t Storage::read_bundle(uint8_t* out, size_t out_cap) {
  // TODO: read the stored encrypted bundle into `out`, return bytes read (0 if none present)
  (void)out;
  (void)out_cap;
  return 0;
}

}  // namespace tessera
