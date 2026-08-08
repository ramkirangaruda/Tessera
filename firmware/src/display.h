// SSD1306 128x64 OLED (I2C) — live readout, e.g. "CODE · 5-bit · 1.1 GB" (spec §7 BOM).
#pragma once

#include <stdint.h>

namespace tessera {

class Display {
 public:
  bool begin();  // I2C init + SSD1306 reset/config sequence

  // Renders "<DOMAIN> · <bits>-bit · <footprint>" — the header line mirrored on the dashboard
  // (spec §8: "header strip mirroring the OLED").
  void show_profile(const char* domain, int bits, uint32_t footprint_bytes);

  void show_status(const char* text);  // handshake / streaming / error messages
  void clear();
};

}  // namespace tessera
