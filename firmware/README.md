# Tessera dongle firmware

RP2040 (Arduino-Pico core) or ESP32-S3, TinyUSB composite device — see spec §7/§15 open decision
5 for the board choice ("pick by whichever is physically in hand").

## BOM (spec §7)

- MCU: RP2040 (Pico) or ESP32-S3
- ATECC608A secure element (I2C) — key storage, ECDH, signing; the raw private key never leaves
  this chip (spec §6). `daemon/crypto.py`'s `SoftwareSecureElement` is the host-side dev
  stand-in for this while firmware/hardware isn't wired up yet.
- SSD1306 128x64 OLED (I2C) — live readout: `CODE · 5-bit · 1.1 GB`
- WS2812 RGB LED — state: idle / handshake / streaming / active
- Momentary button — cycle Quality / Balanced / Survival
- SPDT slide switch — hardware write-protect (physically blocks writes, not just a firmware flag)
- microSD or onboard flash — bundle storage

## Build

Not yet buildable — no PlatformIO/Arduino-Pico project file checked in (pending board choice,
spec §15). `src/` below is the intended module layout; each file is a stub with the interface
it needs to expose to satisfy the protocol in `daemon/protocol.py` (mirror the frame format and
message types there exactly — the two are tested against each other on the host side via
`daemon/usb.py::DongleSimulator`, and firmware must match that wire format byte-for-byte).

## Milestone (spec §10, M5)

Acceptance: full handshake completes; bundle round-trips encrypted; OLED and LED reflect state;
write-protect switch actually blocks writes. Can develop in parallel against `DongleSimulator`
from day one — does not block on the compiler (M1/M2) or format (M3) work.
