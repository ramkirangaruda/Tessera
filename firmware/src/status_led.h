// WS2812 RGB LED — state indicator (spec §7 BOM): idle / handshake / streaming / active.
#pragma once

namespace tessera {

enum class LedState {
  IDLE,       // dim white breathing
  HANDSHAKE,  // amber pulse
  STREAMING,  // blue chase (bundle transfer in flight)
  ACTIVE,     // solid green (session live on host)
  ERROR,      // solid red
};

class StatusLed {
 public:
  bool begin();
  void set_state(LedState state);
  void tick();  // call from the main loop to advance animations (pulse/chase/breathe)
};

}  // namespace tessera
