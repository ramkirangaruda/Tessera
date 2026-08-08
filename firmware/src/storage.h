// Bundle storage (microSD or onboard flash) + the physical write-protect switch (spec §7 BOM).
//
// The SPDT slide switch is read directly as a GPIO level, not debounced into a "mode" the rest
// of firmware can override — `write_protected()` must reflect the physical switch position on
// every call so a write attempted while the switch is in the read-only position always fails,
// with no code path (bug or otherwise) able to bypass it. That physical guarantee is the whole
// point (spec §7: "it *physically cannot* write to your context").
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace tessera {

class Storage {
 public:
  bool begin();

  bool write_protected() const;  // reads the slide switch GPIO directly, every call

  // Returns false without writing anything if write_protected() is true.
  bool write_bundle(const uint8_t* encrypted_bundle, size_t len);

  size_t read_bundle(uint8_t* out, size_t out_cap);
};

}  // namespace tessera
