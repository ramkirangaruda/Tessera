# Demo script

Choreography is fixed in [`spec.md` §12](spec.md#12-demo-choreography) — this file is the
rehearsal checklist and line-by-line narration underneath it. Update as the rig stabilizes;
`spec.md` §12 itself should not need to change.

## Setup (before doors open)

- [ ] Three devices on the table: laptop, Raspberry Pi 5, Raspberry Pi Zero 2W — all powered,
      all with `daemon/server.py` (or the Pi-side equivalent) running and reachable
- [ ] Dongle provisioned with the manifest grid for both Qwen2.5-1.5B and -0.5B, ATECC608A keyed
- [ ] Dashboard (`dashboard/`) projected, connected to the laptop's daemon over websocket
- [ ] Write-protect switch confirmed in the *unlocked* position (needed for step 2's write)
- [ ] A real bug pulled from an actual project, ready to paste for step 1 — not a toy snippet
- [ ] Pareto figure (`eval/figures.py::plot_pareto`) pre-generated and on standby in case the
      live dashboard panel misbehaves

## Walkthrough

1. **Laptop, dongle in.** Plug in. OLED reads `CODE · 6-bit · 2.1 GB`. Paste the real bug, ask
   Tessera to debug it. Narrate the heatmap: MLP tensors (mlp_gate/up/down) run richer than
   attention — "that's not a hand-tuned recipe, the allocator found that MLP is ~88% of the
   weight and spent its budget there."
2. **Unplug.** Point at the dashboard: session visibly wipes (`wiped: true`, the zeroize
   confirmation). Say it out loud: "the laptop cannot answer a follow-up right now — nothing of
   the conversation is left on it."
3. **Into the Pi 5.** OLED reads `CODE · 4-bit · 1.1 GB`. Ask a follow-up that only makes sense
   with the earlier context (reference something specific from step 1's bug). It answers
   correctly. Heatmap redraws leaner — same shape, lower bit-widths throughout.
4. **Into the Pi Zero 2W.** OLED reads `CHAT · 3-bit · 380 MB`. This is a *different model*
   (0.5B, not 1.5B) — say so explicitly. Slower, simpler, still coherent, still remembers the
   session. This is the strongest defense of the transport design (spec §3): the session is a
   transcript, not a KV cache, so it survives a model swap, not just a precision swap.
5. **Ask a math question.** Domain shift detected (or button-toggled, per stretch goal 1 status
   at the time), manifest hot-swaps to `math`. Same `.tsra` file on disk — narrate that: "no
   re-download, no re-quantization, we're just reading more planes off the tensors that matter
   for arithmetic." Heatmap visibly redistributes.

Close on the Pareto plot: every Tessera profile marked above the uniform baseline curve.

## Anticipated questions (see spec.md §14 for the full prior-art table)

- "Isn't this just AWQ/GPTQ?" → one-breath answer in spec.md §14.
- "What if the cross-domain correlation turns out high?" → spec.md §4.3: pivot the pitch onto
  the nested format + device-conditioning, say so honestly.
- "Why not transport the KV cache?" → spec.md §6: size, numerical validity, model portability.

## Numbers to have on a slide, not improvised (spec §13)

Pareto curve, measured RSS per device/profile, tokens/sec per device/profile, resume latency,
cross-domain Spearman matrix, nesting tax, bundle size distribution.
