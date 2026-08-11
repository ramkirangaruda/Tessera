# Demo script

Choreography is fixed in [`spec.md` §12](spec.md#12-demo-choreography) — this file is the
rehearsal checklist and line-by-line narration underneath it. Update as the rig stabilizes;
`spec.md` §12 itself should not need to change.

## Setup (before doors open)

- [ ] Three devices on the table: laptop, Raspberry Pi 5, Raspberry Pi Zero 2W — all powered,
      all with `daemon/server.py` (or the Pi-side equivalent) running and reachable
- [ ] Pi 5: NVMe HAT attached and booted from it (required for the M3 load-time budget — spec
      §3), active cooler running, Raspberry Pi OS Lite 64-bit headless with CMA minimized
- [ ] Dongle provisioned with the manifest grid for both Qwen3-1.7B and -0.6B, ATECC608A keyed
- [ ] Dashboard (`dashboard/`) projected, connected to the laptop's daemon over websocket
- [ ] Write-protect switch confirmed in the *unlocked* position (needed for step 2's write)
- [ ] A real bug pulled from an actual project, ready to paste for step 1 — not a toy snippet
- [ ] For step 3: a real long document ready to paste on the Pi 5 (a stack trace + a few related
      files, or similar) — enough material to land the session around 32k tokens. Rehearsed, not
      improvised; the crossover only reads as intentional if it isn't visibly padded on stage
- [ ] Pareto figure (`eval/figures.py::plot_pareto`) pre-generated and on standby in case the
      live dashboard panel misbehaves
- [ ] **All manifests in the demo grid pre-materialized to GGUF and cached** (spec §9 ggml
      amendment — single backend, no PyTorch demo moments). Every reload the choreography below
      calls for should hit a warm cache and complete in ~1.6 s (measured cold-load figure); if
      any step is doing a cold `.tsra` → GGUF conversion live, that step will stall and the
      "fast enough nobody notices" framing breaks. Verify the cache is warm as part of setup, not
      during the walkthrough.

## Walkthrough

**Framing note (spec §1/§9 amendment): this is no longer a "live, no-reload" demo.** Every
precision or domain switch below is a fast reload (~1.6 s measured) of a different pre-baked GGUF
materialized from the same `.tsra` artifact — not an in-place mmap adjustment. Narrate it that
way; don't say "no reload," say "watch how fast."

1. **Laptop, dongle in.** Plug in. OLED reads `CODE · Q6_K · ~1.4 GB` (figures illustrative until
   M4 measures them — see spec §12; bit-widths now map to the nearest GGUF quant type, e.g. 6-bit
   → `Q6_K`, not an arbitrary continuous value). Paste the real bug, ask Tessera to debug it.
   Narrate the heatmap: MLP tensors (mlp_gate/up/down) run richer than attention — but note for
   Qwen3 specifically MLP is only ~75% of per-layer weight, not the ~88% the original
   Qwen2.5-based scoping assumed (spec §3) — the allocator still spends most of its budget there,
   just less lopsidedly, so check this actually holds after M1 before saying "88%" on stage.
2. **Unplug.** Point at the dashboard: session visibly wipes (`wiped: true`, the zeroize
   confirmation). Say it out loud: "the laptop cannot answer a follow-up right now — nothing of
   the conversation is left on it."
3. **Into the Pi 5 — same model as the laptop, not a smaller one (spec §3 amendment).** OLED
   reads `CODE · Q8_0 · ~1.7 GB` on resume — say out loud that a 4 GB Pi 5 genuinely isn't
   constrained for this model at rest, so it doesn't demote for no reason. Then paste the
   prepared long document to push the session to ~32k tokens. The daemon detects the session's
   KV-cache reservation has outgrown the current profile's headroom and **re-materializes the
   next profile down from the same `.tsra` artifact — a ~1.6 s reload, not a live shrink.**
   Narrate the reload, don't hide it: "watch this — the daemon just noticed we're out of room and
   swapped to a leaner profile, same file, under two seconds." OLED settles around
   `CODE · ~Q6_K · ~2.4 GB` (bytes now include the KV reservation). Ask a follow-up that only
   makes sense with the earlier context (reference something specific from step 1's bug) — the
   reload didn't lose it, because the session lives in the portable bundle (spec §6), not in the
   old process's memory. It answers correctly, at the new precision. The line to land: "this isn't
   a weaker device, it's the same device running out of room because the conversation got long —
   and switching profiles for it is fast enough that it doesn't feel like a hiccup."
4. **Into the Pi Zero 2W.** OLED reads `CHAT · Q3_K · ~250 MB`. *This* is the device-class swap
   — deliberately last, so it doesn't get confused with step 3's budget-pressure reload. This is
   a *different model* (Qwen3-0.6B, not 1.7B) — say so explicitly. Slower, simpler, still
   coherent, still remembers the session. This is the strongest defense of the transport design
   (spec §3): the session is a transcript, not a KV cache, so it survives a model swap, not just
   a precision swap. If asked about headroom on this tier: it's real but thin (~925 tokens of
   fp16 KV cache after loading the model at 3-bit, spec §3) — KV-cache quantization (now critical
   path, not a stretch goal) is what makes it comfortable rather than merely possible.
5. **Ask a math question.** Domain shift detected (or button-toggled, per stretch goal 1 status
   at the time) — the daemon reloads the `math` domain's pre-baked profile, materialized from the
   same `.tsra` artifact the `code` profile came from. Narrate that: "same 1.7 GB file on disk
   covers every domain and every precision tier — that reload just pulled a different slice of
   it, not a different download." Heatmap visibly redistributes toward the tensors that matter
   for arithmetic.

Close on the Pareto plot: every Tessera profile marked above the uniform baseline curve.

## Anticipated questions (see spec.md §14 for the full prior-art table)

- "Isn't this just AWQ/GPTQ?" → one-breath answer in spec.md §14.
- "What if the cross-domain correlation turns out high?" → spec.md §4.3: pivot the pitch onto
  the nested format + device-conditioning, say so honestly.
- "Why not transport the KV cache?" → spec.md §6: size, numerical validity, model portability.
- "Why not a MoE or hybrid model, those benchmark better?" → spec.md §14 model family question:
  the model is the substrate, not the contribution — MoE routing frequency would confound the
  sensitivity sweep itself (§2).
- "Isn't a 4 GB Pi running the same 1.7B model as your laptop kind of a non-demo?" → spec.md §3
  Pi 5 retiering amendment: it would be, at rest — the point of step 3 is that it's not a static
  device profile, it's live budget pressure from a real 32k-token session competing with the
  weights for the same memory. Have the crossover numbers (~11.2k / ~29.9k tokens) ready if
  pushed on whether that's real or staged.
- "Wait, didn't you say precision switches were live, no reload?" → that framing changed (spec §1/
  §9 amendment) once ggml became the required production backend for real memory reduction: GGUF
  is fixed-precision per *file*, though it does support per-*tensor* type overrides within a file
  (`llama-quantize --tensor-type`), which is how the allocator's actual per-tensor decisions still
  deploy unchanged. Precision/domain switches are now fast reloads (~1.6 s measured) of a
  different pre-baked file materialized from the same `.tsra` artifact — say this straightforwardly
  if asked; claiming otherwise on stage is the failure mode judges catch fastest.
- "Why not just run everything on PyTorch, it's simpler?" → spec.md §9: PyTorch carries a fixed
  ~400-600 MB tax regardless of chosen bit-width, which breaks the RSS criterion on Pi tiers and
  doesn't reduce memory at all — measured directly (a Q4_K_M GGUF used 1614 MiB RSS; PyTorch used
  ~3.3 GB regardless of manifest). ggml is required everywhere in the demo for this reason, not
  just on the Pi tiers as originally scoped.

## Numbers to have on a slide, not improvised (spec §13)

Pareto curve, measured RSS per device/profile, tokens/sec per device/profile, resume latency,
cross-domain Spearman matrix, nesting tax, bundle size distribution.
