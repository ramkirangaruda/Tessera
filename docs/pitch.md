# Pitch outline

Working notes for the pitch deck (owned by D, spec §11). Not a script — the demo (§12 /
`demo-script.md`) carries the narrative; this is the argument underneath it.

## The one-liner

Your assistant lives on a hardware key, not a machine. Plug it into any device and the model
reshapes its own precision to fit that hardware and the task at hand. Unplug and it leaves
nothing behind.

## The four claims (spec §1), in pitch order

1. **One artifact serves every precision.** Lead with this — it's the visually legible one (the
   heatmap) and the one that makes "reshapes itself" concrete instead of a slogan. A 3-bit
   device and an 8-bit device read the same file; switching precision is an mmap range change.
2. **Bit allocation is device-conditioned.** Solved against the device's *actual* free memory at
   load time, not a fixed recipe shipped to everyone.
3. **Bit allocation is task-conditioned.** The tensors that matter for code aren't the tensors
   that matter for arithmetic — show the correlation matrix (spec §4.3) as the receipt.
4. **Context is portable and leaves no residue.** The product claim, and the one the demo's
   middle act (unplug → wipe → resume on a different device) exists to prove viscerally.

## Why judges won't have seen this combination (spec §14)

Closest prior art is HAWQ (per-layer bit allocation, but offline/fixed-budget, no nesting, no
task conditioning) and Any-precision LLM / MatQuant (nested/multi-width, but no
device-conditioning, no task conditioning, no portable session). Tessera's claim isn't beating
any one of these on its own axis — it's that no one has shipped the combination, out of a single
artifact, with the session traveling on hardware.

## Anticipated skepticism, addressed head-on

- **"Is the task-conditioning result real?"** Spearman correlation matrix, computed and known
  before the event (spec §4.3) — say the honest number whichever way it lands.
- **"Why not just ship five model files?"** That's the kill-switch fallback (spec §10), not the
  pitch — if asked, be straightforward that nesting is the harder, more interesting bet and
  explain why it's worth it (one artifact, runtime bit-plane drop under memory pressure, no
  re-download when a device's free RAM changes mid-session).
- **"This is a lot of infrastructure for a hackathon."** It is — that's why the repo was built
  ahead of the event; the event is the demo, not the build window (spec header).

## Closing beat

The Pareto plot: every Tessera profile sitting above the uniform-quantization baseline curve.
Not a new SOTA quantizer — better quality per byte, at the budget and task you actually have,
out of one file, on hardware you can carry.
