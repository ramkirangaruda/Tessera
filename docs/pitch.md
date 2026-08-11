# Pitch outline

Working notes for the pitch deck (owned by D, spec §11). Not a script — the demo (§12 /
`demo-script.md`) carries the narrative; this is the argument underneath it.

## The one-liner

Your assistant lives on a hardware key, not a machine. Plug it into any device and the model
reshapes its own precision to fit that hardware and the task at hand. Unplug and it leaves
nothing behind.

## The four claims (spec §1), in pitch order

1. **One artifact serves every precision.** Lead with this — it's the visually legible one (the
   heatmap) and the one that makes "reshapes itself" concrete instead of a slogan. **Reframed
   from runtime to storage (spec §1/§9 amendment):** a 3-bit device and an 8-bit device are
   served from the same 1.7 GB `.tsra` file; the production runtime (ggml, required for real
   memory reduction) materializes a chosen precision from it in under two seconds, measured. The
   comparison that lands this: a community GGUF repo covering the same model's precision range
   ships **24 separate files**; Tessera ships one. Don't claim the switch is instantaneous or
   reload-free — that was the original framing and it doesn't survive the "did the memory
   actually move" question. Claim it's fast enough not to matter (1.6 s measured cold load).
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
- **"Why not just ship five model files?"** That's close to what actually happens at the
  production-runtime layer now (ggml materializes a GGUF per profile, spec §9 amendment) — the
  honest answer is that those files are all *derived* from one 1.7 GB `.tsra` artifact, not five
  independently-shipped downloads. One artifact to build, sign, and distribute to a device; the
  per-profile files are a fast, cached, on-device materialization step, not separate things the
  dongle has to carry or the user has to trust independently.
- **"Didn't you say switching was live, no reload?"** That framing changed (spec §1/§9 amendment):
  ggml's GGUF format is fixed-precision per file, so a switch is a ~1.6 s reload of a different
  file materialized from the same artifact, not an in-place mmap adjustment. Say this plainly if
  asked — the allocator's actual per-tensor bit assignment still deploys unchanged via GGUF's
  per-tensor type override (`llama-quantize --tensor-type`), which is the part that was ever the
  real contribution. True in-place plane-dropping stays a stretch goal on the `.tsra` runtime path
  specifically, not a claim about the ggml production path.
- **"This is a lot of infrastructure for a hackathon."** It is — that's why the repo was built
  ahead of the event; the event is the demo, not the build window (spec header).

## Closing beat

The Pareto plot: every Tessera profile sitting above the uniform-quantization baseline curve.
Not a new SOTA quantizer — better quality per byte, at the budget and task you actually have,
out of one file, on hardware you can carry.
