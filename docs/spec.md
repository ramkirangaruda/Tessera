# Tessera — engineering spec
> Portable, precision-adaptive local LLM. Your assistant lives on a hardware key, not a machine.
> Plug it into any device and the model reshapes its own precision to fit that hardware and the
> task at hand. Unplug and it leaves nothing behind.
**Status:** greenfield. Nothing built yet.
**Deadline shape:** built ahead of a hackathon; the event is the demo, not the build window.
**Team:** 4 people (roles in §11).
---
## 1. The thesis
Standard LLM quantization applies one bit-width to the whole model. Better systems (AWQ, K-quants,
SpQR) use fixed, hand-tuned mixed-precision recipes shipped identically to every user.
Tessera claims three things those don't do:
1. **Bit allocation is task-conditioned.** The tensors that matter for code generation are not the
   same tensors that matter for arithmetic. Measure it, exploit it.
2. **Bit allocation is device-conditioned.** Allocation is solved against the *actual* free memory
   of the machine the model lands on, at load time.
3. **One artifact serves every precision.** Weights are stored as nested bit-planes. A 3-bit device
   and an 8-bit device read the same file — one just stops reading earlier. Switching precision is
   an mmap range change, not a re-quantization.
And one product claim, which is what makes it demoable:
4. **Context is portable and leaves no residue.** A hardware dongle carries the session. Any host
   can resume it; on unplug the host wipes its copy.
---
## 2. Scope
### In scope
- Sensitivity profiler and bit allocator (offline)
- Nested bit-plane weight format + packer + loader
- Inference runtime capable of loading a manifest and generating
- USB dongle firmware, host daemon, handshake protocol
- Live dashboard for the demo
- Evaluation harness producing the Pareto and heatmap figures
### Explicitly out of scope
- Training or fine-tuning anything
- KV-cache transport across devices (see §6 — deliberately rejected)
- Multi-user, cloud sync, or accounts
- Beating state-of-the-art quantization on absolute quality. The claim is *better quality per byte
  at a given device budget and task*, not a new SOTA quantizer.
### Non-negotiable design decisions (do not relitigate)
- Model family is **Qwen3 dense** (`Qwen3-1.7B` primary, `Qwen3-0.6B` lowest tier). Frozen as of
  this amendment — see "Model family amendment" below for the full rationale and the numbers that
  were re-derived (not assumed) before this was locked. `Qwen2.5-1.5B` / `Qwen2.5-0.5B` are kept as
  a **secondary validation run**, not the primary substrate (§13 cross-model replication).
- The nested artifact is **pre-installed on each device**. The dongle carries only manifests +
  session state (kilobytes), never gigabytes of weights.
- Portable context is **transcript + summary + embeddings**, never a raw KV cache.

### Model family amendment (Qwen2.5 → Qwen3, applied before M0)

**Why the switch.** Qwen3 dense models keep the same architecture shape as Qwen2.5 — GQA, SwiGLU,
RoPE, RMSNorm pre-norm — so the format, packer, and allocator are unchanged by this amendment.
`Qwen3-1.7B` has materially more headroom on STEM and coding than `Qwen2.5-1.5B`, which matters
because the task-conditioned sensitivity claim (§1 claim 1) is only measurable if the base model
has real competence in every domain being tested — a domain the model can't do at all doesn't
produce a meaningful sensitivity signal, it produces noise.

**Explicitly rejected, with reasons (a judge will ask):**
- **Qwen3.5 small series (0.8B/2B/4B), hybrid Gated DeltaNet + sparse MoE.** MoE confounds
  sensitivity measurement — a rarely-routed expert reads as low-sensitivity in the sweep and gets
  crushed, then fails catastrophically on the inputs that actually route to it. Sparse activation
  is also not sparse residency, which breaks the budget-vs-footprint narrative (§3's whole point is
  that the allocator spends against what's *resident*). And Gated DeltaNet has no conventional KV
  cache, which invalidates the §6 argument for why the KV cache isn't transported.
- **Gemma 4 E2B/E4B.** Effective-parameter models, not plain dense — same "what's actually
  resident" problem as above. Also ships QAT checkpoints, an awkward substrate for a project whose
  entire contribution is post-training quantization.
- **Principle to hold:** the model is the substrate, not the contribution. Pick the most boring,
  best-characterized dense transformer available, so measured deltas are attributable to the
  allocator alone. Qwen3 dense is that model; the hybrid/MoE options are not.

**Architecture verification (done before M0, not assumed — pulled `config.json` directly):**
both `Qwen3-1.7B` and `Qwen3-0.6B` are `Qwen3ForCausalLM`, `tie_word_embeddings: true`,
28 layers, `head_dim=128`, `num_key_value_heads=8`. Full rebuilt tables are in §3. One real
architectural difference from Qwen2.5 worth flagging: Qwen3 decouples `head_dim` from
`hidden_size / num_attention_heads` (head_dim is fixed at 128 regardless), so `q_proj`/`o_proj`
are not square for `Qwen3-0.6B` the way they were for every Qwen2.5 size. Qwen3 also adds a
per-head `q_norm`/`k_norm` RMSNorm (on `head_dim`, i.e. 128 params each) that Qwen2.5 doesn't
have — negligible bytes, same treatment as every other norm tensor (always fp16, excluded from
the search), but it's two more raw tensors per layer the runtime's state-dict mapping needs to
know about.

**Decision gates hit by this amendment (both triggered — see §3 for the numbers):**
1. KV-cache bytes/token for Qwen3 is **exactly 4.0×** the Qwen2.5-1.5B figure (114,688 vs 28,672
   B/token) — over the 3× threshold, so KV-cache quantization is **promoted from stretch goal
   (old §9 item 4) to the M4 critical path**. See §3 KV cache and the updated §9/§10.
2. `Qwen3-0.6B` **does fit** the Pi Zero 2W's ~350 MB realistically-free budget at 3-bit including
   the embedding table (248.7 MiB, ~101 MiB headroom) — but that headroom is tight enough
   (~925 tokens of fp16 KV cache) that it's KV-cache quantization, not slack, that makes the
   survival tier comfortable. No alternative third tier was needed; see §3 for the full number.
---
## 3. Target models and hardware
| Device | RAM | Model | Profiles |
|---|---|---|---|
| Laptop (dev machine) | 8–16 GB | Qwen3-1.7B | quality, balanced |
| Raspberry Pi 5 | 4 GB (confirmed) | Qwen3-1.7B | quality (short context) → balanced/survival (long context) — retiered by contested memory, not device class; see below |
| Raspberry Pi Zero 2W | 512 MB | Qwen3-0.6B | survival only |
Secondary validation tier (§13 cross-model replication, not part of the live demo): the same
three profiles re-run against `Qwen2.5-1.5B-Instruct` / `Qwen2.5-0.5B-Instruct`.
The Zero cannot host 1.7B at any usable bit-width once the OS is accounted for. It runs 0.6B.
This is a feature, not a compromise: because portable context is a transcript rather than a KV
cache, the session survives a **model swap**, not just a precision swap. Say this out loud in the
pitch — it is the strongest defence of the transport design.

### Pi 5 hardware confirmed — retiering amendment (applied before M0)

**Realistic memory budget** (Raspberry Pi OS Lite 64-bit, headless — see hardware notes below):
~3.2 GB free at the OS level, minus ~150 MB for the daemon + dashboard backend, minus runtime
overhead. **~2.9 GB usable with a ggml/llama.cpp backend, ~2.4 GB with PyTorch** — this gap is
itself why the runtime backend decision below is no longer optional (§15 open decision 6,
resolved by this amendment).

**The problem:** `Qwen3-1.7B` at 8-bit is 1743.3 MiB (≈1.70 GiB, computed §3 table below) — a
4 GB Pi 5 is not genuinely memory-constrained for this model at rest. Forcing the demo down to
4-bit on the Pi 5 for no real reason would be a manufactured constraint, and it's exactly the
kind of thing a judge who's done the arithmetic will call out.

**The fix — retier by contested memory, not device class.** Keep `Qwen3-1.7B` on both laptop and
Pi 5 (no second model, no second sensitivity sweep). On the Pi 5, the demo runs a long session
(target ~30k+ tokens) so the KV cache genuinely competes with the weights for the same budget —
the allocator has to shrink the weights to make room, live, on the same artifact. This is the
existing device-conditioned allocation machinery (§7 `CAPS`/`PROFILE` handshake) doing real work,
not a scripted precision drop.

**The numbers that decide this (computed, not assumed — kv_heads=8, 114,688 B/token = 112.0
KiB/token fp16, per §3 KV cache table below):**

| Context | KV (fp16) | Weight budget left (of 2.9 GB) | KV (8-bit K / 4-bit V, 42.0 KiB/token) | Weight budget left |
|---|---|---|---|---|
| 8k | 896.0 MiB | 2073.6 MiB (8-bit weights fit, 330 MiB spare) | 336.0 MiB | 2633.6 MiB (fits easily) |
| ~11.2k | crosses over | 8-bit weights (1743.3 MiB) no longer fit | — | still fits |
| 16k | 1792.0 MiB | 1177.6 MiB (forces ≤4-bit) | 672.0 MiB | 2297.6 MiB (fits) |
| ~29.9k | — | — | crosses over | 8-bit weights no longer fit |
| 32k | 3584.0 MiB (**over budget alone**) | infeasible | 1344.0 MiB | 1625.6 MiB (forces ~6–7-bit) |

Two crossover points, both real: **without** KV-cache quantization, weights must drop below
8-bit past **~11,200 tokens**; **with** it (8-bit K / 4-bit V, promoted to the M4 critical path
by the previous amendment — §9), the crossover moves out to **~29,900 tokens**, almost exactly
the "30k+" target. **Recommendation: demo the Pi 5 tier at 32k tokens of context** — a round,
defensible number, past both crossovers, that forces a real (not cosmetic) reduction below
8-bit even with KV quantization live. Rewrite demo step 3 (§12) around this: same device, same
`.tsra` artifact, weights visibly shrink on the heatmap as the session's KV reservation grows —
not a device swap, a budget-pressure swap.

**Alternative considered and rejected (reported per the decision-gate instruction, not silently
dropped): move the Pi 5 tier to `Qwen3-4B`.** Verified against the real `config.json` (36 layers,
`hidden_size=2560`, `kv_heads=8`, `head_dim=128`, `intermediate_size=9728`, ~4.02B params — the
user's approximate 2.2 GB@4-bit / 4.2 GB@8-bit figures check out: computed 2157.7 MiB@4-bit /
4075.7 MiB@8-bit). This would make the memory constraint real too (4-bit is a genuine squeeze
against a 2.9 GB budget) — but it costs a **second full sensitivity sweep** (another 4,728
GPU-evaluations, §4.1) for a device tier that isn't even the one under real pressure at rest, and
it's a *weaker* demo: "a bigger model forced to run small" restates the device-conditioning claim
you already have from the Pi Zero tier, where "the same artifact shrinks live as a real session
grows" demonstrates something new — that allocation responds to *live* memory pressure, not just
a static device profile. Rejected in favor of the long-context retier above.

### Pi 5 — required host hardware and OS additions (spec amendment)

Not dongle BOM (§7 — that's the USB key itself) — this is the Raspberry Pi 5 host device.

- **NVMe HAT (official Pi 5 M.2 HAT+ or equivalent) — required.** Justified by the M3 load-time
  criterion (§10): a ~1.1 GB profile at microSD's ~90 MB/s sequential read is ~12.2 s just for
  I/O, already over the 8 s cold-load budget before parse/setup overhead. Over the Pi 5's stock
  PCIe Gen2 x1 NVMe link (~450 MB/s sustained is a realistic, commonly-cited figure for this HAT
  class), the same read is ~2.4 s — comfortably inside budget. Pi Zero 2W is unaffected (no PCIe,
  microSD only) and doesn't need this: its profiles are ~250–320 MiB (§3 Pi Zero fit check), and
  250 MB at 90 MB/s is ~2.9 s — already well under 8 s on microSD alone.
- **Active cooling (official Pi 5 Active Cooler or equivalent) — required.** Sustained inference
  throttles the BCM2712 SoC without it, which makes the tokens/sec figures in §13 non-reproducible
  run to run — a fan isn't cosmetic here, it's what makes the throughput numbers trustworthy.
- **Target OS: Raspberry Pi OS Lite 64-bit, headless.** No desktop environment; the dashboard is
  served over the network to whatever's projecting it, not rendered locally. Minimize CMA
  (contiguous memory reserved for the GPU/display path) in `config.txt` since no display is
  attached — reduce or remove the `dtoverlay=vc4-kms-v3d` CMA reservation (default is commonly
  ~256 MB on Pi 5; a fully headless box needs little to none of it). **This reclaim estimate is
  unverified — it needs to be measured on the actual device** (`vcgencmd get_mem reloc_total` /
  `/proc/meminfo` before and after the config change) at M4/M5, not assumed from general Pi
  tuning knowledge the way it's stated here. Report the real reclaimed figure once measured; the
  ~2.9 GB / ~2.4 GB budgets above already treat this as done, so an underperforming reclaim
  tightens every number in this section and should be re-checked against them.

### Qwen3-1.7B / Qwen3-0.6B architecture
Pulled from `config.json` directly, not assumed — verify again if either checkpoint is updated
upstream.
```
                     Qwen3-1.7B   Qwen3-0.6B
layers               28           28
hidden_size          2048         1024
attn heads           16           16       head_dim 128 (both — fixed, decoupled from hidden_size)
kv heads             8            8        (GQA)
intermediate_size    6144         3072
vocab                151936       151936
tied embeddings      yes          yes
```
Note `head_dim=128` is fixed for both sizes rather than derived as `hidden_size / heads` (that
was true for every Qwen2.5 size, isn't for Qwen3-0.6B: `16 × 128 = 2048 ≠ hidden_size(1024)`), so
`attn_q`/`attn_o` are not square for the 0.6B model. Both sizes also have identical KV geometry
(28 layers × 8 kv_heads × 128 head_dim) — model size scales `hidden_size`/`intermediate_size`
only, so **KV-cache cost per token is identical across both tiers** (see below), unlike weight
footprint which scales with the model.

Parameter accounting per layer — **Qwen3-1.7B**:
| Tensor | Shape | Params |
|---|---|---|
| `attn_q` | 2048 × 2048 | 4.19 M |
| `attn_k` | 2048 × 1024 | 2.10 M |
| `attn_v` | 2048 × 1024 | 2.10 M |
| `attn_o` | 2048 × 2048 | 4.19 M |
| `mlp_gate` | 2048 × 6144 | 12.58 M |
| `mlp_up` | 2048 × 6144 | 12.58 M |
| `mlp_down` | 6144 × 2048 | 12.58 M |
| **total** | | **~50.33 M** |
× 28 layers = ~1.409 B, plus `token_embd` at 151936 × 2048 = **311.16 M** (~18.1% of the whole
model in one tensor — up from ~15% on Qwen2.5-1.5B, since `hidden_size` grew but `vocab` didn't).
Total ≈ **1.720 B** (Qwen3-1.7B naming checks out).

Parameter accounting per layer — **Qwen3-0.6B**:
| Tensor | Shape | Params |
|---|---|---|
| `attn_q` | 1024 × 2048 | 2.10 M |
| `attn_k` | 1024 × 1024 | 1.05 M |
| `attn_v` | 1024 × 1024 | 1.05 M |
| `attn_o` | 2048 × 1024 | 2.10 M |
| `mlp_gate` | 1024 × 3072 | 3.15 M |
| `mlp_up` | 1024 × 3072 | 3.15 M |
| `mlp_down` | 3072 × 1024 | 3.15 M |
| **total** | | **~15.73 M** |
× 28 layers = ~440.4 M, plus `token_embd` at 151936 × 1024 = **155.58 M** (~26.1% of the whole
model — the smaller the model, the more `token_embd` dominates, since vocab is shared across
sizes). Total ≈ **0.596 B** (Qwen3-0.6B naming checks out).

**Two consequences that shape the whole allocator (unchanged from the original Qwen2.5 analysis,
now re-verified against real Qwen3 numbers):**
- MLP tensors are ~75% of per-layer weight on Qwen3-1.7B (37.75 M / 50.33 M — down from ~88% on
  Qwen2.5-1.5B, because Qwen3's GQA is less aggressive: 8 kv_heads vs Qwen2.5's 2, which grows
  `attn_k`/`attn_v` relative to MLP). MLP is still where most of the bytes are, just less
  lopsidedly than before — worth re-checking after M1 rather than assuming the old 88% figure.
- `token_embd` is bigger relative to the whole model than it was on Qwen2.5 (18–26% depending on
  size, vs ~15%). It must be in the search space, not pinned by convention — more true now, not
  less.

**Quantizable tensor count:** 7 per layer × 28 + `token_embd` = **197 knobs** — verified unchanged
from the Qwen2.5 figure (the new `q_norm`/`k_norm` tensors are 1-D and excluded from the search
the same way every other norm tensor is, so they don't add knobs; they do add 2 more raw-fp16
tensors per layer the runtime's state-dict mapping must cover).
LayerNorm/`q_norm`/`k_norm` weights and biases stay fp16 — negligible bytes, high sensitivity.

### KV cache
`2 (K,V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes = 114,688 B/token ≈ 112 KiB/token`
— identical for both Qwen3-1.7B and Qwen3-0.6B (KV geometry doesn't scale with model size here).
This is **exactly 4.0×** the Qwen2.5-1.5B figure (114,688 vs 28,672 B/token), because Qwen3 uses
8 kv_heads where Qwen2.5-1.5B used only 2 — less aggressive GQA.

| Context | Qwen3 fp16 (either size) | Qwen3 quantized (8-bit K / 4-bit V, 42.0 KiB/token) | Qwen2.5-1.5B fp16 (reference) |
|---|---|---|---|
| 2k | 224.0 MiB | 84.0 MiB | 56.0 MiB |
| 8k | 896.0 MiB | 336.0 MiB | 224.0 MiB |
| 32k | 3584.0 MiB | 1344.0 MiB | 896.0 MiB |
Pi 5 retiering (below) is built directly on this table — see the crossover analysis there for
why ~30k tokens is the number to design the Pi 5 demo tier around.

**Decision gate (spec amendment, non-negotiable): 4.0× ≥ the 3× threshold, so KV-cache
quantization is promoted from stretch goal to the M4 critical path** — see §9/§10. Weight
quantization alone does not solve on-device memory even at the best of times; on Qwen3 it solves
noticeably less of it than it did on Qwen2.5.

**Pi Zero 2W fit check (decision gate, computed not assumed):** `Qwen3-0.6B` fully quantized
(all 197 tensors + `token_embd`) at the allocator's 3-bit floor-adjacent budget:
- 2-bit floor: 177.6 MiB · 3-bit: 248.7 MiB · 4-bit: 319.7 MiB
Against a ~350 MB realistically-free budget, 3-bit fits with **~101 MiB headroom**. That headroom
converts to KV cache at 112 KiB/token: **~925 tokens** of fp16 KV cache before the device is out
of memory (~1,575 tokens at the 2-bit floor, ~172 MiB headroom). This *does* fit — no alternative
third tier is needed — but the margin is thin enough that it's real evidence for, not just
motivation for, promoting KV-cache quantization to the critical path: 8-bit K / 4-bit V roughly
halves that per-token cost, meaningfully widening the survival tier's usable context window. The
bounded warm-prefill design (§6 — constant-size prefill regardless of session length) is what
keeps typical turns well inside this budget even before KV quantization lands; it was already the
right call for a different reason (§6), and this amendment is further confirmation of it.
---
## 4. Component 1 — the precision compiler
Offline. Runs on the dev machine / any GPU. Produces manifests.
### 4.1 Sensitivity sweep

**Metric correction (spec amendment — the original two-metrics-per-domain table below was
wrong as written):** the sweep does **not** call HumanEval or GSM8K per tensor per bit-width.
That's 4,728 *generative* evaluations — sampling variance on top of a measurement that's
already trying to detect small deltas, and orders of magnitude more compute than a forward pass.
Two-tier scheme instead:

- **Sweep metric (all 4,728 evaluations): perplexity on domain-specific held-out text.**
  Forward pass only, deterministic, no sampling. This is what every tensor × bits × domain
  combination measures.
- **Confirmation metric (~20 evaluations total, run once, on M2's output, not per-tensor):**
  HumanEval pass@1 / GSM8K exact match, run only on the *final allocations* the allocator
  actually picks for each domain × budget tier — i.e., "does the manifest the allocator
  produced actually perform," not "does every individual tensor perform." See §4.2.

**Proxy validation — run before the full sweep, not after (spec amendment).** The sweep metric
being perplexity is only justified if perplexity delta actually predicts task-quality delta.
Before committing to an unattended overnight run:
1. Pick 15 tensors stratified across the expected sensitivity range (mix of layer depth,
   attn vs. mlp, and `token_embd`) — not a random sample, a deliberately spread one.
2. Fake-quantize each to one representative low bit-width (3-bit — low enough that damage is
   visible, high enough not to be degenerate).
3. Measure **both** delta-ppl (domain held-out text) and delta-pass@1 (HumanEval) for each —
   ~30 evaluations total.
4. Compute the Spearman correlation between the two rankings.
5. **If they correlate: record the number, it's a slide, proceed to the full sweep. If they
   don't: stop and report before running the full sweep** — the proxy isn't justified and
   4,728 perplexity deltas would be measuring the wrong thing.

For each tensor `t` in the 197, and each candidate bit-width `b ∈ {2, 3, 4, 5, 6, 8}`:
1. Fake-quantize **only** `t` to `b` bits (quantize → dequantize back to fp16 in-place). Everything
   else stays fp16.
2. Evaluate perplexity on the domain's held-out text (not the calibration set — see below).
3. **Assert bitwise equality of every other tensor against a pristine fp16 copy, every
   iteration, not sampled** (spec amendment, pre-flight check 1). Accumulated quantization
   damage across the sweep from an incomplete restore is silent and corrupts every measurement
   downstream of the first occurrence — this is the cheapest possible check against the most
   expensive possible failure mode.
4. Record `Δppl` and `bytes_saved`, checkpointed incrementally (see pre-flight below).
Fake quantization is ~30 lines of PyTorch. Do not fight real kernels at this stage — the sweep is
about *ranking*, and a quant-dequant round trip has the same numerics as the real thing.
**Grouping:** per-channel groups of 128 along the input dimension, asymmetric (scale + zero point).
Group size is a config knob; 128 is the default.

**Calibration source vs. held-out eval text — deliberately different data (spec amendment,
generalizing the distinction the original table only stated for `chat`):**
| Domain | Calibration source (what gets fake-quantized against) | Held-out text (what ppl is measured on) |
|---|---|---|
| `chat` | OpenAssistant | WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split — same set as the M0 baseline) |
| `code` | The Stack (Python subset), first N samples of `train` | The Stack, next disjoint N samples of `train` |
| `math` | GSM8K `train` split | GSM8K `test` split |
| `summ` | CNN/DailyMail `train` split | CNN/DailyMail `validation` split |
Confirmation-metric sources (used once, post-M2, not per sweep iteration): HumanEval (code),
GSM8K test subset (math) — same GSM8K test split as the held-out ppl set, since exact-match and
perplexity aren't in tension the way calibration-vs-eval leakage would be.

**Cost:** 197 tensors × 6 widths × 4 domains = 4,728 evaluations. Each is a forward pass over the
held-out set on a ~1.7B model — seconds on a GPU. Parallelize across tensors; this can run
unattended overnight. **Start this early.** It is the long pole and it is embarrassingly parallel.

**Pre-flight checks — all must pass before the long run starts (spec amendment):**
1. **Weight restoration assertion**, above — every iteration, not sampled.
2. **Incremental checkpointing.** Write to `sensitivity.parquet` after each evaluation, keyed on
   `(tensor_name, bits, domain)`. The run must be resumable from partial results — a crash at
   evaluation 3,000 of 4,728 should cost zero completed work, not restart from scratch.
3. **Noise floor.** Before trusting any delta as signal: measure one tensor twice at the same
   bit-width (repeat), and measure one tensor expected to be insensitive (e.g., a late-layer
   `attn_v`). **If the insensitive tensor's delta sits inside the repeat measurement's
   run-to-run variance, the held-out sets are too small to resolve real signal — report this
   and stop**, don't run 4,728 evaluations that can't be trusted.
4. **Stratified ordering.** Iterate width-major and domain-major, not tensor-major — i.e., the
   outer loops are `(domain, bits)` and the inner loop is over tensors, not the reverse. This
   way a partial/interrupted run has *complete slices* (every tensor at some width/domain
   combos) rather than *fragments* (some tensors done at all widths, others untouched).
5. **Lock and commit the eval methodology.** Stride, context length (2048), dataset revision
   (`Salesforce/wikitext`, pinned per M0), tokenizer (`Qwen/Qwen3-1.7B`'s, `enable_thinking`
   verified off — see below). Record this in the repo (`compiler/sweep_methodology.json`,
   written alongside `sensitivity.parquet`). **All 4,728 deltas are relative to the 16.7835
   WikiText-2 baseline (§10 M0) and are invalidated if any of these change** — this is what
   makes a re-run six months from now comparable to today's, or tells you honestly that it
   isn't.

**Non-negotiable eval config: `enable_thinking=False` for every sweep and eval run, asserted in
the harness, not just defaulted.** Qwen3 has a hybrid thinking mode; a reasoning trace on every
one of the 4,728 evaluations would inflate the sweep by orders of magnitude and inject variance
that swamps the quantization signal you're actually trying to measure. This is a chat-template
kwarg (`tokenizer.apply_chat_template(..., enable_thinking=False)`), verified present on both
`Qwen3-1.7B` and `Qwen3-0.6B`'s chat templates — assert it's actually being passed rather than
trusting a default, since a silent default flip would be a very expensive bug to discover after
an overnight sweep. (Note: the sweep's perplexity measurement is a forward pass over raw text,
not a chat completion, so `enable_thinking` doesn't directly apply to the 4,728 ppl evaluations
themselves — it matters for the confirmation-metric and proxy-validation generative evals, which
do go through the chat template.)

**On the M0 baseline (16.7835) and cross-tokenizer comparison (spec amendment):** this is higher
than published perplexities for similarly-sized Llama-family models. Expected, not a bug —
Qwen's 151,936-token vocabulary yields fewer tokens per word than Llama's ~32K vocabulary, and
per-token perplexity rises as vocabulary grows (more plausible next-tokens to spread probability
mass over). **Never compare perplexity across tokenizer families in any output or figure.** The
`Qwen2.5` secondary validation run (§13) *is* directly comparable to the Qwen3 numbers — same
151,936-token vocabulary — and that comparison should be made explicitly; a Qwen3-vs-Llama
comparison should not appear anywhere in the deck.

**Output:** `sensitivity.parquet`
```
tensor_name, domain, bits, delta_ppl, bytes_at_bits, bytes_saved_vs_fp16
```
(`delta_task_metric` removed from the per-row schema — task metrics are now a separate,
much smaller table produced by the confirmation-metric step in §4.2, keyed on manifest/profile,
not on individual tensor.)
### 4.2 Allocator
Given `(domain, budget_bytes)`, choose a bit-width per tensor minimizing total damage subject to
total bytes ≤ budget.
This is a multiple-choice knapsack. Solve with a Lagrangian sweep: for a multiplier `λ`, pick for
each tensor independently the `b` minimizing `damage(t,b) + λ · bytes(t,b)`. Binary-search `λ`
until total bytes hits the budget. Fast, near-optimal, ~50 lines.
Guardrails the allocator must respect:
- LayerNorms and biases are always fp16, excluded from the search.
- Hard floor of 2 bits on any tensor.
- Optional: cap the number of distinct bit-widths per model to keep the loader simple.
**Output:** a profile manifest.
```json
{
  "profile_id": "code@1100MB",
  "schema_version": 1,
  "model": "qwen3-1.7b",
  "artifact_sha256": "…",
  "domain": "code",
  "budget_bytes": 1153433600,
  "measured_bytes": 1149203968,
  "allocation": {
    "token_embd": 6,
    "blk.0.attn_q": 4,
    "blk.0.mlp_down": 6,
    "…": 0
  },
  "eval": { "wikitext_ppl": 11.24, "humaneval_pass1": 0.281 }
}
```
Manifests are ~3–5 KB of JSON. Generate the full grid: 4 domains × 4 budget tiers = 16 manifests,
plus uniform baselines for comparison.

**Confirmation metric (spec amendment, §4.1): the manifest's `eval` block (`wikitext_ppl`,
`humaneval_pass1` above) is populated once per manifest, after M2 produces it** — not
accumulated from the sweep. ~16 manifests × ~1-2 confirmation evals each ≈ the "~20 evaluations
total" figure in §4.1. This is the only place HumanEval/GSM8K generative evaluation happens in
this milestone.

**M2 status (spec amendment — do not overclaim this milestone):** "beats uniform quantization on
synthetic sensitivity data" is not validation. A knapsack allocator beats uniform allocation on
*any* synthetic sensitivity data with per-tensor variance — that's a property of the optimization
problem, not evidence the allocator is doing anything useful on real damage numbers. **M2 stays
marked scaffolded, not validated, until it runs against real `sensitivity.parquet` output from
M1** and the confirmation metrics above are measured on the manifests it actually produces.
### 4.3 The result that matters
Compute the **rank correlation between domain sensitivity orderings** (Spearman, code vs math vs
chat). If that correlation is low, task-conditioned allocation is justified and you have a genuine
finding. If it is high, the honest move is to say so and pivot the emphasis onto the nested format
and the hardware. Decide this before the event — do not discover it on stage.
---
## 5. Component 2 — the nested bit-plane format
The clever bit. One file, readable at any bit-width from 2 to 8, without re-quantization.
### 5.1 Encoding
Per group of 128 weights:
1. Quantize to 8-bit unsigned with an asymmetric scale/zero: `q = clamp(round(w/s + z), 0, 255)`
2. Decompose `q` into 8 bit-planes, **MSB first**. Plane 0 holds bit 7 of every weight in the
   group, plane 1 holds bit 6, and so on.
3. Store planes contiguously per tensor: all of plane 0, then all of plane 1, …
Reading planes `0..k-1` yields the top `k` bits of `q`. Dequantize with a midpoint correction:
```
q_k     = value assembled from k planes           # in [0, 2^k - 1]
q_est   = q_k · 2^(8-k) + 2^(7-k)                 # midpoint of the represented interval
w_hat   = (q_est - z) · s
```
**Refinement scales (do this second).** A per-`k` scale `s_k` per group, fitted to minimize MSE at
that truncation level, materially closes the gap to independently-optimized quantization. Cost:
8 × 2 bytes per group = 0.125 bits/weight per level, 1 bit/weight if you store all eight. Start
with derived scales (above), add fitted `s_k` as the first optimization.
### 5.2 File layout (`.tsra`)
```
[ magic "TSRA" | u32 version | u32 tensor_count | u64 dir_offset ]
[ plane data — contiguous per tensor, plane-major                ]
[ scale data                                                     ]
[ tensor directory                                               ]
    per tensor:
      name (len-prefixed utf8)
      shape[], dtype, group_size, bits_max
      plane_offsets[8]  (u64)
      plane_size_bytes  (u64)
      scale_offset      (u64)
```
Contiguity is the whole point: loading a tensor at `k` bits is a single `mmap` range
`[plane_offsets[0], plane_offsets[k])`. No seeking, no scatter-gather, page-cache friendly.
### 5.3 Runtime bit-plane drop
Because planes are ordered MSB-first and stored contiguously, a running process under memory
pressure can **release the tail planes** and continue generating at lower precision. Implement this.
It is a 15-second segment of the demo and there is no cheaper way to look impressive.
### 5.4 Deliverables
- `format/spec.md` — this section, formalized
- `format/pack.py` — fp16 safetensors → `.tsra`
- `format/load.py` — mmap loader taking a manifest, returning materialized tensors
- Round-trip test: loading at `k` bits must match the reference `k`-bit fake-quant within 1e-3 MSE
---
## 6. Component 3 — portable session (what travels)
### What is NOT transported, and why
The KV cache. Three independent reasons, all of which a judge may probe:
1. **Size.** ~112 KiB/token on Qwen3 (§3 — 4× the original Qwen2.5 estimate this argument was
   first written against); ~3.5 GB at 32k context. Over USB CDC this is minutes, not seconds — more
   true now than when this was scoped, not less.
2. **Numerical validity.** A KV cache is an activation produced by *specific* weights. A cache from
   a 4-bit forward pass fed into a 6-bit forward pass is unprincipled and degrades silently.
3. **Model portability.** It cannot survive the 1.7B → 0.6B switch at all.
### What IS transported
```
bundle/
  meta.json          # session_id, created_at, model_family, last_device, turn_count
  transcript.jsonl   # {role, content, ts} per turn
  summary.md         # rolling semantic summary, regenerated every N turns
  memory.npz         # embeddings + text spans for retrieval
  manifests/*.json   # the profile grid
```
Total size: kilobytes to low megabytes. Precision-agnostic. Model-agnostic.
### Bounded warm prefill
On resume, do **not** prefill the entire transcript — that cost grows without bound and is brutal
on a Pi Zero. Instead prefill:
```
system prompt + summary.md + last N turns verbatim  (N ≈ 6, tuned)
+ top-k retrieved spans from memory.npz relevant to the incoming query
```
Constant prefill cost regardless of session length. This is the correct engineering answer and it
should be stated as such, not hidden.
### Encryption
Whole bundle sealed with AES-256-GCM. The data key is wrapped by the ATECC608A secure element on
the dongle; the raw private key never leaves the chip. Host receives a decrypted bundle in memory
only, never on disk.
---
## 7. Component 4 — the hardware
### Bill of materials
| Part | Purpose |
|---|---|
| RP2040 (Pico) or ESP32-S3 | MCU; USB composite device (MSC + CDC) |
| ATECC608A | Secure element — key storage, ECDH, signing |
| SSD1306 128×64 OLED (I²C) | Live readout: `CODE · 5-bit · 1.1 GB` (illustrative — real figures come from M4 measurement on Qwen3, see §3/§12) |
| WS2812 RGB LED | State: idle / handshake / streaming / active |
| Momentary button | Cycle Quality / Balanced / Survival |
| SPDT slide switch | **Hardware write-protect.** Physical read-only. |
| microSD or onboard flash | Bundle storage |
The write-protect switch is worth building even though it is trivial. "Plug this into a stranger's
laptop and it *physically cannot* write to your context" is a security claim you can demonstrate by
flipping a switch, and judges remember physical affordances.
### Handshake protocol
Length-prefixed frames over the CDC endpoint. `[u16 len][u8 type][payload][u32 crc]`
| # | Direction | Message | Payload |
|---|---|---|---|
| 1 | host → dongle | `HELLO` | protocol version, host pubkey |
| 2 | dongle → host | `CHALLENGE` | 32-byte nonce |
| 3 | host → dongle | `AUTH` | signature over nonce |
| 4 | — | *ECDH → session key* | |
| 5 | host → dongle | `CAPS` | `{ram_free, has_gpu, backend, mem_bw_mbps, thermal_ok}` |
| 6 | dongle → host | `PROFILE` | chosen `profile_id` + full manifest |
| 7 | dongle → host | `SESSION_BEGIN` / `CHUNK`* / `END` | encrypted bundle |
| 8 | host → dongle | `STATE_PUSH` | updated encrypted bundle (on eject) |
| 9 | dongle → host | `WIPE_ACK` | |
Profile selection lives on the dongle: it holds the manifest grid and picks by matching `ram_free`
against `measured_bytes` plus a safety margin, filtered by the domain the user selected with the
button (or auto-detected by the host, see §9).
### Host daemon
Python. `pyserial` for the CDC link, `FastAPI` + websockets to feed the dashboard.
Responsibilities: device detect, handshake, decrypt to memory, hand the manifest to the runtime,
watch for eject, push state back, **wipe** (zeroize the in-memory bundle, clear any temp files,
emit a visible confirmation to the dashboard).
The wipe must be *visible*. Step 2 of the demo is the laptop clearing the session on screen.
---
## 8. Component 5 — dashboard
React + Vite. Three panels, all live over websocket:
1. **Memory budget bar** — device free RAM, model footprint, headroom. Updates on profile switch.
2. **Allocation heatmap** — 28 rows (layers) × 7 columns (tensor types), each cell coloured by
   assigned bit-width, plus a separate cell for `token_embd`. *This is the single most important
   visual in the project.* It makes the entire thesis legible in two seconds. Build it early and
   make it beautiful.
3. **Pareto plot** — quality vs memory. Plot the uniform-quantization baseline curve (Q2/Q3/Q4/Q6/Q8)
   as a line, and mark the current Tessera profile as a point. The point should sit above the line.
   Animate it moving when the profile switches.
Plus a header strip mirroring the OLED: device name, domain, bit profile, footprint.
---
## 9. Stretch goals (build only after §4–§8 are green)
Ranked by demo value per unit effort:
1. **Automatic domain detection** — classify the incoming query into one of the four domains and
   hot-swap the manifest. Turns step 5 of the demo from a manual toggle into apparent intelligence.
2. **Fitted per-`k` refinement scales** (§5.1) — closes the nesting quality tax.
3. **Runtime bit-plane drop under simulated memory pressure** — allocate a ballast buffer, watch the
   model shed a plane and keep going.

**No longer a stretch goal — promoted to the M4 critical path** (spec amendment, §3 decision
gate): KV-cache quantization (8-bit K, 4-bit V is the usual sweet spot). Qwen3's KV-cache cost is
4.0× the figure this was originally scoped against, which crossed the promotion threshold. See §10.

**No longer a stretch goal — promoted to a hard M4 dependency on Pi 5 and Pi Zero 2W (spec
amendment, resolves §15 open decision 6):** the ggml/llama.cpp backend. PyTorch carries a fixed
~400–600 MB process tax before any weights load — on an 8–16 GB laptop that's noise, but on a
2.9 GB (ggml) / 2.4 GB (PyTorch) Pi 5 budget it's the difference between the M4 RSS criterion
being meaningful or not (see §10), and on the Pi Zero's ~350 MB survival tier it doesn't fit at
all. **PyTorch remains the laptop backend** (better iteration speed during development, budget is
roomy enough that the tax doesn't matter); **ggml/llama.cpp is required on both Pi tiers.**

**Recorded but not building (future work, spec amendment):** per-expert bit allocation for MoE
models. Routing frequency confounds sensitivity measurement the same way it did for the rejected
Qwen3.5 hybrid option (§2) — a rarely-routed expert looks insensitive in a sweep that never
exercises it, then fails badly on the inputs that do route to it. Would need a routing-frequency-
weighted sensitivity metric, not the flat per-tensor sweep this project uses. Out of scope for
this build; noted here so it isn't rediscovered as if new.
---
## 10. Milestones and acceptance criteria
Ship in this order. Each milestone has a binary acceptance test.
| # | Milestone | Acceptance criteria |
|---|---|---|
| **M0** | Repo + eval harness | Qwen3-1.7B loads with `enable_thinking=False` asserted; baseline WikiText-2 ppl and HumanEval subset reproduce to ±0.05 / ±1 problem across two runs |
| **M1** | Fake-quant sweep | `sensitivity.parquet` populated for 197 tensors × 6 widths × 4 domains on Qwen3-1.7B; no NaNs; per-domain Spearman correlations computed |
| **M2** | Allocator | Given (domain, budget) emits a valid manifest; at equal bytes, beats uniform quantization on ≥2 of 3 metrics for at least 3 of 4 domains |
| **M3** | Bit-plane format | `.tsra` round-trips; loading at `k` planes matches reference `k`-bit fake-quant within 1e-3 MSE; load time for a 1.1 GB profile under 8 s cold **on Pi 5 with the required NVMe HAT (spec amendment — microSD's ~90 MB/s makes 8 s unachievable for a profile this size, ~12.2 s just for I/O; NVMe's ~450 MB/s brings it to ~2.4 s). Pi Zero is unaffected and stays on microSD — its ~250–320 MiB profiles clear 8 s there without help** |
| **M4** | Runtime | Generates coherent text at quality / balanced / survival; tokens/sec recorded per device per profile; **KV-cache quantized (8-bit K / 4-bit V) and included in the RSS/footprint measurement on Pi 5 and Pi Zero — promoted from stretch goal by the §3 decision gate (4.0× the originally-scoped KV cost), not optional for M4 sign-off on those two devices.** **RSS criterion is backend-conditioned (spec amendment — PyTorch's ~400–600 MB fixed tax makes a flat 5% bound meaningless on a 2.4–2.9 GB device budget): on Pi 5/Pi Zero (ggml/llama.cpp, now required — §9/§15), measured RSS within 5% of (`measured_bytes` weights + KV-cache bytes at the session's actual context length); on laptop (PyTorch), RSS is reported for information, not gated at 5%.** Pi 5 tier specifically must also demonstrate the §3 long-context retier: weights visibly shrink as a session's KV reservation grows past the ~11.2k (unquantized KV) / ~29.9k (quantized KV) crossover, at a fixed device profile — not a device swap |
| **M5** | Firmware + protocol | Full handshake completes; bundle round-trips encrypted; OLED and LED reflect state; write-protect switch actually blocks writes |
| **M6** | Daemon + wipe | Unplug wipes host state verifiably; replug on a *different* device resumes the conversation with correct context |
| **M7** | Dashboard + demo rig | The 5-step choreography (§12) runs end-to-end **twice consecutively** with no human intervention beyond plugging and typing |
**Critical path:** M1 → M2 → M3 → M4. M5/M6 can develop in parallel against a mocked bundle from
day one — do not let firmware wait on the compiler.
**Kill-switch decision point:** if M3 is not green two weeks before the event, drop the nested
format. Fall back to five pre-built mixed-precision model files selected by profile, and demote
nesting to a measured stretch goal demonstrated on a single tensor. Make this call on the calendar,
not on vibes.
---
## 11. Repo structure and stack
```
tessera/
  compiler/
    sweep.py            # fake-quant sensitivity sweep
    allocate.py         # Lagrangian knapsack
    calib/              # calibration set loaders
  format/
    spec.md
    pack.py
    load.py
    test_roundtrip.py
  runtime/
    model.py            # manifest-driven Qwen loader
    generate.py
  daemon/
    usb.py              # CDC link, framing
    protocol.py
    crypto.py
    server.py           # FastAPI + websockets
  firmware/
    src/                # RP2040, TinyUSB composite
  dashboard/            # React + Vite
  eval/
    harness.py
    figures.py          # Pareto, heatmap, correlation matrix
  docs/
    demo-script.md
    pitch.md
```
**Stack:** Python 3.11, PyTorch 2.x, `transformers`, `safetensors`, `datasets`, `lm-eval-harness`,
`pyserial`, `FastAPI`. Firmware in C++ (Arduino-Pico core, TinyUSB). Dashboard React + Vite +
Recharts.
**Team split (4 people):**
- **A — compiler:** M1, M2, the sensitivity result, the correlation analysis
- **B — format + runtime:** M3, M4 (including KV-cache quantization — now critical path, §3/§9),
  stretch goals 2 and 3
- **C — hardware:** M5, M6, firmware, protocol, daemon
- **D — dashboard + demo:** M7, figures, pitch deck, running the rig
D is not a spare part. Half of judging is whether the room can *see* what is happening.
---
## 12. Demo choreography
Three devices on the table, one dongle, dashboard projected. Rehearse this until it is muscle
memory. OLED figures below are illustrative placeholders — real per-device footprints come out of
M4 measurement on Qwen3 (§3 has the computed weight-only figures at 2/3/4-bit; the OLED shows
weights + KV cache + runtime overhead, which isn't pinned down until M4 actually runs).
1. **Laptop, dongle in.** OLED: `CODE · 6-bit · ~1.4 GB` (Qwen3-1.7B). Debug a real code snippet —
   pull one from an actual project, not a toy. Heatmap shows MLP tensors running rich.
2. **Unplug.** Laptop visibly wipes the session on screen. Dashboard shows the zeroize.
3. **Into the Pi 5 — same model, contested memory, not a device swap (spec amendment, §3).**
   Resumes at quality/8-bit (`CODE · 8-bit · ~1.7 GB`) — a 4 GB Pi 5 genuinely isn't constrained
   for `Qwen3-1.7B` at rest, and pretending otherwise is the thing a judge would catch. Then push
   the session long: paste in the surrounding file(s) / a stack trace / a few related modules —
   enough real material to land around **32k tokens** of context (spec §3 crossover: ~11.2k
   without KV-cache quantization, ~29.9k with it — 32k clears both with margin). Narrate the
   heatmap redrawing leaner **live**, on the same `.tsra` artifact, as the KV-cache reservation
   grows: OLED settles around `CODE · ~7-bit · ~2.4 GB` (weights + quantized KV, illustrative
   pending M4 measurement). Ask a follow-up that's only answerable from the earlier context — it
   still answers correctly, at the new precision. The point being made: allocation responds to
   *live* memory pressure, not a static per-device recipe.
4. **Into the Pi Zero 2W.** OLED: `CHAT · 3-bit · ~250 MB` (Qwen3-0.6B). Different model entirely
   — *this* is the device-class swap, deliberately saved for last so it isn't confused with step
   3's budget-pressure swap. Slower, simpler, still coherent, still remembers. Call out that the
   context survived a model swap — and that this tier is exactly why KV-cache quantization got
   promoted to the critical path (§3 Pi Zero fit check): the headroom here is real but thin.
5. **Ask a math question.** Domain shift detected; manifest swaps to `math`. Same file on disk, more
   planes on the tensors that matter for arithmetic. Heatmap visibly redistributes.
Close on the Pareto plot with all profiles marked above the uniform baseline curve.
---
## 13. Numbers to have ready
Collect these as you go, not the night before.
- Pareto: Tessera vs uniform Q2/Q3/Q4/Q6/Q8 on WikiText-2 ppl, MMLU subset, HumanEval subset
- Measured RSS per device per profile (real, not theoretical)
- Tokens/sec per device per profile
- Resume latency: handshake + transfer + warm prefill, vs cold restart from scratch
- Cross-domain Spearman correlation matrix (§4.3) — the headline result if it comes out low
- Nesting tax: nested `k`-bit vs independently optimized `k`-bit, per width
- Bundle size distribution across session lengths
- **Cross-model replication (spec amendment):** do the domain sensitivity rankings from
  `Qwen3-1.7B` reproduce on `Qwen2.5-1.5B` (§2's secondary validation run)? Report the Spearman
  correlation between the two models' per-domain tensor-sensitivity rankings. If they replicate,
  this upgrades the task-conditioning finding from a checkpoint quirk to a property of the
  architecture class — a headline result, not a footnote. Requires the Qwen2.5 sweep to actually
  run (not just be scaffolded) — budget time for it precisely because this number is worth having.
- **Pi 5 weight/KV crossover context length, measured (spec amendment):** the computed figures
  (~11.2k tokens without KV-cache quantization, ~29.9k with it — §3) are derived from config.json
  and the allocator's byte formulas, not measured on-device. Confirm them against real M4
  measurement on the actual Pi 5 + NVMe HAT before the event — if the measured crossover is far
  off from the computed one, the demo's 32k-token target (§12 step 3) needs to move with it.
- **CMA reclaim on the Pi 5, measured (spec amendment):** `vcgencmd get_mem reloc_total` /
  `/proc/meminfo` before and after the headless config change (§3) — the ~2.9 GB/~2.4 GB budgets
  everything else in §3 is computed against assume this reclaim happened; report the real number.
---
## 14. Prior art — know it, do not be blindsided by it
Someone will ask. Rehearse the one-breath answer.
| Work | What it does | Why Tessera differs |
|---|---|---|
| GPTQ | Layer-wise PTQ with second-order info | Uniform target width; no per-task allocation |
| AWQ | Protects ~1% salient weights by activation magnitude | Fixed recipe, task-agnostic, single output width |
| SpQR / SqueezeLLM | Isolate outliers into a high-precision sparse tail | Orthogonal; could be composed with Tessera |
| HAWQ / HAWQ-V2 | Hessian-based per-layer bit allocation | Closest prior art. Offline, fixed budget, no nesting, no task conditioning |
| llama.cpp K-quants | Hand-tuned mixed precision (`Q4_K_M` bumps `down_proj`, `v_proj`) | Hand-tuned, one-size-fits-all, one file per width |
| Any-precision LLM / MatQuant | Nested / multi-width from one artifact | Closest on nesting. No device-conditioned allocation, no task conditioning, no portable session |
**The one-breath answer:** *"Those ship a fixed recipe at a fixed width. We compute allocation
automatically, per task domain, against the live memory budget of whatever device you plug into —
out of a single nested artifact, with the session travelling on hardware."*
Do a literature check before the event; this field moves fast and something may have landed
recently that either validates or scoops a specific angle. Better to find it now.

**Model family question ("why Qwen3 and not a MoE / hybrid model?", spec amendment §2):** a judge
who knows the current landscape may ask why this doesn't use a sparse or hybrid architecture
(Qwen3.5's small series, Gemma 4's effective-parameter models) given those are newer and often
benchmark better per active-parameter. The answer is the same "substrate, not contribution"
principle as the rest of this table: MoE routing frequency confounds per-tensor sensitivity
measurement (§2, §9), and effective-parameter/hybrid models break the "budget is what's resident"
premise the whole allocator is built on. Tessera's contribution is the allocator and the format,
not the model — so the model should be the most boring, best-characterized dense transformer
available, which is Qwen3 dense, not whatever benchmarks best this month.
---
## 15. Open decisions — ask, do not guess
1. Group size: 128 vs 64. Affects scale overhead and quality. Sweep it at M1.
2. Symmetric vs asymmetric quantization. Asymmetric is the default assumption here.
3. `token_embd`: free variable in the allocator, or pinned at 6-bit? Let M2 decide empirically.
4. Domain detection (stretch 1): a small classifier, keyword heuristics, or manual button only?
5. RP2040 vs ESP32-S3. ESP32-S3 gives more RAM and wireless headroom; RP2040 has cleaner TinyUSB
   composite support. Pick by whichever is physically in hand.
6. ~~Does the runtime stay in PyTorch, or port to ggml for the Pis? Assess at M4.~~ **Resolved by
   the Pi 5 hardware amendment (§3, §9): PyTorch on laptop, ggml/llama.cpp required on both Pi
   tiers.** PyTorch's ~400–600 MB fixed tax doesn't fit the Pi Zero's ~350 MB budget at all, and
   makes the Pi 5's M4 RSS criterion unmeasurable at its 2.4 GB PyTorch budget vs 2.9 GB ggml.
7. How many turns before the rolling summary regenerates? Tune against resume quality at M6.
