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
- Model family is **Qwen2.5**. Frozen on day one.
- The nested artifact is **pre-installed on each device**. The dongle carries only manifests +
  session state (kilobytes), never gigabytes of weights.
- Portable context is **transcript + summary + embeddings**, never a raw KV cache.
---
## 3. Target models and hardware
| Device | RAM | Model | Profiles |
|---|---|---|---|
| Laptop (dev machine) | 8–16 GB | Qwen2.5-1.5B-Instruct | quality, balanced |
| Raspberry Pi 5 | 4 GB | Qwen2.5-1.5B-Instruct | balanced, survival |
| Raspberry Pi Zero 2W | 512 MB | Qwen2.5-0.5B-Instruct | survival only |
The Zero cannot host 1.5B at any usable bit-width once the OS is accounted for. It runs 0.5B.
This is a feature, not a compromise: because portable context is a transcript rather than a KV
cache, the session survives a **model swap**, not just a precision swap. Say this out loud in the
pitch — it is the strongest defence of the transport design.
### Qwen2.5-1.5B-Instruct architecture (verify against `config.json` before relying on these)
```
layers              28
hidden_size         1536
attn heads          12       head_dim 128
kv heads            2        (GQA)
intermediate_size   8960
vocab               151936
tied embeddings     yes
```
Parameter accounting per layer:
| Tensor | Shape | Params |
|---|---|---|
| `attn_q` | 1536 × 1536 | 2.36 M |
| `attn_k` | 1536 × 256 | 0.39 M |
| `attn_v` | 1536 × 256 | 0.39 M |
| `attn_o` | 1536 × 1536 | 2.36 M |
| `mlp_gate` | 1536 × 8960 | 13.76 M |
| `mlp_up` | 1536 × 8960 | 13.76 M |
| `mlp_down` | 8960 × 1536 | 13.76 M |
| **total** | | **~46.8 M** |
× 28 layers = ~1.31 B, plus `token_embd` at 151936 × 1536 = **233 M** (~15% of the whole model in
one tensor). Total ≈ 1.54 B.
**Two consequences that shape the whole allocator:**
- MLP tensors are ~88% of per-layer weight. That is where the bytes are, and therefore where the
  allocator spends most of its decisions.
- `token_embd` is a single tensor worth more than five transformer blocks. It must be in the search
  space, not pinned by convention.
**Quantizable tensor count:** 7 per layer × 28 + `token_embd` = **197 knobs**.
LayerNorm weights and biases stay fp16 — negligible bytes, high sensitivity.
### KV cache
`2 (K,V) × 28 layers × 2 kv_heads × 128 head_dim × 2 bytes = 28,672 B/token ≈ 28 KB/token`
At 8k context that is ~230 MB; at 32k, ~900 MB — comparable to the quantized weights themselves.
Weight quantization alone does not solve on-device memory. KV-cache quantization is tracked as a
stretch goal (§9).
---
## 4. Component 1 — the precision compiler
Offline. Runs on the dev machine / any GPU. Produces manifests.
### 4.1 Sensitivity sweep
For each tensor `t` in the 197, and each candidate bit-width `b ∈ {2, 3, 4, 5, 6, 8}`:
1. Fake-quantize **only** `t` to `b` bits (quantize → dequantize back to fp16 in-place). Everything
   else stays fp16.
2. Evaluate on the domain calibration set.
3. Record `Δquality` and `bytes_saved`.
Fake quantization is ~30 lines of PyTorch. Do not fight real kernels at this stage — the sweep is
about *ranking*, and a quant-dequant round trip has the same numerics as the real thing.
**Grouping:** per-channel groups of 128 along the input dimension, asymmetric (scale + zero point).
Group size is a config knob; 128 is the default.
**Domains** (four calibration sets, ~128 samples each, ~2048 tokens per sample):
| Domain | Calibration source | Eval metric |
|---|---|---|
| `chat` | LMSYS-style multi-turn or OpenAssistant | WikiText-2 perplexity |
| `code` | The Stack (Python subset) or CodeAlpaca | HumanEval subset, pass@1 |
| `math` | GSM8K train split | GSM8K test subset, exact match |
| `summ` | CNN/DailyMail | ROUGE-L or perplexity on held-out |
**Cost:** 197 tensors × 6 widths × 4 domains = 4,728 evaluations. Each is a forward pass over the
calibration set on a 1.5B model — seconds on a GPU. Parallelize across tensors; this can run
unattended overnight. **Start this early.** It is the long pole and it is embarrassingly parallel.
**Output:** `sensitivity.parquet`
```
tensor_name, domain, bits, delta_ppl, delta_task_metric, bytes_at_bits, bytes_saved_vs_fp16
```
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
  "model": "qwen2.5-1.5b-instruct",
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
1. **Size.** ~28 KB/token; ~900 MB at 32k context. Over USB CDC this is minutes.
2. **Numerical validity.** A KV cache is an activation produced by *specific* weights. A cache from
   a 4-bit forward pass fed into a 6-bit forward pass is unprincipled and degrades silently.
3. **Model portability.** It cannot survive the 1.5B → 0.5B switch at all.
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
| SSD1306 128×64 OLED (I²C) | Live readout: `CODE · 5-bit · 1.1 GB` |
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
4. **KV-cache quantization** (8-bit K, 4-bit V is the usual sweet spot) — the largest remaining
   memory win, and nobody else at the event will have done it.
5. **ggml / llama.cpp backend** for real speed on the Pis.
---
## 10. Milestones and acceptance criteria
Ship in this order. Each milestone has a binary acceptance test.
| # | Milestone | Acceptance criteria |
|---|---|---|
| **M0** | Repo + eval harness | Qwen2.5-1.5B loads; baseline WikiText-2 ppl and HumanEval subset reproduce to ±0.05 / ±1 problem across two runs |
| **M1** | Fake-quant sweep | `sensitivity.parquet` populated for 197 tensors × 6 widths × 4 domains; no NaNs; per-domain Spearman correlations computed |
| **M2** | Allocator | Given (domain, budget) emits a valid manifest; at equal bytes, beats uniform quantization on ≥2 of 3 metrics for at least 3 of 4 domains |
| **M3** | Bit-plane format | `.tsra` round-trips; loading at `k` planes matches reference `k`-bit fake-quant within 1e-3 MSE; load time for a 1.1 GB profile under 8 s cold |
| **M4** | Runtime | Generates coherent text at quality / balanced / survival; measured RSS within 5% of `measured_bytes`; tokens/sec recorded per device per profile |
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
- **B — format + runtime:** M3, M4, stretch goals 2 and 3
- **C — hardware:** M5, M6, firmware, protocol, daemon
- **D — dashboard + demo:** M7, figures, pitch deck, running the rig
D is not a spare part. Half of judging is whether the room can *see* what is happening.
---
## 12. Demo choreography
Three devices on the table, one dongle, dashboard projected. Rehearse this until it is muscle
memory.
1. **Laptop, dongle in.** OLED: `CODE · 6-bit · 2.1 GB`. Debug a real code snippet — pull one from
   an actual project, not a toy. Heatmap shows MLP tensors running rich.
2. **Unplug.** Laptop visibly wipes the session on screen. Dashboard shows the zeroize.
3. **Into the Pi 5.** OLED: `CODE · 4-bit · 1.1 GB`. Ask a follow-up that is *only* answerable with
   the earlier context. It answers correctly. Heatmap redraws leaner.
4. **Into the Pi Zero 2W.** OLED: `CHAT · 3-bit · 380 MB`. Different model entirely — 0.5B. Slower,
   simpler, still coherent, still remembers. Call out that the context survived a model swap.
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
---
## 15. Open decisions — ask, do not guess
1. Group size: 128 vs 64. Affects scale overhead and quality. Sweep it at M1.
2. Symmetric vs asymmetric quantization. Asymmetric is the default assumption here.
3. `token_embd`: free variable in the allocator, or pinned at 6-bit? Let M2 decide empirically.
4. Domain detection (stretch 1): a small classifier, keyword heuristics, or manual button only?
5. RP2040 vs ESP32-S3. ESP32-S3 gives more RAM and wireless headroom; RP2040 has cleaner TinyUSB
   composite support. Pick by whichever is physically in hand.
6. Does the runtime stay in PyTorch, or port to ggml for the Pis? PyTorch on a Pi Zero will be
   painful. Assess at M4.
7. How many turns before the rolling summary regenerates? Tune against resume quality at M6.
