# Tessera

> Portable, precision-adaptive local LLM. Your assistant lives on a hardware key, not a machine.
> Plug it into any device and the model reshapes its own precision to fit that hardware and the
> task at hand. Unplug and it leaves nothing behind.

Status: greenfield build-out, in progress. See [`docs/spec.md`](docs/spec.md) for the full
engineering spec this repo implements.

## What this is

Standard quantization applies one bit-width to a whole model, uniformly, for every user. Tessera
computes bit allocation **per-tensor**, conditioned on **task domain** (chat / code / math / summ)
and the **actual free memory** of whatever device it lands on — out of a single nested bit-plane
artifact that every device reads at whatever precision it can afford. A session (transcript +
summary + embeddings, never a raw KV cache) travels on a USB dongle and resumes on a different
device, at a different precision, even on a different model size.

Model family: Qwen3 dense (1.7B laptop/Pi5, 0.6B Pi Zero 2W), amended from an initial Qwen2.5
choice — see `docs/spec.md` §2/§3 for the full rationale and the config-derived numbers behind
it. Qwen2.5-1.5B/-0.5B run as a secondary validation tier (§13 cross-model replication).

## Repo layout

```
compiler/   sensitivity sweep + Lagrangian bit allocator (offline, produces manifests)
format/     nested bit-plane weight format (.tsra) — pack / mmap load / round-trip test
runtime/    manifest-driven Qwen3 loader + generation
daemon/     host daemon — USB link, handshake protocol, crypto, FastAPI/websocket server
firmware/   RP2040/ESP32-S3 dongle firmware (TinyUSB composite, OLED, LED, write-protect)
dashboard/  React + Vite live dashboard (memory bar, allocation heatmap, Pareto plot)
eval/       evaluation harness + figure generation (Pareto, heatmap, correlation matrix)
docs/       spec, demo script, pitch
```

## Milestones

See `docs/spec.md` §10. Critical path is **M0 → M1 → M2 → M3 → M4**; firmware/daemon (M5/M6)
develop in parallel against a mocked bundle.

| # | Milestone | Status |
|---|---|---|
| M0 | Repo + eval harness | in progress |
| M1 | Fake-quant sensitivity sweep | scaffolded, needs GPU run |
| M2 | Allocator | implemented |
| M3 | Bit-plane format | implemented, round-trip tested |
| M4 | Runtime | scaffolded |
| M5 | Firmware + protocol | scaffolded |
| M6 | Daemon + wipe | scaffolded |
| M7 | Dashboard + demo rig | scaffolded |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Format round-trip test (no model / GPU required)
python -m pytest format/test_roundtrip.py -v

# Allocator smoke test on synthetic sensitivity data (no model / GPU required)
python -m pytest compiler/test_allocate.py -v
```

Everything else (`compiler/sweep.py`, `runtime/`, `daemon/`, `firmware/`) needs the actual
Qwen3-1.7B weights and, for the sweep, a GPU — see inline docstrings for how to point them at a
local checkout of the model.

## Team

- **A — compiler:** sensitivity sweep, allocator, correlation analysis
- **B — format + runtime:** `.tsra` format, manifest-driven runtime, bit-plane drop
- **C — hardware:** firmware, handshake protocol, daemon
- **D — dashboard + demo:** dashboard, figures, pitch, demo rig
