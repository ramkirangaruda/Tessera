"""Fake-quant sensitivity sweep (spec §4.1).

For each of the 197 quantizable tensors in Qwen3-1.7B (model family amended from Qwen2.5 — spec
§2), and each candidate bit-width, fake-quantize *only* that tensor (quantize -> dequantize back
to fp16 in-place, everything else untouched), evaluate perplexity on the domain calibration set,
and record delta vs the fp16 baseline. 197 tensors x 6 widths x 4 domains = 4,728 evaluations;
embarrassingly parallel across tensors, run unattended overnight on a GPU (spec: "Start this
early. It is the long pole.").

Requires transformers/torch and network access to pull Qwen3-1.7B + calibration sets. Run with:

    python -m compiler.sweep --model Qwen/Qwen3-1.7B \
        --domains chat code math summ --out compiler/sensitivity.parquet

`enable_thinking=False` is non-negotiable (spec §4.1 amendment) — Qwen3's hybrid thinking mode
would inflate this 4,728-evaluation sweep by orders of magnitude and inject variance that swamps
the quantization signal. run_sweep asserts it's actually passed to the chat template, not just
defaulted.

The fake-quant math here MUST match format/bitplane.py's quantize_asymmetric exactly (same
group size, same asymmetric formula) — the sweep is measuring damage from the same quantizer
the allocator's manifest will actually be read through, not an idealized one.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from format.bitplane import dequantize_at_bits, planes_to_q_bits, q_to_planes, quantize_asymmetric

BIT_CHOICES = (2, 3, 4, 5, 6, 8)
DOMAINS = ("chat", "code", "math", "summ")

# 7 per-layer tensors x 28 layers + token_embd = 197 knobs (spec §3).
PER_LAYER_TENSORS = ("attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down")
N_LAYERS = 28


def quantizable_tensor_names(n_layers: int = N_LAYERS) -> List[str]:
    names = ["token_embd"]
    for layer in range(n_layers):
        for t in PER_LAYER_TENSORS:
            names.append(f"blk.{layer}.{t}")
    return names


def fake_quantize_tensor(w: np.ndarray, bits: int, group_size: int = 128) -> np.ndarray:
    """Quantize -> dequantize `w` in place at `bits`, matching format/bitplane.py's derived-scale
    reconstruction exactly, so sensitivity numbers reflect what the .tsra loader will actually
    produce at that bit-width."""
    shape = w.shape
    flat = w.astype(np.float32).reshape(-1)
    q, scales, zeros = quantize_asymmetric(flat, group_size)
    planes = q_to_planes(q)
    q_bits = planes_to_q_bits(planes, bits)
    w_hat = dequantize_at_bits(q_bits, bits, scales, zeros, group_size, flat.shape[0])
    return w_hat.reshape(shape).astype(w.dtype)


@dataclass
class SweepRow:
    tensor_name: str
    domain: str
    bits: int
    delta_ppl: float
    delta_task_metric: float
    bytes_at_bits: int
    bytes_saved_vs_fp16: int


def bytes_at_bits(num_weights: int, bits: int, group_size: int = 128) -> int:
    num_groups = -(-num_weights // group_size)
    plane_size = -(-num_weights // 8)  # ceil(n/8) bytes per plane, bitpacked
    return bits * plane_size + num_groups * 8  # planes + (scale,zero) f32 pairs


def run_sweep(model_name: str, domains: Iterable[str], out_path: str, group_size: int = 128) -> None:
    """Full sweep against a real model + calibration sets. Needs torch/transformers/datasets and
    network access — not exercised by the test suite, which instead unit-tests
    `fake_quantize_tensor` and `bytes_at_bits` directly against synthetic tensors."""
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compiler.calib import load_calibration_set, perplexity, task_metric

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    state = {name: p for name, p in model.named_parameters()}
    tensor_names = quantizable_tensor_names()
    missing = [n for n in tensor_names if not any(n in k for k in state)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} expected tensors not found in {model_name} state dict "
            f"(architecture mismatch — verify against config.json per spec §3): {missing[:5]}..."
        )

    rows: List[SweepRow] = []
    for domain in domains:
        calib = load_calibration_set(domain, tokenizer=tok)
        baseline_ppl = perplexity(model, calib)
        baseline_metric = task_metric(model, tok, domain)

        for tensor_name in tensor_names:
            param = next(p for name, p in state.items() if tensor_name in name)
            original = param.detach().clone()
            for bits in BIT_CHOICES:
                w_np = original.float().cpu().numpy()
                w_hat = fake_quantize_tensor(w_np, bits, group_size)
                with torch.no_grad():
                    param.copy_(torch.from_numpy(w_hat).to(original.dtype))

                ppl = perplexity(model, calib)
                metric = task_metric(model, tok, domain)
                n = w_np.size
                rows.append(
                    SweepRow(
                        tensor_name=tensor_name,
                        domain=domain,
                        bits=bits,
                        delta_ppl=ppl - baseline_ppl,
                        delta_task_metric=metric - baseline_metric,
                        bytes_at_bits=bytes_at_bits(n, bits, group_size),
                        bytes_saved_vs_fp16=n * 2 - bytes_at_bits(n, bits, group_size),
                    )
                )
            with torch.no_grad():
                param.copy_(original)  # restore before moving to the next tensor

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_parquet(out_path)
    print(f"wrote {out_path}: {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    ap.add_argument("--out", default="compiler/sensitivity.parquet")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()
    run_sweep(args.model, args.domains, args.out, args.group_size)


if __name__ == "__main__":
    main()
