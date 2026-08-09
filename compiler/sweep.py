"""Fake-quant sensitivity sweep (spec §4.1, amended).

For each of the 197 quantizable tensors in Qwen3-1.7B, and each candidate bit-width, fake-quantize
*only* that tensor (quantize -> dequantize back to fp16 in-place, everything else untouched),
evaluate **perplexity on domain-specific held-out text** (NOT HumanEval/GSM8K — see spec §4.1
amendment), and record delta vs the fp16 baseline. 197 tensors x 6 widths x 4 domains = 4,728
evaluations; embarrassingly parallel across tensors, run unattended overnight on a GPU.

Two-tier metric scheme (spec §4.1 amendment):
- sweep metric (all 4,728 evals): perplexity, forward-pass only, deterministic
- confirmation metric (~20 evals, once, post-M2): HumanEval/GSM8K on the allocator's actual
  manifest output — see eval/harness.py, NOT this file

Do not run this without first running, in order:
1. run_noise_floor_check() — confirms held-out sets can resolve real signal
2. run_proxy_validation() — confirms delta-ppl actually predicts delta-pass@1
Both are pre-flight gates (spec §4.1), not optional steps to skip when in a hurry.

Requires transformers/torch and network access to pull Qwen3-1.7B + calibration/held-out sets.
Run with:

    python -m compiler.sweep preflight --model Qwen/Qwen3-1.7B
    python -m compiler.sweep run --model Qwen/Qwen3-1.7B \
        --domains chat code math summ --out compiler/sensitivity.parquet

`enable_thinking=False` is non-negotiable (spec §4.1 amendment) for the proxy validation's
HumanEval calls (the sweep's own ppl measurement is a raw forward pass, not a chat completion,
so `enable_thinking` doesn't apply to it directly — see spec §4.1 note).

The fake-quant math here MUST match format/bitplane.py's quantize_asymmetric exactly (same
group size, same asymmetric formula) — the sweep is measuring damage from the same quantizer
the allocator's manifest will actually be read through, not an idealized one.

Checkpointing note (spec §4.1 pre-flight check 2, "write to parquet after each evaluation"):
implemented as an append-only JSONL checkpoint, not literal per-row parquet rewrites. A parquet
file has no footer until the writer closes cleanly, so a hard crash mid-write of a *full-file*
rewrite can corrupt or truncate the *entire* accumulated result, which is the opposite of
resumable. JSONL is append-only and each line is independently valid, so a crash loses at most
the (already-detectable, half-written) last line. `checkpoint_to_parquet` consolidates to the
spec's `sensitivity.parquet` schema at the end or on demand.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from format.bitplane import dequantize_at_bits, planes_to_q_bits, q_to_planes, quantize_asymmetric

BIT_CHOICES = (2, 3, 4, 5, 6, 8)
DOMAINS = ("chat", "code", "math", "summ")
PROXY_VALIDATION_BITS = 3  # low enough that damage is visible, high enough not to be degenerate

# 7 per-layer tensors x 28 layers + token_embd = 197 knobs (spec §3).
PER_LAYER_TENSORS = ("attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down")
N_LAYERS = 28

# WikiText-2 baseline this whole sweep's deltas are relative to (spec §10 M0, verified 2026-08-08
# against Qwen3-1.7B, Salesforce/wikitext:wikitext-2-raw-v1:test, seq_len=2048). Recorded here so
# write_methodology can flag drift instead of silently comparing against a stale number.
M0_WIKITEXT2_BASELINE_PPL = 16.7835


def quantizable_tensor_names(n_layers: int = N_LAYERS) -> List[str]:
    names = ["token_embd"]
    for layer in range(n_layers):
        for t in PER_LAYER_TENSORS:
            names.append(f"blk.{layer}.{t}")
    return names


def select_stratified_tensor_sample() -> List[str]:
    """15 tensors spanning layer depth and tensor kind (spec §4.1 proxy validation) — a
    deliberately spread, deterministic selection, not a random sample. Mixes mlp_down (expected
    more sensitive: larger, carries most of the per-layer weight — spec §3) against attn_k
    (expected less sensitive: small, shared across GQA groups) at 7 depths, plus token_embd."""
    layer_sample = [0, 4, 9, 13, 18, 22, 27]  # spans early/mid/late, includes first and last
    names = ["token_embd"]
    for layer in layer_sample:
        names.append(f"blk.{layer}.mlp_down")
        names.append(f"blk.{layer}.attn_k")
    return names  # 1 + 7*2 = 15


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
    bytes_at_bits: int
    bytes_saved_vs_fp16: int


def bytes_at_bits(num_weights: int, bits: int, group_size: int = 128) -> int:
    num_groups = -(-num_weights // group_size)
    plane_size = -(-num_weights // 8)  # ceil(n/8) bytes per plane, bitpacked
    return bits * plane_size + num_groups * 8  # planes + (scale,zero) f32 pairs


def get_param(state: Dict[str, "object"], tsra_name: str):
    """Look up a canonical .tsra tensor name in a real model's named_parameters() dict, via the
    same runtime.model mapping the loader/packer use — NOT ad-hoc substring matching, which
    silently fails since the two naming schemes ("blk.0.attn_q" vs
    "model.layers.0.self_attn.q_proj.weight") don't share substrings at all."""
    from runtime.model import tsra_name_to_hf_name

    hf_name = tsra_name_to_hf_name(tsra_name)
    if hf_name not in state:
        raise RuntimeError(f"tensor {tsra_name!r} (hf name {hf_name!r}) not found in model state dict")
    return state[hf_name]


def restore_and_assert(param, original) -> None:
    """Pre-flight check 1 (spec §4.1): restore `param` to `original` and assert bitwise
    equality, every call, not sampled. A copy from a detached clone should be exactly
    bit-identical by construction — if it isn't, something is aliasing memory or referencing the
    wrong tensor, and every measurement after that point in the sweep is silently corrupted."""
    import torch

    with torch.no_grad():
        param.copy_(original)
    if not torch.equal(param, original):
        raise AssertionError(
            "weight restoration failed — param does not bitwise-match its pristine copy after "
            "restore. Stopping immediately: every sweep row after this point would be measuring "
            "a corrupted baseline."
        )


# ---------------------------------------------------------------------------
# Checkpointing (spec §4.1 pre-flight check 2)
# ---------------------------------------------------------------------------


def _checkpoint_key(tensor_name: str, bits: int, domain: str) -> Tuple[str, int, str]:
    return (tensor_name, bits, domain)


def load_checkpoint_keys(checkpoint_path: str) -> Set[Tuple[str, int, str]]:
    """Read an existing JSONL checkpoint (if any) and return the set of already-completed
    (tensor, bits, domain) keys, so a resumed run skips them instead of redoing them."""
    path = Path(checkpoint_path)
    if not path.exists():
        return set()
    keys: Set[Tuple[str, int, str]] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # last line of a crashed run may be half-written; skip, don't fail
        keys.add(_checkpoint_key(row["tensor_name"], row["bits"], row["domain"]))
    return keys


def append_checkpoint(checkpoint_path: str, row: SweepRow) -> None:
    """Append one evaluation's result as a single JSON line, flushed immediately — this is the
    unit of resumability. See module docstring for why JSONL, not literal parquet rewrites."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row)) + "\n")
        f.flush()


def checkpoint_to_parquet(checkpoint_path: str, out_path: str) -> int:
    """Consolidate the JSONL checkpoint into the spec's sensitivity.parquet schema. Returns the
    row count written. Safe to call mid-run for a partial snapshot, or at the end for the final
    artifact — the checkpoint is the source of truth, parquet is a derived view of it."""
    import pandas as pd

    rows = []
    for line in Path(checkpoint_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return len(df)


def write_methodology(
    path: str,
    *,
    model_name: str,
    group_size: int,
    seq_len: int,
    n_held_out_samples: int,
) -> None:
    """Pre-flight check 5 (spec §4.1): lock and record the eval methodology this sweep's deltas
    are relative to. All 4,728 deltas are invalidated if any of these change without a re-run."""
    methodology = {
        "model": model_name,
        "group_size": group_size,
        "held_out_seq_len": seq_len,
        "n_held_out_samples": n_held_out_samples,
        "wikitext_dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "m0_wikitext2_baseline_ppl": M0_WIKITEXT2_BASELINE_PPL,
        "bit_choices": list(BIT_CHOICES),
        "domains": list(DOMAINS),
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(methodology, indent=2))


# ---------------------------------------------------------------------------
# Pre-flight gates (spec §4.1) — run these BEFORE run_sweep, not after
# ---------------------------------------------------------------------------


def run_noise_floor_check(
    model_name: str,
    domain: str = "chat",
    group_size: int = 128,
    repeat_tensor: str = "blk.13.mlp_down",
    insensitive_tensor: str = "blk.27.attn_k",
    bits: int = PROXY_VALIDATION_BITS,
    device: Optional[str] = None,
) -> dict:
    """Pre-flight check 3 (spec §4.1). Two complementary measurements:

    1. **Determinism check**: fake-quantize `repeat_tensor` once, measure ppl on the *identical*
       held-out set twice with no requantization between. Confirms the pipeline is deterministic
       (this repo's measured result: it is — 0.0 difference). This is necessary but NOT
       sufficient for the real question, since a deterministic pipeline trivially reproduces
       itself regardless of whether the held-out set is big enough to resolve real signal —
       repeating the same fixed data can never surface a "too small a sample" problem.
    2. **Split-half noise floor** (the actual gate): with `repeat_tensor` still quantized, split
       the held-out set into two disjoint halves and measure delta-ppl on each half separately.
       The spread between the two halves' deltas *is* the real noise floor — it answers "if I'd
       drawn a different, equally-sized held-out sample, how much would my delta estimate move."
       `insensitive_tensor`'s delta (measured on the full set) is then compared against this
       split-half spread, not against the (always ~0) repeat spread.

    If |insensitive delta| sits inside the split-half noise floor, the held-out set is too small
    to resolve real signal from this tensor — the caller should stop, not proceed to the full
    sweep. Does its own weight restoration + bitwise assertion (spec §4.1 check 1)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compiler.calib import load_domain_held_out_ppl_set, perplexity

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    model.eval()

    held_out = [ids.to(device) for ids in load_domain_held_out_ppl_set(domain, tok)]
    mid = len(held_out) // 2
    half_a, half_b = held_out[:mid], held_out[mid:]

    state = dict(model.named_parameters())
    baseline_full = perplexity(model, held_out)
    baseline_a = perplexity(model, half_a)
    baseline_b = perplexity(model, half_b)

    # quantize repeat_tensor once; take all measurements against it before restoring
    param = get_param(state, repeat_tensor)
    original = param.detach().clone()
    w_hat = fake_quantize_tensor(original.float().cpu().numpy(), bits, group_size)
    with torch.no_grad():
        param.copy_(torch.from_numpy(w_hat).to(original.dtype))

    ppl_full_1 = perplexity(model, held_out)
    ppl_full_2 = perplexity(model, held_out)  # identical-data repeat -> determinism check
    ppl_a = perplexity(model, half_a)
    ppl_b = perplexity(model, half_b)
    restore_and_assert(param, original)

    determinism_delta = abs(ppl_full_1 - ppl_full_2)
    delta_a = ppl_a - baseline_a
    delta_b = ppl_b - baseline_b
    split_half_noise_floor = abs(delta_a - delta_b)

    # insensitive tensor: single full-set measurement, compared against the split-half floor
    param2 = get_param(state, insensitive_tensor)
    original2 = param2.detach().clone()
    w_hat2 = fake_quantize_tensor(original2.float().cpu().numpy(), bits, group_size)
    with torch.no_grad():
        param2.copy_(torch.from_numpy(w_hat2).to(original2.dtype))
    ppl_insensitive = perplexity(model, held_out)
    restore_and_assert(param2, original2)
    insensitive_delta = ppl_insensitive - baseline_full

    verdict = (
        "STOP — signal below noise floor" if abs(insensitive_delta) <= split_half_noise_floor else "PASS"
    )
    return {
        "domain": domain,
        "bits": bits,
        "n_held_out": len(held_out),
        "baseline_ppl_full": baseline_full,
        "determinism_delta": determinism_delta,
        "repeat_tensor": repeat_tensor,
        "repeat_delta_half_a": delta_a,
        "repeat_delta_half_b": delta_b,
        "split_half_noise_floor": split_half_noise_floor,
        "insensitive_tensor": insensitive_tensor,
        "insensitive_delta": insensitive_delta,
        "verdict": verdict,
    }


def run_proxy_validation(
    model_name: str,
    group_size: int = 128,
    n_problems: int = 50,
    domain: str = "general",
    num_fewshot: int = 2,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Proxy validation (spec §4.1) — run before the full sweep. 15 stratified tensors
    (select_stratified_tensor_sample), each fake-quantized at PROXY_VALIDATION_BITS, measuring
    both delta-ppl and delta-task-metric — ~30 evaluations total. Returns the Spearman
    correlation between the two rankings.

    **Domain is `general`/HellaSwag accuracy by default — two levels of deviation from the
    spec's original `code`/HumanEval pass@1 framing, both forced by real constraints hit while
    actually running this, not chosen for convenience:**
    1. `code`/HumanEval is out: HuggingFace `evaluate`'s `code_eval` metric hard-refuses to run
       on Windows (`NotImplementedError`, confirmed directly). Tried `math`/GSM8K instead.
    2. `math`/GSM8K (generative, exact-match) is *also* out for this specific step: 0-shot
       floored at exactly 0.0000 on every one of 6 completed tensors before a run crashed
       (answer-formatting failure, not a damage measurement — GSM8K's exact-match scorer needs
       few-shot examples to establish the "#### <number>" format). 2-shot still floored at
       0.0000 baseline. And even fixed, GSM8K is generative (~30-50s/problem) — 15 tensors x 2
       (ppl+metric) would take hours for a *pre-flight* check.
    `general`/HellaSwag (log-likelihood scored — a single forward pass per candidate, no
    autoregressive generation) fixes both: no code execution, and measured ~11s of actual
    inference for 20 problems (dataset load/mapping is a one-time per-process cost, not
    per-tensor). Not domain-matched to code/math specifically, but proxy validation's actual
    question — "does ppl predict *some* real downstream task quality" — doesn't require it to be.
    `domain="code"` (HumanEval) and `domain="math"` (GSM8K, `num_fewshot` applies) remain
    available for a Linux/CI environment where the code_eval block doesn't apply and the longer
    runtime is acceptable.

    Incremental checkpointing (JSONL, same pattern as run_sweep) — an earlier GSM8K-based attempt
    crashed with a CUDA illegal-memory-access error partway through (cause not fully diagnosed;
    possibly resource buildup from repeated HFLM construction inside evaluate_task_metric). Kept
    here even though HellaSwag is much faster, since the cause of that crash isn't confirmed
    fixed — a checkpoint means a recurrence costs the in-flight tensor's work, not the whole run.

    Sign convention: damage should INCREASE ppl (positive delta_ppl) and DECREASE the task
    metric (negative delta_metric) for the same sensitive tensors, so a real relationship shows
    up as a NEGATIVE Spearman correlation. |rho| close to 1 (negative) = proxy justified. Near 0
    = stop, the sweep would be measuring the wrong thing."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compiler.calib import assert_thinking_disabled, load_domain_held_out_ppl_set, perplexity, spearman_correlation

    ppl_domain = domain if domain in ("chat", "code", "math", "summ") else "chat"

    if domain == "code":
        from eval.harness import evaluate_humaneval_pass1 as evaluate_task_metric
    elif domain == "math":
        from eval.harness import evaluate_gsm8k_exact_match as _eval_gsm8k

        def evaluate_task_metric(model, tok, n_problems):
            return _eval_gsm8k(model, tok, n_problems=n_problems, num_fewshot=num_fewshot)
    elif domain == "general":
        from eval.harness import evaluate_hellaswag_acc as evaluate_task_metric
    else:
        raise ValueError(f"proxy validation needs a generative/loglikelihood task metric; no support for domain {domain!r}")

    def _cleanup():
        # defensive against the resource buildup that may have contributed to the earlier crash
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or "compiler/proxy_validation.checkpoint.jsonl"

    checkpoint_rows: Dict[str, dict] = {}
    if Path(checkpoint_path).exists():
        for line in Path(checkpoint_path).read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                checkpoint_rows[row["tensor_name"]] = row
        if checkpoint_rows:
            print(f"resuming: {len(checkpoint_rows)} tensors already checkpointed", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    assert_thinking_disabled(tok)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    model.eval()

    held_out = [ids.to(device) for ids in load_domain_held_out_ppl_set(ppl_domain, tok)]
    state = dict(model.named_parameters())

    baseline_ppl = perplexity(model, held_out)
    baseline_metric = evaluate_task_metric(model, tok, n_problems=n_problems)
    _cleanup()
    print(f"baseline: ppl={baseline_ppl:.4f} metric={baseline_metric:.4f}", flush=True)

    tensor_names = select_stratified_tensor_sample()
    for tensor_name in tensor_names:
        if tensor_name in checkpoint_rows:
            continue

        param = get_param(state, tensor_name)
        original = param.detach().clone()
        w_hat = fake_quantize_tensor(original.float().cpu().numpy(), PROXY_VALIDATION_BITS, group_size)
        with torch.no_grad():
            param.copy_(torch.from_numpy(w_hat).to(original.dtype))

        ppl = perplexity(model, held_out)
        metric = evaluate_task_metric(model, tok, n_problems=n_problems)
        _cleanup()

        restore_and_assert(param, original)

        d_ppl = ppl - baseline_ppl
        d_metric = metric - baseline_metric
        row = {"tensor_name": tensor_name, "delta_ppl": d_ppl, "delta_metric": d_metric}
        checkpoint_rows[tensor_name] = row
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        print(f"  {tensor_name}: delta_ppl={d_ppl:+.4f} delta_metric={d_metric:+.4f}", flush=True)

    ppl_deltas = [checkpoint_rows[t]["delta_ppl"] for t in tensor_names]
    metric_deltas = [checkpoint_rows[t]["delta_metric"] for t in tensor_names]
    rho = spearman_correlation(ppl_deltas, metric_deltas)
    return {
        "domain": domain,
        "ppl_domain": ppl_domain,
        "tensor_names": tensor_names,
        "delta_ppl": ppl_deltas,
        "delta_metric": metric_deltas,
        "spearman": rho,
        "n_problems": n_problems,
        "num_fewshot": num_fewshot,
        "bits": PROXY_VALIDATION_BITS,
        "baseline_ppl": baseline_ppl,
        "baseline_metric": baseline_metric,
    }


# ---------------------------------------------------------------------------
# The full sweep — do not call without pre-flight checks passing first
# ---------------------------------------------------------------------------


def run_sweep(
    model_name: str,
    domains: Iterable[str],
    out_path: str,
    group_size: int = 128,
    checkpoint_path: Optional[str] = None,
    seq_len: int = 2048,
    n_held_out_samples: int = 128,
    device: Optional[str] = None,
) -> None:
    """Full sweep against a real model + held-out sets. Stratified width-major/domain-major
    order (spec §4.1 pre-flight check 4): outer loops are (domain, bits), inner loop is over
    tensors — so an interrupted run has complete slices, not fragments. Resumable from
    `checkpoint_path` (spec §4.1 pre-flight check 2)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compiler.calib import load_domain_held_out_ppl_set, perplexity

    checkpoint_path = checkpoint_path or str(Path(out_path).with_suffix(".checkpoint.jsonl"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    write_methodology(
        str(Path(out_path).with_suffix(".methodology.json")),
        model_name=model_name,
        group_size=group_size,
        seq_len=seq_len,
        n_held_out_samples=n_held_out_samples,
    )

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    model.eval()

    state = dict(model.named_parameters())
    tensor_names = quantizable_tensor_names()

    # validate every tensor resolves before starting the (potentially many-hour) run
    from runtime.model import tsra_name_to_hf_name

    missing = [n for n in tensor_names if tsra_name_to_hf_name(n) not in state]
    if missing:
        raise RuntimeError(
            f"{len(missing)} expected tensors not found in {model_name} state dict "
            f"(architecture mismatch — verify against config.json per spec §3): {missing[:5]}..."
        )

    completed = load_checkpoint_keys(checkpoint_path)
    if completed:
        print(f"resuming: {len(completed)} evaluations already checkpointed", flush=True)

    total = len(tensor_names) * len(BIT_CHOICES) * len(list(domains))
    done = 0
    for domain in domains:
        held_out = [ids.to(device) for ids in load_domain_held_out_ppl_set(domain, tok, n_samples=n_held_out_samples, seq_len=seq_len)]
        baseline_ppl = perplexity(model, held_out)

        for bits in BIT_CHOICES:
            for tensor_name in tensor_names:
                done += 1
                key = _checkpoint_key(tensor_name, bits, domain)
                if key in completed:
                    continue

                param = get_param(state, tensor_name)
                original = param.detach().clone()
                w_np = original.float().cpu().numpy()
                w_hat = fake_quantize_tensor(w_np, bits, group_size)
                with torch.no_grad():
                    param.copy_(torch.from_numpy(w_hat).to(original.dtype))

                ppl = perplexity(model, held_out)

                restore_and_assert(param, original)  # pre-flight check 1, every iteration

                n = w_np.size
                row = SweepRow(
                    tensor_name=tensor_name,
                    domain=domain,
                    bits=bits,
                    delta_ppl=ppl - baseline_ppl,
                    bytes_at_bits=bytes_at_bits(n, bits, group_size),
                    bytes_saved_vs_fp16=n * 2 - bytes_at_bits(n, bits, group_size),
                )
                append_checkpoint(checkpoint_path, row)

                if done % 100 == 0:
                    print(f"{done}/{total} ({domain}, {bits}-bit, {tensor_name})", flush=True)

    n_rows = checkpoint_to_parquet(checkpoint_path, out_path)
    print(f"wrote {out_path}: {n_rows} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    nf = sub.add_parser("noise-floor", help="pre-flight check 3")
    nf.add_argument("--model", default="Qwen/Qwen3-1.7B")
    nf.add_argument("--domain", default="chat", choices=list(DOMAINS))

    pv = sub.add_parser("proxy-validation", help="pre-flight: does delta-ppl predict delta-task-metric?")
    pv.add_argument("--model", default="Qwen/Qwen3-1.7B")
    pv.add_argument("--n-problems", type=int, default=50)
    pv.add_argument("--domain", default="general", choices=["general", "code", "math"])

    run = sub.add_parser("run", help="the full 4,728-evaluation sweep")
    run.add_argument("--model", default="Qwen/Qwen3-1.7B")
    run.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    run.add_argument("--out", default="compiler/sensitivity.parquet")
    run.add_argument("--group-size", type=int, default=128)
    run.add_argument("--checkpoint", default=None)

    args = ap.parse_args()
    if args.cmd == "noise-floor":
        result = run_noise_floor_check(args.model, domain=args.domain)
        print(json.dumps(result, indent=2))
    elif args.cmd == "proxy-validation":
        result = run_proxy_validation(args.model, n_problems=args.n_problems, domain=args.domain)
        print(json.dumps({k: v for k, v in result.items()}, indent=2))
    elif args.cmd == "run":
        run_sweep(args.model, args.domains, args.out, args.group_size, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
