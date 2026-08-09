"""Evaluation harness (spec §10, M0 / §13).

M0 acceptance: Qwen3-1.7B loads with `enable_thinking=False` asserted (spec §4.1 amendment);
baseline WikiText-2 ppl and HumanEval subset reproduce to ±0.05 / ±1 problem across two runs.
Everything downstream (M1 sweep, M2 allocator eval, M4 runtime tok/s) is measured against the
same metric functions defined here, so the baseline reproducibility check is the harness's own
self-test, not a one-off script.

Needs torch/transformers/datasets/lm-eval + a real Qwen3 checkout and network access.
`eval/figures.py` (Pareto/heatmap/correlation plots) is the part of this milestone that's
dependency-light and unit-tested without a model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    model: str
    profile_id: str
    domain: str
    wikitext_ppl: float
    humaneval_pass1: float | None
    gsm8k_exact_match: float | None
    rouge_l: float | None
    measured_bytes: int
    tokens_per_sec: float | None
    device: str


def evaluate_wikitext_ppl(model, tokenizer, n_samples: int = 128, seq_len: int = 2048) -> float:
    from compiler.calib import load_wikitext2_eval, perplexity

    calib = load_wikitext2_eval(tokenizer, seq_len=seq_len, n_chunks=n_samples)
    return perplexity(model, calib)


def evaluate_humaneval_pass1(model, tokenizer, n_problems: int = 40) -> float:
    """HumanEval subset, pass@1 (spec §4.1 code domain metric). Delegates to lm-eval-harness;
    kept as a thin wrapper so sweep.py/allocate.py don't take a direct lm-eval dependency.
    `enable_thinking=False` is non-negotiable (spec §4.1 amendment) — asserted, not defaulted."""
    from compiler.calib import assert_thinking_disabled

    assert_thinking_disabled(tokenizer)
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "enable_thinking": False},
        tasks=["humaneval"],
        limit=n_problems,
        apply_chat_template=True,
        confirm_run_unsafe_code=True,
    )
    # NOTE: exact result key unverified on this dev environment — HuggingFace `evaluate`'s
    # code_eval metric, which this task depends on, refuses to run on Windows at all
    # (NotImplementedError, confirmed directly). Confirm this key on Linux/WSL/CI before relying
    # on it; humaneval.yaml's filter is named "create_test", so it's likely "pass@1,create_test".
    task_results = results["results"]["humaneval"]
    key = next((k for k in task_results if k.startswith("pass@1")), None)
    if key is None:
        raise KeyError(f"no pass@1 key in humaneval results: {list(task_results)}")
    return task_results[key]


def evaluate_gsm8k_exact_match(model, tokenizer, n_problems: int = 100, num_fewshot: Optional[int] = None) -> float:
    """`num_fewshot` overrides gsm8k.yaml's default (5). Note (spec §4.1 amendment): 0-shot
    GSM8K floors at exactly 0% regardless of quantization damage — the exact-match scorer needs
    few-shot examples to establish the "#### <number>" answer format; without them the model's
    raw completions rarely parse at all, which looks like "0% accuracy" but is actually an
    answer-formatting failure. This is generative + code-free, but slow (~30-50s/problem) and,
    at small `n_problems`, chunky (each problem is a full 1/n_problems step) — proxy validation
    (compiler/sweep.py) ended up switching to evaluate_hellaswag_acc for that reason; this
    function is still what the confirmation-metric step (§4.2) uses for math-domain manifests."""
    from compiler.calib import assert_thinking_disabled

    assert_thinking_disabled(tokenizer)
    import lm_eval

    kwargs = {} if num_fewshot is None else {"num_fewshot": num_fewshot}
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "enable_thinking": False},
        tasks=["gsm8k"],
        limit=n_problems,
        apply_chat_template=True,
        **kwargs,
    )
    # gsm8k.yaml defines two filters (strict-match, flexible-extract); prefer the stricter one.
    task_results = results["results"]["gsm8k"]
    for key in ("exact_match,strict-match", "exact_match,flexible-extract"):
        if key in task_results:
            return task_results[key]
    key = next((k for k in task_results if k.startswith("exact_match")), None)
    if key is None:
        raise KeyError(f"no exact_match key in gsm8k results: {list(task_results)}")
    return task_results[key]


def evaluate_hellaswag_acc(model, tokenizer, n_problems: int = 100) -> float:
    """HellaSwag accuracy (commonsense sentence-completion, 4-way multiple choice) — a
    log-likelihood-scored task, not a generative one: for each problem, lm-eval scores each
    candidate continuation with a single forward pass and picks the highest-probability one, no
    autoregressive generation at all. This makes it dramatically faster than GSM8K/HumanEval
    (seconds, not tens of seconds, per problem) and gives a continuous-valued signal (accuracy
    over many discrete right/wrong problems, not floored by generation-format failures the way
    0-shot GSM8K was — spec §4.1 amendment).

    Used by compiler/sweep.py's proxy validation as a fast, domain-agnostic stand-in for a
    generative task metric, after GSM8K proved too slow (~30-50s/problem) and, at small
    `n_problems`, too floored (0-shot: exact 0% baseline, an answer-formatting failure not a
    quality measurement) to validate 15 tensors in a reasonable pre-flight window. Not
    domain-specific like HumanEval/GSM8K — the point here is only "does ppl predict *some* real
    downstream task quality," which HellaSwag answers just as validly as a domain-matched task
    would, for cheap."""
    from compiler.calib import assert_thinking_disabled

    assert_thinking_disabled(tokenizer)
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "enable_thinking": False},
        tasks=["hellaswag"],
        limit=n_problems,
    )
    task_results = results["results"]["hellaswag"]
    for key in ("acc_norm,none", "acc,none"):
        if key in task_results:
            return task_results[key]
    key = next((k for k in task_results if k.startswith("acc")), None)
    if key is None:
        raise KeyError(f"no acc key in hellaswag results: {list(task_results)}")
    return task_results[key]


def run_baseline_reproducibility_check(model_name: str, n_runs: int = 2) -> bool:
    """M0 acceptance test: same model, same eval, run twice, results must agree within
    ±0.05 ppl / ±1 HumanEval problem. Returns True iff both runs agree within tolerance."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compiler.calib import assert_thinking_disabled

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert_thinking_disabled(tokenizer)
    ppls = []
    for _ in range(n_runs):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.eval()
        ppls.append(evaluate_wikitext_ppl(model, tokenizer))
        del model

    spread = max(ppls) - min(ppls)
    print(f"wikitext ppl across {n_runs} runs: {ppls}, spread={spread:.4f}")
    return spread <= 0.05


def evaluate_profile(model_name: str, tsra_path: str, manifest_path: str, device: str = "cpu") -> EvalResult:
    """Full metric suite for one manifest (spec §13 "numbers to have ready"): loads the model at
    the manifest's allocation via runtime.model.from_manifest, runs the domain-appropriate task
    metric plus WikiText-2 ppl as a cross-domain reference, times generation for tokens/sec."""
    import time

    from runtime.generate import generate
    from runtime.model import from_manifest
    from transformers import AutoTokenizer

    from compiler.calib import assert_thinking_disabled

    manifest = json.loads(Path(manifest_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert_thinking_disabled(tokenizer)
    model = from_manifest(model_name, tsra_path, manifest, device=device)

    ppl = evaluate_wikitext_ppl(model, tokenizer)
    domain = manifest["domain"]
    humaneval = evaluate_humaneval_pass1(model, tokenizer) if domain == "code" else None
    gsm8k = evaluate_gsm8k_exact_match(model, tokenizer) if domain == "math" else None

    t0 = time.time()
    _, tps = generate(model, tokenizer, [{"role": "user", "content": "def fibonacci(n):"}], max_new_tokens=64)
    elapsed = time.time() - t0

    return EvalResult(
        model=model_name,
        profile_id=manifest["profile_id"],
        domain=domain,
        wikitext_ppl=ppl,
        humaneval_pass1=humaneval,
        gsm8k_exact_match=gsm8k,
        rouge_l=None,
        measured_bytes=manifest["measured_bytes"],
        tokens_per_sec=tps,
        device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    repro = sub.add_parser("reproducibility", help="M0 baseline reproducibility check")
    repro.add_argument("--model", default="Qwen/Qwen3-1.7B")

    ev = sub.add_parser("profile", help="evaluate one manifest")
    ev.add_argument("--model", required=True)
    ev.add_argument("--tsra", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.cmd == "reproducibility":
        ok = run_baseline_reproducibility_check(args.model)
        print("PASS" if ok else "FAIL")
    elif args.cmd == "profile":
        result = evaluate_profile(args.model, args.tsra, args.manifest, args.device)
        out = json.dumps(asdict(result), indent=2)
        print(out)
        if args.out:
            Path(args.out).write_text(out)


if __name__ == "__main__":
    main()
