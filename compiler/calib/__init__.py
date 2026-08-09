"""Calibration set loaders + eval metrics for the sensitivity sweep (spec §4.1, amended).

Two-tier metric scheme (spec §4.1 amendment — the sweep does NOT run HumanEval/GSM8K per
tensor; that's 4,728 generative evaluations with sampling variance on top of an already-small
signal):

- **Sweep metric** (all 4,728 evaluations): perplexity on domain-specific **held-out** text —
  `load_domain_held_out_ppl_set` below. Deliberately a *different* split/slice than the
  calibration source, so the sweep never scores a tensor against the same data it was
  fake-quantized against.
- **Confirmation metric** (~20 evaluations total, run once on M2's manifest output, not per
  sweep iteration): HumanEval pass@1 / GSM8K exact match — see `eval/harness.py`.

| Domain | Calibration source (fake-quant against)     | Held-out ppl source (sweep metric)      |
|--------|----------------------------------------------|------------------------------------------|
| chat   | OpenAssistant                                 | WikiText-2 (same set as the M0 baseline)  |
| code   | The Stack Python, `train` samples [0:N]       | The Stack Python, `train` samples [N:2N]  |
| math   | GSM8K `train` split                           | GSM8K `test` split                        |
| summ   | CNN/DailyMail `train` split                   | CNN/DailyMail `validation` split          |

~128 samples/domain, ~2048 tokens/sample. Needs `datasets` + network access; not exercised by the
test suite. Kept as a thin, swappable layer so a team member can point it at local calibration
files instead of the Hub without touching sweep.py.
"""
from __future__ import annotations

from typing import List

N_CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 2048

# Qwen3 non-thinking-mode sampling guidance (model card) — distinct from the thinking-mode
# defaults in generation_config.json (temperature 0.6/top_p 0.95), which assume a reasoning
# trace precedes the answer. Used wherever this repo generates from a Qwen3 chat template.
QWEN3_NO_THINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def assert_thinking_disabled(tokenizer) -> None:
    """Spec §4.1 amendment: enable_thinking=False is non-negotiable for every Qwen3 sweep/eval
    run — a reasoning trace on each of the sweep's 4,728 evaluations would inflate it by orders
    of magnitude and inject variance that swamps the quantization signal being measured.

    No-ops for tokenizers with no thinking-mode switch at all (the §13 secondary validation run
    uses Qwen2.5, which has no such concept — nothing to disable, nothing to assert). For a
    tokenizer that *does* advertise `enable_thinking`, actually renders a chat template call with
    it set to False and fails loudly if that call errors, rather than trusting the kwarg was
    accepted without checking — call this once at harness startup rather than assuming a kwarg
    default holds all the way through a multi-hour sweep.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" not in template:
        return  # model has no thinking mode to disable (e.g. Qwen2.5 secondary validation)
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as e:
        raise AssertionError(
            "tokenizer's chat template advertises enable_thinking but rendering with "
            f"enable_thinking=False failed: {e}. Do not proceed with a sweep/eval run without "
            "verifying reasoning traces are actually suppressed."
        ) from e


def render_chat_prompt(tokenizer, messages: list, add_generation_prompt: bool = True) -> str:
    """Applies the tokenizer's chat template, passing `enable_thinking=False` when the template
    supports it (Qwen3) and omitting the kwarg entirely when it doesn't (Qwen2.5 secondary
    validation run, spec §13 — which has no thinking-mode concept at all). Avoids a TypeError
    from an unsupported kwarg while still guaranteeing thinking is off everywhere it could be on.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )

_DATASET_BY_DOMAIN = {
    "chat": ("OpenAssistant/oasst1", None),
    "code": ("bigcode/the-stack-smol", "data/python"),
    "math": ("gsm8k", "main"),
    "summ": ("cnn_dailymail", "3.0.0"),
}


def load_calibration_set(domain: str, tokenizer, n_samples: int = N_CALIB_SAMPLES, seq_len: int = CALIB_SEQ_LEN):
    """Return a list of tokenized (input_ids) tensors, ~seq_len tokens each, for `domain`."""
    from datasets import load_dataset

    name, config = _DATASET_BY_DOMAIN[domain]
    split = "train"
    ds = load_dataset(name, config, split=split, streaming=True)

    texts: List[str] = []
    for example in ds:
        text = example.get("text") or example.get("content") or example.get("article") or str(example)
        if len(text.strip()) > 0:
            texts.append(text)
        if len(texts) >= n_samples:
            break

    samples = []
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"]
        samples.append(ids)
    return samples


def load_domain_held_out_ppl_set(
    domain: str, tokenizer, n_samples: int = N_CALIB_SAMPLES, seq_len: int = CALIB_SEQ_LEN
):
    """Domain-specific held-out text for the sweep's perplexity metric (spec §4.1 amendment) —
    disjoint from `load_calibration_set`'s source for the same domain, per the module docstring
    table. `chat` reuses `load_wikitext2_eval` directly (same set as the M0 baseline, per the
    spec's original chat<->WikiText-2 pairing)."""
    if domain == "chat":
        return load_wikitext2_eval(tokenizer, seq_len=seq_len, n_chunks=n_samples)

    from datasets import load_dataset

    if domain == "code":
        name, config = _DATASET_BY_DOMAIN["code"]
        ds = load_dataset(name, config, split="train", streaming=True).skip(n_samples)
    elif domain == "math":
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    elif domain == "summ":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True)
    else:
        raise ValueError(f"unknown domain: {domain!r}")

    texts: List[str] = []
    for example in ds:
        text = example.get("text") or example.get("content") or example.get("article") or example.get("question") or str(example)
        if len(text.strip()) > 0:
            texts.append(text)
        if len(texts) >= n_samples:
            break

    samples = []
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"]
        samples.append(ids)
    return samples


def load_wikitext2_eval(tokenizer, seq_len: int = CALIB_SEQ_LEN, n_chunks: int = N_CALIB_SAMPLES):
    """Dedicated WikiText-2 perplexity eval set (spec §4.1 M0 acceptance metric) — deliberately
    separate from `load_calibration_set`, since the spec's own domain table distinguishes
    calibration *source* (what gets fake-quantized against) from eval *metric* (the held-out
    benchmark quality is measured on): the `chat` domain calibrates on OpenAssistant but is
    scored on WikiText-2 ppl, not on OpenAssistant ppl. Concatenates the raw test split and
    chunks into non-overlapping `seq_len` windows — the standard GPTQ/AWQ-style convention."""
    from datasets import load_dataset

    # "wikitext" (unnamespaced, script-based) is deprecated on the Hub; Salesforce/wikitext is
    # the current canonical rehost of the same data under the same config names.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for start in range(0, ids.shape[0] - seq_len, seq_len):
        chunks.append(ids[start : start + seq_len].unsqueeze(0))
        if len(chunks) >= n_chunks:
            break
    return chunks


def perplexity(model, calib_samples) -> float:
    import math

    import torch

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in calib_samples:
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += out.loss.item() * n
            total_tokens += n
    return math.exp(total_nll / max(total_tokens, 1))


def spearman_correlation(a: list, b: list) -> float:
    """Spearman rank correlation, used by the proxy validation step (spec §4.1) to check
    whether delta-ppl actually predicts delta-pass@1 before committing to the full sweep, and
    by eval/figures.py for the cross-domain correlation result (spec §4.3)."""
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)
