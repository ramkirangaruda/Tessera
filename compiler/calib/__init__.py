"""Calibration set loaders + eval metrics for the sensitivity sweep (spec §4.1 table).

| Domain | Calibration source                          | Eval metric                     |
|--------|----------------------------------------------|----------------------------------|
| chat   | LMSYS-style multi-turn / OpenAssistant        | WikiText-2 perplexity            |
| code   | The Stack (Python subset) / CodeAlpaca        | HumanEval subset, pass@1         |
| math   | GSM8K train split                             | GSM8K test subset, exact match   |
| summ   | CNN/DailyMail                                 | ROUGE-L or perplexity            |

~128 samples/domain, ~2048 tokens/sample. Needs `datasets` + network access; not exercised by the
test suite. Kept as a thin, swappable layer so a team member can point it at local calibration
files instead of the Hub without touching sweep.py.
"""
from __future__ import annotations

from typing import List

N_CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 2048

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


def task_metric(model, tokenizer, domain: str) -> float:
    """HumanEval pass@1 (code), GSM8K exact-match (math), ROUGE-L (summ), or WikiText-2 ppl
    proxy (chat) — delegates to lm-eval-harness. Placeholder returning 0.0 pending harness
    wiring in eval/harness.py, which M0 stands up first."""
    return 0.0
