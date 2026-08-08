"""Manifest-driven generation entrypoint (spec §4, M4).

Thin CLI over runtime/model.py: pick a profile manifest, load the model at that precision, run
bounded warm prefill (spec §6 — system prompt + summary + last N turns + retrieved spans, never
the full transcript), and generate. Needs torch/transformers + a real .tsra artifact; not
runnable in this environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

WARM_PREFILL_LAST_N_TURNS = 6  # spec §6, "N ~= 6, tuned" — see docs/spec.md §15 open decision 7


def build_prefill_prompt(system_prompt: str, summary_md: str, last_turns: list[dict], retrieved_spans: list[str]) -> str:
    """Constant-cost prefill regardless of session length (spec §6)."""
    parts = [system_prompt.strip()]
    if summary_md.strip():
        parts.append(f"## Session summary\n{summary_md.strip()}")
    if retrieved_spans:
        parts.append("## Relevant earlier context\n" + "\n---\n".join(retrieved_spans))
    for turn in last_turns[-WARM_PREFILL_LAST_N_TURNS:]:
        parts.append(f"{turn['role']}: {turn['content']}")
    return "\n\n".join(parts)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = n_new / elapsed if elapsed > 0 else float("inf")
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, tokens_per_sec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="dir with config.json + tokenizer files")
    ap.add_argument("--tsra", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bundle", help="portable session bundle dir (spec §6); optional")
    ap.add_argument("--prompt", required=True, help="the new user turn")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from runtime.model import from_manifest
    from transformers import AutoTokenizer

    manifest = json.loads(Path(args.manifest).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.config)
    model = from_manifest(args.config, args.tsra, manifest, device=args.device)

    system_prompt = "You are a helpful assistant."
    summary_md, last_turns, retrieved_spans = "", [], []
    if args.bundle:
        bundle = Path(args.bundle)
        summary_md = (bundle / "summary.md").read_text() if (bundle / "summary.md").exists() else ""
        transcript_path = bundle / "transcript.jsonl"
        if transcript_path.exists():
            last_turns = [json.loads(line) for line in transcript_path.read_text().splitlines() if line.strip()]

    prefill = build_prefill_prompt(system_prompt, summary_md, last_turns, retrieved_spans)
    full_prompt = f"{prefill}\n\nuser: {args.prompt}\nassistant:"

    text, tps = generate(model, tokenizer, full_prompt)
    print(text)
    print(f"\n[{tps:.1f} tok/s, profile={manifest['profile_id']}, measured_bytes={manifest['measured_bytes']}]")


if __name__ == "__main__":
    main()
