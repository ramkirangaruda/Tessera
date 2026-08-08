"""Lagrangian knapsack allocator (spec §4.2).

Given per-tensor (bits -> (damage, bytes)) options from the sensitivity sweep, and a total byte
budget, choose a bit-width per tensor minimizing total damage subject to total bytes <= budget.

For a fixed multiplier `lambda`, the multiple-choice knapsack decomposes into an independent
per-tensor argmin of `damage(t,b) + lambda * bytes(t,b)`. Bisecting `lambda` until total bytes
hits budget gives a fast, near-optimal solution (this is the standard Lagrangian relaxation for
multiple-choice knapsack — exact when the per-tensor damage/bytes trade-off is convex in bytes,
which quantization damage curves are in practice).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

TensorOptions = Dict[int, Tuple[float, int]]  # bits -> (damage, bytes)

DEFAULT_BIT_CHOICES = (2, 3, 4, 5, 6, 8)
BIT_FLOOR = 2  # spec §4.2: hard floor of 2 bits on any tensor


def _solve_for_lambda(
    tensors: Dict[str, TensorOptions], lam: float, bit_floor: int = BIT_FLOOR
) -> Tuple[Dict[str, int], int, float]:
    allocation: Dict[str, int] = {}
    total_bytes = 0
    total_damage = 0.0
    for name, options in tensors.items():
        candidates = [(b, d, n) for b, (d, n) in options.items() if b >= bit_floor]
        if not candidates:
            raise ValueError(f"tensor {name!r} has no candidate bit-width >= floor {bit_floor}")
        b_best, d_best, n_best = min(candidates, key=lambda x: x[1] + lam * x[2])
        allocation[name] = b_best
        total_bytes += n_best
        total_damage += d_best
    return allocation, total_bytes, total_damage


def allocate(
    tensors: Dict[str, TensorOptions],
    budget_bytes: int,
    bit_floor: int = BIT_FLOOR,
    lam_hi_init: float = 1.0,
    iters: int = 60,
) -> Dict[str, int]:
    """Return {tensor_name: bits} minimizing total damage s.t. total bytes <= budget_bytes.

    Raises ValueError if the budget is infeasible even with every tensor at `bit_floor`.
    """
    alloc_lo, bytes_lo, _ = _solve_for_lambda(tensors, 0.0, bit_floor)
    if bytes_lo <= budget_bytes:
        return alloc_lo  # budget is generous enough for the min-damage (max bits) config

    # Grow lam_hi until the floor allocation is reachable / bytes_hi <= budget.
    lam_hi = lam_hi_init
    alloc_hi, bytes_hi, _ = _solve_for_lambda(tensors, lam_hi, bit_floor)
    tries = 0
    while bytes_hi > budget_bytes and tries < 64:
        lam_hi *= 2.0
        alloc_hi, bytes_hi, _ = _solve_for_lambda(tensors, lam_hi, bit_floor)
        tries += 1
    if bytes_hi > budget_bytes:
        raise ValueError(
            f"budget_bytes={budget_bytes} infeasible: floor allocation needs {bytes_hi} bytes"
        )

    lo, hi = 0.0, lam_hi
    best_alloc, best_bytes = alloc_hi, bytes_hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        alloc, nbytes, _ = _solve_for_lambda(tensors, mid, bit_floor)
        if nbytes <= budget_bytes:
            best_alloc, best_bytes = alloc, nbytes
            hi = mid
        else:
            lo = mid
    return best_alloc


def uniform_allocation(tensors: Dict[str, TensorOptions], bits: int) -> Dict[str, int]:
    """Baseline: every tensor at the same bit-width (nearest available <= bits)."""
    allocation = {}
    for name, options in tensors.items():
        available = sorted(b for b in options if b <= bits) or sorted(options)
        allocation[name] = available[-1] if available[-1] <= bits else min(options)
    return allocation


def total_bytes_of(tensors: Dict[str, TensorOptions], allocation: Dict[str, int]) -> int:
    return sum(tensors[name][allocation[name]][1] for name in tensors)


def total_damage_of(tensors: Dict[str, TensorOptions], allocation: Dict[str, int]) -> float:
    return sum(tensors[name][allocation[name]][0] for name in tensors)


def build_manifest(
    *,
    tensors: Dict[str, TensorOptions],
    allocation: Dict[str, int],
    domain: str,
    budget_bytes: int,
    model: str = "qwen3-1.7b",
    artifact_sha256: str = "",
    eval_metrics: dict | None = None,
) -> dict:
    return {
        "profile_id": f"{domain}@{budget_bytes // (1024 * 1024)}MB",
        "schema_version": 1,
        "model": model,
        "artifact_sha256": artifact_sha256,
        "domain": domain,
        "budget_bytes": budget_bytes,
        "measured_bytes": total_bytes_of(tensors, allocation),
        "allocation": allocation,
        "eval": eval_metrics or {},
    }


def load_sensitivity(parquet_path: str, domain: str) -> Dict[str, TensorOptions]:
    """Build the {tensor: {bits: (damage, bytes)}} structure for one domain from
    compiler/sweep.py's `sensitivity.parquet` output (columns: tensor_name, domain, bits,
    delta_ppl, delta_task_metric, bytes_at_bits, bytes_saved_vs_fp16)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = df[df["domain"] == domain]
    tensors: Dict[str, TensorOptions] = {}
    for name, group in df.groupby("tensor_name"):
        tensors[name] = {
            int(row.bits): (float(row.delta_ppl), int(row.bytes_at_bits))
            for row in group.itertuples()
        }
    return tensors


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensitivity", required=True, help="sensitivity.parquet from sweep.py")
    ap.add_argument("--domain", required=True, choices=["chat", "code", "math", "summ"])
    ap.add_argument("--budget-mb", type=float, required=True)
    ap.add_argument("--out", required=True, help="output manifest .json path")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--artifact", default=None, help=".tsra path, for the sha256 field")
    args = ap.parse_args()

    tensors = load_sensitivity(args.sensitivity, args.domain)
    budget_bytes = int(args.budget_mb * 1024 * 1024)
    allocation = allocate(tensors, budget_bytes)
    sha = sha256_of(args.artifact) if args.artifact else ""
    manifest = build_manifest(
        tensors=tensors,
        allocation=allocation,
        domain=args.domain,
        budget_bytes=budget_bytes,
        model=args.model,
        artifact_sha256=sha,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}: {manifest['measured_bytes']} bytes (budget {budget_bytes})")


if __name__ == "__main__":
    main()
