"""Pareto plot, allocation heatmap, and cross-domain correlation matrix (spec §8, §4.3, §13).

Static-image counterparts of the live dashboard panels, for the write-up/pitch deck rather than
the demo itself. Pure numpy/matplotlib/scipy — runs on synthetic data with no model or GPU, so
this module is fully testable (test_figures.py) even before M1's sensitivity.parquet exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; no display required in CI or on the demo laptop's terminal
import matplotlib.pyplot as plt
import numpy as np

from compiler.sweep import N_LAYERS, PER_LAYER_TENSORS


def plot_pareto(points: List[dict], out_path: str) -> None:
    """points: [{label, memory_mb, quality, kind: 'uniform'|'tessera'}, ...] (spec §8 panel 3,
    §13: "Tessera vs uniform Q2/Q3/Q4/Q6/Q8")."""
    fig, ax = plt.subplots(figsize=(7, 5))

    uniform = sorted((p for p in points if p["kind"] == "uniform"), key=lambda p: p["memory_mb"])
    tessera = [p for p in points if p["kind"] == "tessera"]

    if uniform:
        ax.plot(
            [p["memory_mb"] for p in uniform],
            [p["quality"] for p in uniform],
            "o-",
            color="#7a7a7a",
            label="uniform baseline",
        )
        for p in uniform:
            ax.annotate(p["label"], (p["memory_mb"], p["quality"]), fontsize=8, color="#7a7a7a")

    if tessera:
        ax.scatter(
            [p["memory_mb"] for p in tessera],
            [p["quality"] for p in tessera],
            color="#4bb89e",
            s=60,
            zorder=3,
            label="Tessera profiles",
        )
        for p in tessera:
            ax.annotate(p["label"], (p["memory_mb"], p["quality"]), fontsize=8, color="#2f7a68")

    ax.set_xlabel("memory (MB)")
    ax.set_ylabel("quality")
    ax.set_title("Quality vs. memory")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(allocation: Dict[str, int], out_path: str, n_layers: int = N_LAYERS) -> None:
    """28 rows (layers) x 7 columns (tensor kinds), each cell colored by bit-width, plus a
    separate token_embd swatch (spec §8 panel 2 — "the single most important visual")."""
    grid = np.full((n_layers, len(PER_LAYER_TENSORS)), np.nan)
    for layer in range(n_layers):
        for col, kind in enumerate(PER_LAYER_TENSORS):
            name = f"blk.{layer}.{kind}"
            if name in allocation:
                grid[layer, col] = allocation[name]

    fig, (ax_grid, ax_embd) = plt.subplots(
        1, 2, figsize=(6, 8), gridspec_kw={"width_ratios": [7, 1]}, sharey=False
    )

    im = ax_grid.imshow(grid, cmap="viridis", vmin=2, vmax=8, aspect="auto")
    ax_grid.set_xticks(range(len(PER_LAYER_TENSORS)))
    ax_grid.set_xticklabels(PER_LAYER_TENSORS, rotation=45, ha="right", fontsize=8)
    ax_grid.set_yticks(range(0, n_layers, 2))
    ax_grid.set_yticklabels([f"L{i}" for i in range(0, n_layers, 2)], fontsize=7)
    ax_grid.set_title("Per-layer allocation")

    embd_bits = allocation.get("token_embd", np.nan)
    ax_embd.imshow([[embd_bits]], cmap="viridis", vmin=2, vmax=8, aspect="auto")
    ax_embd.set_xticks([0])
    ax_embd.set_xticklabels(["token_embd"], rotation=45, ha="right", fontsize=8)
    ax_embd.set_yticks([])
    ax_embd.set_title(f"{embd_bits}b" if not np.isnan(embd_bits) else "?")

    fig.colorbar(im, ax=[ax_grid, ax_embd], label="bits", shrink=0.6)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def spearman_correlation_matrix(domain_rankings: Dict[str, Sequence[float]]) -> "np.ndarray":
    """domain_rankings: {domain: [damage_at_some_fixed_bits_per_tensor, ...]} in a consistent
    tensor order across domains. Returns the len(domains) x len(domains) Spearman correlation
    matrix (spec §4.3: "compute the rank correlation between domain sensitivity orderings").
    Low correlation = task-conditioned allocation is justified; high = say so honestly."""
    from scipy.stats import spearmanr

    domains = list(domain_rankings)
    n = len(domains)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(domain_rankings[domains[i]], domain_rankings[domains[j]])
            corr[i, j] = corr[j, i] = rho
    return corr


def plot_correlation_matrix(domain_rankings: Dict[str, Sequence[float]], out_path: str) -> None:
    domains = list(domain_rankings)
    corr = spearman_correlation_matrix(domain_rankings)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)
    for i in range(len(domains)):
        for j in range(len(domains)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cross-domain sensitivity rank correlation (Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
