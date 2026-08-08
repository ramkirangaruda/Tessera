"""Tests for eval/figures.py against synthetic data — no model/GPU/sensitivity.parquet
required. Verifies each plotting function runs end to end and produces a non-empty image file
(not pixel-level correctness, but catches import/shape/API breakage)."""
from __future__ import annotations

import numpy as np
import pytest

from compiler.sweep import N_LAYERS, PER_LAYER_TENSORS, quantizable_tensor_names
from eval.figures import (
    plot_correlation_matrix,
    plot_heatmap,
    plot_pareto,
    spearman_correlation_matrix,
)


def test_plot_pareto_writes_file(tmp_path):
    points = [
        {"label": "Q2", "memory_mb": 420, "quality": 0.31, "kind": "uniform"},
        {"label": "Q4", "memory_mb": 810, "quality": 0.62, "kind": "uniform"},
        {"label": "Q8", "memory_mb": 1540, "quality": 0.88, "kind": "uniform"},
        {"label": "code@1100MB", "memory_mb": 1096, "quality": 0.74, "kind": "tessera"},
    ]
    out = tmp_path / "pareto.png"
    plot_pareto(points, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_heatmap_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    allocation = {name: int(rng.integers(2, 9)) for name in quantizable_tensor_names()}
    out = tmp_path / "heatmap.png"
    plot_heatmap(allocation, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_heatmap_handles_missing_tensors(tmp_path):
    """A partial allocation (e.g. mid-sweep) shouldn't crash the figure — missing cells are NaN."""
    out = tmp_path / "heatmap_partial.png"
    plot_heatmap({"token_embd": 6, "blk.0.attn_q": 4}, str(out))
    assert out.exists()


def test_spearman_correlation_matrix_diagonal_is_one():
    rng = np.random.default_rng(1)
    rankings = {
        "chat": rng.normal(size=50).tolist(),
        "code": rng.normal(size=50).tolist(),
        "math": rng.normal(size=50).tolist(),
    }
    corr = spearman_correlation_matrix(rankings)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)  # symmetric


def test_spearman_correlation_matrix_detects_identical_rankings():
    values = list(range(50))
    rankings = {"chat": values, "code": values, "math": values[::-1]}
    corr = spearman_correlation_matrix(rankings)
    assert corr[0, 1] == pytest.approx(1.0)  # identical rankings -> perfect positive correlation
    assert corr[0, 2] == pytest.approx(-1.0)  # reversed ranking -> perfect negative correlation


def test_plot_correlation_matrix_writes_file(tmp_path):
    rng = np.random.default_rng(2)
    rankings = {d: rng.normal(size=30).tolist() for d in ("chat", "code", "math", "summ")}
    out = tmp_path / "corr.png"
    plot_correlation_matrix(rankings, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
