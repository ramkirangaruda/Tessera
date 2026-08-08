"""Unit tests for sweep.py's pure-numpy pieces (fake_quantize_tensor, bytes_at_bits,
quantizable_tensor_names) — no model/GPU/network required. The parts of sweep.py that need a
real model (run_sweep) are exercised only when pointed at an actual checkout; see module
docstring in compiler/sweep.py.
"""
from __future__ import annotations

import numpy as np

from compiler.sweep import (
    N_LAYERS,
    PER_LAYER_TENSORS,
    bytes_at_bits,
    fake_quantize_tensor,
    quantizable_tensor_names,
)


def test_quantizable_tensor_names_count_matches_spec():
    names = quantizable_tensor_names()
    # 7 per-layer tensors x 28 layers + token_embd == 197 knobs (spec §3).
    assert len(names) == len(PER_LAYER_TENSORS) * N_LAYERS + 1
    assert len(names) == 197
    assert "token_embd" in names
    assert "blk.0.attn_q" in names
    assert "blk.27.mlp_down" in names


def test_fake_quantize_tensor_shape_and_dtype_preserved():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 256)).astype(np.float32)
    for bits in (2, 3, 4, 5, 6, 8):
        w_hat = fake_quantize_tensor(w, bits)
        assert w_hat.shape == w.shape
        assert w_hat.dtype == w.dtype


def test_fake_quantize_tensor_error_decreases_with_bits():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(32, 512)).astype(np.float32)
    errors = []
    for bits in (2, 3, 4, 5, 6, 8):
        w_hat = fake_quantize_tensor(w, bits)
        errors.append(float(np.mean((w - w_hat) ** 2)))
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] - 1e-6


def test_bytes_at_bits_monotonic_in_bits():
    n = 1536 * 8960
    sizes = [bytes_at_bits(n, b) for b in (2, 3, 4, 5, 6, 8)]
    assert sizes == sorted(sizes)
    # sanity: well under fp16 size (n*2 bytes) even at 8 bits, since planes+scales are compact
    assert bytes_at_bits(n, 8) < n * 2
