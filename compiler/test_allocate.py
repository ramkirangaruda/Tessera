"""Allocator tests on synthetic sensitivity data (no model / GPU / parquet file required).

Acceptance criteria (spec §10, M2): given (domain, budget), emits a valid manifest; at equal
bytes, beats uniform quantization on quality.
"""
from __future__ import annotations

import random

import pytest

from compiler.allocate import (
    DEFAULT_BIT_CHOICES,
    allocate,
    build_manifest,
    total_bytes_of,
    total_damage_of,
    uniform_allocation,
)


def make_synthetic_tensors(n_tensors=50, seed=0):
    """Damage strictly decreasing in bits, bytes strictly increasing in bits, per tensor —
    with per-tensor-varying slopes so the allocator actually has to trade off across tensors
    (this is what makes uniform allocation suboptimal)."""
    rng = random.Random(seed)
    tensors = {}
    for i in range(n_tensors):
        base_bytes = rng.randint(1000, 50000)
        sensitivity = rng.uniform(0.1, 5.0)  # how much this tensor "hurts" at low bits
        options = {}
        for b in DEFAULT_BIT_CHOICES:
            nbytes = int(base_bytes * b / 8)
            damage = sensitivity * (2.0 ** (8 - b)) / 256.0  # decays fast as bits grow
            options[b] = (damage, nbytes)
        tensors[f"tensor_{i}"] = options
    return tensors


def test_allocate_respects_budget():
    tensors = make_synthetic_tensors()
    max_bytes = total_bytes_of(tensors, {n: 8 for n in tensors})
    budget = max_bytes // 2
    allocation = allocate(tensors, budget)
    assert total_bytes_of(tensors, allocation) <= budget


def test_allocate_respects_bit_floor():
    tensors = make_synthetic_tensors()
    min_bytes = total_bytes_of(tensors, {n: 2 for n in tensors})
    allocation = allocate(tensors, min_bytes)
    assert all(b >= 2 for b in allocation.values())


def test_allocate_infeasible_budget_raises():
    tensors = make_synthetic_tensors()
    min_bytes = total_bytes_of(tensors, {n: 2 for n in tensors})
    with pytest.raises(ValueError):
        allocate(tensors, min_bytes // 2)


def test_allocate_generous_budget_picks_max_bits():
    tensors = make_synthetic_tensors()
    max_bytes = total_bytes_of(tensors, {n: 8 for n in tensors})
    allocation = allocate(tensors, max_bytes * 2)
    assert all(b == 8 for b in allocation.values())


def test_allocate_beats_uniform_at_equal_bytes():
    """The whole point of the allocator: at a matched byte budget, non-uniform beats uniform
    on total damage, because it can spend bytes where sensitivity is highest."""
    tensors = make_synthetic_tensors(n_tensors=100, seed=1)
    max_bytes = total_bytes_of(tensors, {n: 8 for n in tensors})
    budget = int(max_bytes * 0.4)

    smart = allocate(tensors, budget)
    smart_bytes = total_bytes_of(tensors, smart)
    smart_damage = total_damage_of(tensors, smart)

    # nearest uniform bit-width under the same budget
    best_uniform_damage = None
    for bits in DEFAULT_BIT_CHOICES:
        u = uniform_allocation(tensors, bits)
        if total_bytes_of(tensors, u) <= budget:
            d = total_damage_of(tensors, u)
            if best_uniform_damage is None or d < best_uniform_damage:
                best_uniform_damage = d

    assert smart_bytes <= budget
    assert best_uniform_damage is not None
    assert smart_damage <= best_uniform_damage


def test_build_manifest_shape():
    tensors = make_synthetic_tensors(n_tensors=10)
    budget = total_bytes_of(tensors, {n: 8 for n in tensors})
    allocation = allocate(tensors, budget)
    manifest = build_manifest(
        tensors=tensors, allocation=allocation, domain="code", budget_bytes=budget
    )
    assert manifest["domain"] == "code"
    assert manifest["schema_version"] == 1
    assert set(manifest["allocation"]) == set(tensors)
    assert manifest["measured_bytes"] <= budget
    assert manifest["profile_id"].startswith("code@")
