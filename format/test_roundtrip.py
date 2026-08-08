"""Round-trip tests for the .tsra nested bit-plane format.

Acceptance criteria (spec §10, M3): `.tsra` round-trips; loading at `k` planes matches the
reference `k`-bit computation within 1e-3 MSE.

These tests use small synthetic tensors (no model download / GPU required) so they can run in
CI and as a fast local check while iterating on the format.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from format.bitplane import (
    dequantize_at_bits,
    planes_to_q_bits,
    q_to_planes,
    quantize_asymmetric,
    unpack_bits,
    pack_bits,
)
from format.load import TsraFile
from format.pack import pack_file


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_bitplane_pack_unpack_is_identity(rng):
    q = rng.integers(0, 256, size=1000, dtype=np.uint16).astype(np.uint8)
    planes = q_to_planes(q)
    for k in range(8):
        packed = pack_bits(planes[k])
        back = unpack_bits(packed, n=1000)
        assert np.array_equal(back, planes[k])


def test_planes_to_q_bits_reconstructs_full_8_bit(rng):
    q = rng.integers(0, 256, size=500, dtype=np.uint16).astype(np.uint8)
    planes = q_to_planes(q)
    q_bits = planes_to_q_bits(planes, bits=8)
    assert np.array_equal(q_bits.astype(np.uint8), q)


def test_dequantize_at_8_bits_matches_direct_quantize(rng):
    w = rng.normal(size=2048).astype(np.float32)
    q, scales, zeros = quantize_asymmetric(w, group_size=128)
    planes = q_to_planes(q)
    q_bits = planes_to_q_bits(planes, bits=8)
    w_hat = dequantize_at_bits(q_bits, 8, scales, zeros, group_size=128, n=w.shape[0])
    # 8-bit reconstruction of an already-quantized value should be exact (no further truncation).
    w_hat_direct = (q.astype(np.float32) - np.repeat(zeros, 128)[: w.shape[0]]) * np.repeat(scales, 128)[: w.shape[0]]
    assert _mse(w_hat, w_hat_direct) < 1e-6


def test_lower_bits_are_monotonically_worse(rng):
    """Fewer planes should never reconstruct better than more planes, on average."""
    w = rng.normal(size=4096).astype(np.float32)
    q, scales, zeros = quantize_asymmetric(w, group_size=128)
    planes = q_to_planes(q)
    errors = []
    for bits in range(1, 9):
        q_bits = planes_to_q_bits(planes, bits)
        w_hat = dequantize_at_bits(q_bits, bits, scales, zeros, group_size=128, n=w.shape[0])
        errors.append(_mse(w, w_hat))
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] - 1e-6, f"bits={i+1} worse than bits={i+2}? {errors}"
    assert errors[-1] < 1e-2  # 8-bit should be reasonably tight for unit-normal data


def test_file_roundtrip_matches_bitplane_reference(tmp_path, rng):
    """pack_file -> TsraFile.get_tensor must match the same math run directly against
    bitplane.py, for every k. This is the actual format-correctness test (I/O layer is a
    no-op on top of already-verified math)."""
    w = rng.normal(size=(64, 256)).astype(np.float32)  # 2-D so pack.py routes it through planes
    tensors = {"blk.0.mlp_down": w, "blk.0.attn_norm.weight": rng.normal(size=64).astype(np.float32)}

    out_path = tmp_path / "test.tsra"
    pack_file(tensors, str(out_path), group_size=128)

    with TsraFile(str(out_path)) as tf:
        assert set(tf.tensor_names()) == set(tensors)

        # raw fp16 passthrough
        norm = tf.get_tensor("blk.0.attn_norm.weight", bits=8)
        np.testing.assert_allclose(norm, tensors["blk.0.attn_norm.weight"], atol=1e-2, rtol=1e-2)

        flat_ref = w.reshape(-1)
        q_ref, scales_ref, zeros_ref = quantize_asymmetric(flat_ref, group_size=128)
        planes_ref = q_to_planes(q_ref)

        for bits in (1, 2, 3, 4, 5, 6, 7, 8):
            loaded = tf.get_tensor("blk.0.mlp_down", bits=bits)
            q_bits_ref = planes_to_q_bits(planes_ref, bits)
            expected_flat = dequantize_at_bits(
                q_bits_ref, bits, scales_ref, zeros_ref, group_size=128, n=flat_ref.shape[0]
            )
            expected = expected_flat.reshape(w.shape)
            mse = _mse(loaded, expected)
            assert mse < 1e-3, f"bits={bits} mse={mse}"
            np.testing.assert_allclose(loaded, expected, atol=1e-4)


def test_file_header_and_magic(tmp_path, rng):
    w = rng.normal(size=(32, 128)).astype(np.float32)
    out_path = tmp_path / "hdr.tsra"
    pack_file({"t": w}, str(out_path), group_size=128)

    with open(out_path, "rb") as f:
        magic = f.read(4)
        version, tensor_count = struct.unpack("<II", f.read(8))
        (dir_offset,) = struct.unpack("<Q", f.read(8))
    assert magic == b"TSRA"
    assert version == 1
    assert tensor_count == 1
    assert dir_offset > 0


def test_measured_bytes_scales_with_bits(tmp_path, rng):
    w = rng.normal(size=(64, 256)).astype(np.float32)
    out_path = tmp_path / "budget.tsra"
    pack_file({"t": w}, str(out_path), group_size=128)

    with TsraFile(str(out_path)) as tf:
        b2 = tf.measured_bytes({"t": 2})
        b8 = tf.measured_bytes({"t": 8})
        assert b8 == 4 * b2  # linear in bits, same plane_size_bytes each
