"""Core bit-plane quantize/pack/unpack/dequantize math. Pure numpy, no I/O.

Shared by pack.py (writer), load.py (reader), and compiler/sweep.py (fake-quant reference
implementation must match this exactly, or sensitivity numbers don't mean anything).
"""
from __future__ import annotations

import numpy as np

N_PLANES = 8
BITS_MIN = 1
BITS_MAX = 8


def group_ranges(n: int, group_size: int) -> list[tuple[int, int]]:
    """Split `n` elements into contiguous [start, end) groups of at most `group_size`."""
    return [(i, min(i + group_size, n)) for i in range(0, n, group_size)]


def quantize_asymmetric(w_row: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a 1-D fp array to 8-bit unsigned, asymmetric, per group.

    Returns (q uint8 [n], scales f32 [num_groups], zeros f32 [num_groups]).
    """
    w_row = np.asarray(w_row, dtype=np.float32)
    n = w_row.shape[0]
    ranges = group_ranges(n, group_size)
    q = np.empty(n, dtype=np.uint8)
    scales = np.empty(len(ranges), dtype=np.float32)
    zeros = np.empty(len(ranges), dtype=np.float32)

    for gi, (start, end) in enumerate(ranges):
        g = w_row[start:end]
        w_min = float(g.min())
        w_max = float(g.max())
        span = w_max - w_min
        s = span / 255.0 if span > 0 else 1.0
        z = np.clip(round(-w_min / s), 0, 255)
        qg = np.clip(np.round(g / s + z), 0, 255).astype(np.uint8)
        q[start:end] = qg
        scales[gi] = s
        zeros[gi] = z
    return q, scales, zeros


def dequantize_at_bits(
    q_bits: np.ndarray,
    bits: int,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
    n: int,
) -> np.ndarray:
    """Reconstruct fp32 weights from the top `bits` planes' worth of quantized value.

    `q_bits` holds, per weight, the integer value assembled from the first `bits` planes
    (i.e. in [0, 2**bits - 1]). Applies the midpoint-correction formula from spec/spec.md §3,
    reusing the group's single derived (scale, zero) fit for every truncation level.
    """
    q_bits = q_bits.astype(np.float32)
    q_est = q_bits * (2 ** (8 - bits)) + (2 ** (7 - bits) if bits < 8 else 0)
    ranges = group_ranges(n, group_size)
    out = np.empty(n, dtype=np.float32)
    for gi, (start, end) in enumerate(ranges):
        out[start:end] = (q_est[start:end] - zeros[gi]) * scales[gi]
    return out


def q_to_planes(q: np.ndarray) -> np.ndarray:
    """uint8 [n] -> bit planes [8, n] of 0/1 uint8, MSB (plane 0) first."""
    n = q.shape[0]
    planes = np.empty((N_PLANES, n), dtype=np.uint8)
    for k in range(N_PLANES):
        shift = 7 - k  # plane 0 = bit 7 (MSB) ... plane 7 = bit 0 (LSB)
        planes[k] = (q >> shift) & 1
    return planes


def planes_to_q_bits(planes: np.ndarray, bits: int) -> np.ndarray:
    """First `bits` bit-planes [bits, n] of 0/1 -> assembled integer value [n] in [0, 2**bits-1]."""
    n = planes.shape[1]
    q_bits = np.zeros(n, dtype=np.uint16)
    for k in range(bits):
        shift = bits - 1 - k
        q_bits |= (planes[k].astype(np.uint16)) << shift
    return q_bits


def pack_bits(bitrow: np.ndarray) -> bytes:
    """[n] of 0/1 uint8 -> packed bytes, np.packbits (MSB-first within each byte)."""
    return np.packbits(bitrow).tobytes()


def unpack_bits(data: bytes, n: int) -> np.ndarray:
    """Inverse of pack_bits, truncated/padded to exactly n bits."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits[:n]
