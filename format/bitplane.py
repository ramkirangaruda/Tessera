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

    Vectorized over groups (reshape + axis-wise min/max) rather than a Python loop per group —
    at ~1.7B params / group_size=128 that's ~13M groups; a pure-Python loop over that many groups
    is the difference between packing a real model in seconds vs. minutes (measured: ~320s
    unvectorized for Qwen3-1.7B, well over the M3 load-time budget once the same pattern shows up
    on the read side in dequantize_at_bits below).
    """
    w_row = np.asarray(w_row, dtype=np.float32)
    n = w_row.shape[0]
    n_full = n // group_size
    remainder = n - n_full * group_size
    num_groups = n_full + (1 if remainder else 0)

    q = np.empty(n, dtype=np.uint8)
    scales = np.empty(num_groups, dtype=np.float32)
    zeros = np.empty(num_groups, dtype=np.float32)

    if n_full > 0:
        bulk = w_row[: n_full * group_size].reshape(n_full, group_size)
        w_min = bulk.min(axis=1)
        w_max = bulk.max(axis=1)
        span = w_max - w_min
        s = np.where(span > 0, span / 255.0, 1.0).astype(np.float32)
        z = np.clip(np.round(-w_min / s), 0, 255).astype(np.float32)
        qb = np.clip(np.round(bulk / s[:, None] + z[:, None]), 0, 255).astype(np.uint8)
        q[: n_full * group_size] = qb.reshape(-1)
        scales[:n_full] = s
        zeros[:n_full] = z

    if remainder:
        g = w_row[n_full * group_size :]
        w_min, w_max = float(g.min()), float(g.max())
        span = w_max - w_min
        s = span / 255.0 if span > 0 else 1.0
        z = float(np.clip(round(-w_min / s), 0, 255))
        qg = np.clip(np.round(g / s + z), 0, 255).astype(np.uint8)
        q[n_full * group_size :] = qg
        scales[n_full] = s
        zeros[n_full] = z

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

    Vectorized the same way as quantize_asymmetric — this is on the hot path for every model
    load (format/load.py calls it once per tensor), so an unvectorized version here is what
    actually blows the M3 "load a ~1.1 GB profile in under 8s" budget (measured: ~220s for a
    full Qwen3-1.7B materialize before this fix).
    """
    q_bits = q_bits.astype(np.float32)
    q_est = q_bits * (2 ** (8 - bits)) + (2 ** (7 - bits) if bits < 8 else 0)

    n_full = n // group_size
    remainder = n - n_full * group_size
    out = np.empty(n, dtype=np.float32)

    if n_full > 0:
        bulk = q_est[: n_full * group_size].reshape(n_full, group_size)
        s = scales[:n_full][:, None]
        z = zeros[:n_full][:, None]
        out[: n_full * group_size] = ((bulk - z) * s).reshape(-1)

    if remainder:
        g = q_est[n_full * group_size :]
        out[n_full * group_size :] = (g - zeros[n_full]) * scales[n_full]

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
    """First `bits` bit-planes [bits, n] of 0/1 -> assembled integer value [n] in [0, 2**bits-1].

    Deliberately a plain per-plane loop (at most 8 iterations, cheap in Python-call overhead) —
    NOT `(planes[:bits].astype(np.uint16) * weights).sum(axis=0)`, which looks more "vectorized"
    but materializes a full (bits, n) intermediate before reducing it. Measured on a real
    311M-element tensor: this loop is ~6.1s, the multiply-then-sum version is ~85.7s. The earlier
    per-*group* loop in quantize_asymmetric/dequantize_at_bits was a real win to vectorize away
    (millions of groups, Python call overhead dominates); this per-*plane* loop tops out at 8
    iterations, so the loop's overhead was never the bottleneck — the naive "vectorization" here
    was memory-bandwidth-bound and actively worse. Don't re-"optimize" this without re-measuring.
    """
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
