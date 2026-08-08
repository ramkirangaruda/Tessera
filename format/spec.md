# `.tsra` — Tessera nested bit-plane weight format

Formalizes engineering spec §5. One file, readable at any integer bit-width `k ∈ [1, 8]`, without
re-quantization: loading at `k` bits is a single contiguous `mmap` range read.

## 1. Quantization

Weights are grouped along the flattened last (input) dimension into groups of `group_size`
(default 128, config knob — remainder group keeps its own scale even if shorter). Each group is
quantized to 8-bit unsigned, asymmetric:

```
s = (max(w) - min(w)) / 255          # scale
z = round(-min(w) / s)               # zero point, clamped to [0, 255]
q = clamp(round(w / s + z), 0, 255)  # uint8
```

`LayerNorm` weights/biases and any 1-D tensor are stored raw as fp16 — negligible bytes, excluded
from the search space per spec §4.2, never bit-plane encoded.

## 2. Bit-plane decomposition

Each `q` (uint8) is decomposed into 8 single-bit planes, **MSB first**: plane 0 holds bit 7 of
every weight in the tensor, plane 1 holds bit 6, ... plane 7 holds bit 0. Planes are bitpacked
(8 weights/byte) and stored **contiguously per tensor, plane-major**: all of plane 0, then all of
plane 1, etc. This contiguity is the whole point — reading planes `0..k-1` is one mmap range.

## 3. Reconstruction at `k` bits

```
q_k    = value assembled from the first k planes         # in [0, 2^k - 1]
q_est  = q_k * 2^(8-k) + 2^(7-k)                          # midpoint of the represented interval
w_hat  = (q_est - z) * s
```

This is the **derived-scale** reconstruction — it reuses the group's single `(s, z)` fit at 8
bits for every truncation level. It is what `pack.py`/`load.py` implement today.

**Not yet implemented (spec §5.1, tracked as stretch goal #2 in §9 of the top-level spec):**
fitted per-`k` refinement scales `s_k`, one per group per level, chosen to minimize MSE at that
specific truncation depth. These close the "nesting tax" against an independently-optimized
`k`-bit quantization and cost at most 1 bit/weight (8 × 2 bytes/group) if all eight are stored.

## 4. File layout

```
[ magic "TSRA" | u32 version | u32 tensor_count | u64 dir_offset ]   # 20-byte header
[ plane data — contiguous per tensor, plane-major                ]
[ scale/zero data — one (f32 scale, f32 zero) pair per group     ]
[ tensor directory (json, utf-8) at dir_offset                   ]
```

The directory is a single length-prefixed JSON blob (simplicity over a binary struct — this file
is at most ~1.5B params / 197 tensors, directory parse cost is irrelevant next to the mmap read).
Per tensor:

```json
{
  "name": "blk.0.mlp_down",
  "kind": "planes",
  "shape": [8960, 1536],
  "group_size": 128,
  "num_groups": 8960,
  "num_weights": 13762560,
  "plane_offsets": [/* 8 absolute byte offsets into the file, one per plane */],
  "plane_size_bytes": 1720320,
  "scale_offset": 0,
  "scale_dtype": "float32"
}
```

`kind` is `"planes"` (bit-plane encoded) or `"raw_fp16"` (LayerNorm weight/bias, embeddings not
routed through the allocator, etc). Raw entries carry `data_offset` / `data_size_bytes` instead of
plane fields.

## 5. Loading

`load.py` mmaps the file once, parses the directory, and for a given `(tensor_name, bits)`:

1. Slices `plane_offsets[0] : plane_offsets[0] + bits * plane_size_bytes` — one contiguous read
   (planes are laid out back-to-back in bit order, so this is `[plane_offsets[0], plane_offsets[bits])`
   when `bits < 8`; for `bits == 8` it's the whole plane region).
2. Unpacks bits, reassembles `q_k`, applies the reconstruction formula in §3.
3. Returns an `np.ndarray` (`float32`) of the original shape — framework-agnostic; the runtime
   wraps it in a `torch.Tensor` on load.

Runtime bit-plane drop (spec §5.3) is the same operation re-issued with a smaller `bits`, on an
already-open mmap — no file re-read, no re-quantization.
