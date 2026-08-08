"""fp16 safetensors -> .tsra nested bit-plane artifact.

Usage:
    python -m format.pack --in model.safetensors --out model.tsra [--group-size 128]

`is_quantizable(name, tensor)` decides routing: 2-D+ weight matrices are bit-plane encoded,
everything else (LayerNorm weight/bias, 1-D tensors) is stored raw fp16, matching spec §4.2
("LayerNorm weights and biases stay fp16 ... excluded from the search").
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from format.bitplane import N_PLANES, pack_bits, q_to_planes, quantize_asymmetric

MAGIC = b"TSRA"
VERSION = 1


def is_quantizable(name: str, arr: np.ndarray) -> bool:
    if arr.ndim < 2:
        return False
    if "norm" in name.lower():
        return False
    return True


def _pack_tensor_planes(arr2d: np.ndarray, group_size: int):
    """Quantize+bitplane-pack a 2-D tensor, grouping along the flattened row-major layout.

    Returns (plane_bytes: list[bytes] len 8, scale_bytes: bytes, num_groups, num_weights).
    """
    flat = arr2d.astype(np.float32).reshape(-1)
    n = flat.shape[0]
    q, scales, zeros = quantize_asymmetric(flat, group_size)
    planes = q_to_planes(q)  # [8, n]
    plane_bytes = [pack_bits(planes[k]) for k in range(N_PLANES)]
    scale_arr = np.empty((scales.shape[0], 2), dtype=np.float32)
    scale_arr[:, 0] = scales
    scale_arr[:, 1] = zeros
    return plane_bytes, scale_arr.tobytes(), scales.shape[0], n


def pack_file(tensors: Dict[str, np.ndarray], out_path: str, group_size: int = 128) -> None:
    directory = []
    plane_blob = bytearray()
    scale_blob = bytearray()
    raw_blob = bytearray()

    for name, arr in tensors.items():
        if is_quantizable(name, arr):
            plane_bytes, scale_bytes, num_groups, n = _pack_tensor_planes(arr, group_size)
            plane_offsets = []
            for pb in plane_bytes:
                plane_offsets.append(len(plane_blob))  # relative to plane section start; fixed up below
                plane_blob.extend(pb)
            plane_size_bytes = len(plane_bytes[0])
            directory.append({
                "name": name,
                "kind": "planes",
                "shape": list(arr.shape),
                "group_size": group_size,
                "num_groups": num_groups,
                "num_weights": n,
                "plane_offsets": plane_offsets,
                "plane_size_bytes": plane_size_bytes,
                "scale_offset": len(scale_blob),
                "scale_dtype": "float32",
            })
            scale_blob.extend(scale_bytes)
        else:
            data = arr.astype(np.float16).tobytes()
            directory.append({
                "name": name,
                "kind": "raw_fp16",
                "shape": list(arr.shape),
                "data_offset": len(raw_blob),
                "data_size_bytes": len(data),
            })
            raw_blob.extend(data)

    header_size = 4 + 4 + 4 + 8  # magic, version, tensor_count, dir_offset
    plane_section_start = header_size
    scale_section_start = plane_section_start + len(plane_blob)
    raw_section_start = scale_section_start + len(scale_blob)
    dir_start = raw_section_start + len(raw_blob)

    for entry in directory:
        if entry["kind"] == "planes":
            entry["plane_offsets"] = [o + plane_section_start for o in entry["plane_offsets"]]
            entry["scale_offset"] += scale_section_start
        else:
            entry["data_offset"] += raw_section_start

    dir_json = json.dumps({"tensors": directory}).encode("utf-8")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(directory)))
        f.write(struct.pack("<Q", dir_start))
        assert f.tell() == header_size
        f.write(plane_blob)
        f.write(scale_blob)
        f.write(raw_blob)
        f.write(dir_json)


def load_safetensors_as_numpy(path: str) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, help="input .safetensors (fp16)")
    ap.add_argument("--out", dest="out_path", required=True, help="output .tsra path")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    tensors = load_safetensors_as_numpy(args.in_path)
    pack_file(tensors, args.out_path, group_size=args.group_size)
    print(f"wrote {args.out_path}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
