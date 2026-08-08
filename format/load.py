"""mmap loader for .tsra — loading a tensor at `k` bits is one contiguous mmap range read.

Usage:
    tf = TsraFile("model.tsra")
    w = tf.get_tensor("blk.0.mlp_down", bits=4)   # np.float32, original shape
    tf.close()

Or with a manifest (spec §4.2 JSON, {"allocation": {tensor_name: bits, ...}}):
    tensors = tf.load_manifest(manifest["allocation"])
"""
from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Dict

import numpy as np

from format.bitplane import BITS_MAX, BITS_MIN, dequantize_at_bits, planes_to_q_bits, unpack_bits

MAGIC = b"TSRA"


class TsraFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        magic = bytes(self._mm[0:4])
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
        self.version, tensor_count = struct.unpack_from("<II", self._mm, 4)
        (dir_offset,) = struct.unpack_from("<Q", self._mm, 12)

        dir_json = bytes(self._mm[dir_offset:]).decode("utf-8")
        entries = json.loads(dir_json)["tensors"]
        assert len(entries) == tensor_count
        self.directory: Dict[str, dict] = {e["name"]: e for e in entries}

    def tensor_names(self) -> list[str]:
        return list(self.directory)

    def close(self) -> None:
        self._mm.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_tensor(self, name: str, bits: int = 8) -> np.ndarray:
        entry = self.directory[name]
        shape = tuple(entry["shape"])

        if entry["kind"] == "raw_fp16":
            start = entry["data_offset"]
            end = start + entry["data_size_bytes"]
            data = bytes(self._mm[start:end])
            return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)

        bits = max(BITS_MIN, min(bits, BITS_MAX))
        n = entry["num_weights"]
        group_size = entry["group_size"]
        plane_size = entry["plane_size_bytes"]
        plane0 = entry["plane_offsets"][0]

        planes = np.empty((bits, n), dtype=np.uint8)
        for k in range(bits):
            start = plane0 + k * plane_size
            end = start + plane_size
            planes[k] = unpack_bits(bytes(self._mm[start:end]), n)
        q_bits = planes_to_q_bits(planes, bits)

        scale_off = entry["scale_offset"]
        num_groups = entry["num_groups"]
        scale_bytes = bytes(self._mm[scale_off:scale_off + num_groups * 8])
        scale_arr = np.frombuffer(scale_bytes, dtype=np.float32).reshape(num_groups, 2)
        scales, zeros = scale_arr[:, 0], scale_arr[:, 1]

        flat = dequantize_at_bits(q_bits, bits, scales, zeros, group_size, n)
        return flat.reshape(shape)

    def load_manifest(self, allocation: Dict[str, int]) -> Dict[str, np.ndarray]:
        """allocation: {tensor_name: bits}. Tensors not present in the map default to 8 bits;
        raw_fp16 tensors ignore `bits` entirely."""
        out = {}
        for name in self.directory:
            bits = allocation.get(name, 8)
            out[name] = self.get_tensor(name, bits=bits)
        return out

    def measured_bytes(self, allocation: Dict[str, int]) -> int:
        """Bytes actually read off disk/mmap for a given allocation — what the allocator's
        `budget_bytes` is checked against."""
        total = 0
        for name, entry in self.directory.items():
            if entry["kind"] == "raw_fp16":
                total += entry["data_size_bytes"]
            else:
                bits = max(BITS_MIN, min(allocation.get(name, 8), BITS_MAX))
                total += bits * entry["plane_size_bytes"]
        return total
