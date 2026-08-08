"""Manifest-driven Qwen3 loader (spec §4, M4; model family amended from Qwen2.5 — see spec §2).

Loads a HF Qwen3 model shell from config only (no pretrained fp16 download — the weights come
from the .tsra artifact), then materializes each parameter by reading exactly the tensor/bits
called for in a profile manifest's `allocation` map, via format.load.TsraFile (spec §5.3: this is
one mmap range per tensor, no re-quantization).

The name map below covers both `Qwen3ForCausalLM` (primary: `q_norm`/`k_norm`, no attention
bias — `attention_bias: false` in Qwen3's config.json, verified against the real checkpoint) and
`Qwen2ForCausalLM` (secondary validation run, spec §13 cross-model replication: has attention
q/k/v bias, no q_norm/k_norm). A given .tsra file only ever contains the tensors its source model
actually has, so unused entries in the map are simply never looked up.

Requires torch + transformers, and a real Qwen3-1.7B/-0.6B checkout for config.json + tokenizer
(weights are NOT downloaded via HF — see `from_manifest`; they come from `pack_from_pretrained`
below, once, to build the .tsra artifact). The tensor-name mapping itself is unit-testable
without either dependency (test_model.py).
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from format.load import TsraFile

N_LAYERS_QWEN3_1_7B = 28
N_LAYERS_QWEN3_0_6B = 28  # verified against config.json (spec §2 amendment) — both sizes are 28

# .tsra tensor name -> HF state_dict key template. `{i}` is the layer index.
_PER_LAYER_MAP = {
    "attn_q": "model.layers.{i}.self_attn.q_proj.weight",
    "attn_k": "model.layers.{i}.self_attn.k_proj.weight",
    "attn_v": "model.layers.{i}.self_attn.v_proj.weight",
    "attn_o": "model.layers.{i}.self_attn.o_proj.weight",
    "mlp_gate": "model.layers.{i}.mlp.gate_proj.weight",
    "mlp_up": "model.layers.{i}.mlp.up_proj.weight",
    "mlp_down": "model.layers.{i}.mlp.down_proj.weight",
    # Qwen2.5 only (secondary validation run) — Qwen3 has attention_bias=false, no qkv bias.
    "attn_q.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "attn_k.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "attn_v.bias": "model.layers.{i}.self_attn.v_proj.bias",
    # Qwen3 only (spec §2 amendment) — per-head RMSNorm on head_dim, Qwen2.5 has neither.
    "attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
    "attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
}
_GLOBAL_MAP = {
    "token_embd": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    # tied embeddings (spec §3: tied_word_embeddings: true on both Qwen3 sizes, verified) —
    # lm_head shares token_embd, no separate entry
}
_HF_TO_TSRA = {v: k for k, v in _GLOBAL_MAP.items()}
_HF_PER_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# "model.layers.{i}.self_attn.q_proj.weight" -> suffix "self_attn.q_proj.weight" -> kind "attn_q"
_TSRA_KIND_BY_HF_SUFFIX = {tmpl.split(".", 3)[3]: kind for kind, tmpl in _PER_LAYER_MAP.items()}


def tsra_name_to_hf_name(tsra_name: str) -> str:
    if tsra_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[tsra_name]
    # "blk.{i}.<rest>"
    prefix, rest = tsra_name.split(".", 1)
    if prefix != "blk":
        raise ValueError(f"unrecognized .tsra tensor name: {tsra_name!r}")
    i_str, tensor_kind = rest.split(".", 1)
    i = int(i_str)
    if tensor_kind not in _PER_LAYER_MAP:
        raise ValueError(f"unrecognized per-layer tensor kind: {tensor_kind!r} in {tsra_name!r}")
    return _PER_LAYER_MAP[tensor_kind].format(i=i)


def hf_name_to_tsra_name(hf_name: str) -> Optional[str]:
    """Inverse of tsra_name_to_hf_name — used by pack_from_pretrained to translate a real HF
    state dict's key names into the canonical Tessera short-name scheme before packing (spec
    §4.2 manifests, the dashboard heatmap, and compiler/sweep.py all key off this scheme, not
    raw HF names). Returns None for params with no .tsra counterpart (e.g. a separate `lm_head`
    weight, which shouldn't exist here since both target models tie embeddings — spec §3)."""
    if hf_name in _HF_TO_TSRA:
        return _HF_TO_TSRA[hf_name]
    m = _HF_PER_LAYER_PATTERN.match(hf_name)
    if not m:
        return None
    i, suffix = m.group(1), m.group(2)
    kind = _TSRA_KIND_BY_HF_SUFFIX.get(suffix)
    if kind is None:
        return None
    return f"blk.{i}.{kind}"


def pack_from_pretrained(model_name: str, out_path: str, group_size: int = 128) -> None:
    """Load a real HF checkpoint (Qwen3 primary, or Qwen2 for the §13 secondary validation run)
    and pack it to `.tsra` under the canonical Tessera tensor names, via hf_name_to_tsra_name.
    format.pack itself stays model-agnostic (spec §5: "framework-agnostic") — this function is
    the Qwen-specific adapter that lives on the runtime side instead."""
    import torch
    from transformers import AutoModelForCausalLM

    from format.pack import pack_file

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    tensors = {}
    skipped = []
    for hf_name, param in model.state_dict().items():
        tsra_name = hf_name_to_tsra_name(hf_name)
        if tsra_name is None:
            skipped.append(hf_name)
            continue
        tensors[tsra_name] = param.detach().float().cpu().numpy()

    unexpected_skips = [n for n in skipped if "lm_head" not in n and "rotary" not in n]
    if unexpected_skips:
        raise RuntimeError(
            f"{len(unexpected_skips)} unmapped tensors in {model_name} — extend the name map "
            f"in runtime/model.py before packing: {unexpected_skips[:5]}..."
        )

    pack_file(tensors, out_path, group_size=group_size)


def hf_state_dict_from_manifest(tsra_path: str, allocation: Dict[str, int]) -> Dict[str, "object"]:
    """{hf_param_name: np.ndarray} materialized at the manifest's per-tensor bit-width.
    Caller (from_manifest below) wraps these in torch.Tensor and load_state_dict()s them."""
    state: Dict[str, "object"] = {}
    with TsraFile(tsra_path) as tf:
        for tsra_name in tf.tensor_names():
            bits = allocation.get(tsra_name, 8)
            state[tsra_name_to_hf_name(tsra_name)] = tf.get_tensor(tsra_name, bits=bits)
    return state


def from_manifest(config_path: str, tsra_path: str, manifest: dict, device: str = "cpu"):
    """Build a live model from a manifest. `config_path` points at a directory containing the
    target model's config.json + tokenizer files (no weights needed — HF `from_config` builds
    randomly-initialized tensors of the right shape, which we then overwrite in place)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model.eval()

    state_np = hf_state_dict_from_manifest(tsra_path, manifest["allocation"])
    own_state = model.state_dict()
    missing = [k for k in own_state if k not in state_np and "lm_head" not in k]
    if missing:
        raise RuntimeError(f"{len(missing)} model params not covered by .tsra artifact: {missing[:5]}...")

    with torch.no_grad():
        for name, arr in state_np.items():
            own_state[name].copy_(torch.from_numpy(arr).to(own_state[name].dtype))

    return model.to(device)


def drop_to_bits(model, tsra_path: str, allocation: Dict[str, int], new_allocation: Dict[str, int]) -> None:
    """Runtime bit-plane drop under memory pressure (spec §5.3): re-materialize only the tensors
    whose bit-width decreased, in place, from the already-open mmap — no file re-read beyond the
    (smaller) plane range, no re-quantization."""
    import torch

    changed = {n: b for n, b in new_allocation.items() if b < allocation.get(n, 8)}
    if not changed:
        return
    own_state = model.state_dict()
    with TsraFile(tsra_path) as tf:
        with torch.no_grad():
            for tsra_name, bits in changed.items():
                hf_name = tsra_name_to_hf_name(tsra_name)
                arr = tf.get_tensor(tsra_name, bits=bits)
                own_state[hf_name].copy_(torch.from_numpy(arr).to(own_state[hf_name].dtype))
    allocation.update(changed)
