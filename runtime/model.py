"""Manifest-driven Qwen2.5 loader (spec §4, M4).

Loads a HF Qwen2 model shell from config only (no pretrained fp16 download — the weights come
from the .tsra artifact), then materializes each parameter by reading exactly the tensor/bits
called for in a profile manifest's `allocation` map, via format.load.TsraFile (spec §5.3: this is
one mmap range per tensor, no re-quantization).

Requires torch + transformers, and a real Qwen2.5-1.5B/-0.5B checkout for config.json + tokenizer
(weights are NOT downloaded — see `from_manifest`). Not runnable in this environment (no
torch/transformers installed here); the tensor-name mapping is unit-testable on its own in
test_model.py without either dependency.
"""
from __future__ import annotations

from typing import Dict

from format.load import TsraFile

N_LAYERS_1_5B = 28
N_LAYERS_0_5B = 24  # Qwen2.5-0.5B-Instruct — verify against that model's config.json (spec §3 note)

# .tsra tensor name -> HF Qwen2 state_dict key template. `{i}` is the layer index.
_PER_LAYER_MAP = {
    "attn_q": "model.layers.{i}.self_attn.q_proj.weight",
    "attn_k": "model.layers.{i}.self_attn.k_proj.weight",
    "attn_v": "model.layers.{i}.self_attn.v_proj.weight",
    "attn_o": "model.layers.{i}.self_attn.o_proj.weight",
    "mlp_gate": "model.layers.{i}.mlp.gate_proj.weight",
    "mlp_up": "model.layers.{i}.mlp.up_proj.weight",
    "mlp_down": "model.layers.{i}.mlp.down_proj.weight",
    "attn_q.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "attn_k.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "attn_v.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
}
_GLOBAL_MAP = {
    "token_embd": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    # tied embeddings (spec §3: tied_embeddings: yes) — lm_head shares token_embd, no separate entry
}


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
