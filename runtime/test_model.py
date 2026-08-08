"""Unit tests for the .tsra <-> HF Qwen2 name mapping — pure string logic, no torch/transformers
or model download required. The materialization path (from_manifest, drop_to_bits) needs torch +
transformers + a real config.json and is not exercised here.
"""
from __future__ import annotations

import pytest

from runtime.model import tsra_name_to_hf_name


def test_global_tensors():
    assert tsra_name_to_hf_name("token_embd") == "model.embed_tokens.weight"
    assert tsra_name_to_hf_name("output_norm.weight") == "model.norm.weight"


def test_per_layer_weight_tensors():
    assert tsra_name_to_hf_name("blk.0.attn_q") == "model.layers.0.self_attn.q_proj.weight"
    assert tsra_name_to_hf_name("blk.27.mlp_down") == "model.layers.27.mlp.down_proj.weight"
    assert tsra_name_to_hf_name("blk.13.attn_o") == "model.layers.13.self_attn.o_proj.weight"


def test_per_layer_bias_and_norm_tensors():
    assert tsra_name_to_hf_name("blk.5.attn_q.bias") == "model.layers.5.self_attn.q_proj.bias"
    assert tsra_name_to_hf_name("blk.5.attn_norm.weight") == "model.layers.5.input_layernorm.weight"
    assert tsra_name_to_hf_name("blk.5.ffn_norm.weight") == "model.layers.5.post_attention_layernorm.weight"


def test_unrecognized_name_raises():
    with pytest.raises(ValueError):
        tsra_name_to_hf_name("not_a_real_tensor")
    with pytest.raises(ValueError):
        tsra_name_to_hf_name("blk.0.not_a_kind")


def test_all_197_quantizable_names_resolve():
    from compiler.sweep import quantizable_tensor_names

    for name in quantizable_tensor_names():
        # should not raise
        tsra_name_to_hf_name(name)
