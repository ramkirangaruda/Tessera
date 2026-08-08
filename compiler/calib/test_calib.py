"""Tests for assert_thinking_disabled (spec §4.1 amendment) against fake tokenizer stand-ins —
no transformers/network required."""
from __future__ import annotations

import pytest

from compiler.calib import assert_thinking_disabled, render_chat_prompt


class FakeQwen3Tokenizer:
    """Mimics the parts of a real Qwen3 tokenizer assert_thinking_disabled touches: a
    chat_template string containing 'enable_thinking', and an apply_chat_template that accepts
    the kwarg."""

    chat_template = "some jinja source ... {% if enable_thinking %} ... {% endif %} ..."

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert enable_thinking is False  # this is what we're actually verifying got passed
        return "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n"


class FakeBrokenQwen3Tokenizer(FakeQwen3Tokenizer):
    def apply_chat_template(self, *args, **kwargs):
        raise TypeError("enable_thinking is not a recognized argument")


class FakeQwen2TokenizerNoThinking:
    """No thinking mode at all — spec §13 secondary validation run."""

    chat_template = "some jinja source with no thinking concept at all"

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("should never be called — no enable_thinking in this template")


class FakeTokenizerNoTemplate:
    chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("should never be called")


def test_passes_for_qwen3_style_tokenizer():
    assert_thinking_disabled(FakeQwen3Tokenizer())  # should not raise


def test_noop_for_tokenizer_without_thinking_mode():
    assert_thinking_disabled(FakeQwen2TokenizerNoThinking())  # should not raise, not even call the template


def test_noop_for_tokenizer_with_no_chat_template():
    assert_thinking_disabled(FakeTokenizerNoTemplate())  # should not raise


def test_raises_if_enable_thinking_kwarg_rejected():
    with pytest.raises(AssertionError):
        assert_thinking_disabled(FakeBrokenQwen3Tokenizer())


class RecordingQwen3Tokenizer(FakeQwen3Tokenizer):
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_kwargs = kwargs
        return "rendered"


class RecordingQwen2Tokenizer(FakeQwen2TokenizerNoThinking):
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_kwargs = kwargs
        return "rendered"


def test_render_chat_prompt_passes_enable_thinking_false_for_qwen3():
    tok = RecordingQwen3Tokenizer()
    result = render_chat_prompt(tok, [{"role": "user", "content": "hi"}])
    assert result == "rendered"
    assert tok.last_kwargs == {"enable_thinking": False}


def test_render_chat_prompt_omits_kwarg_for_non_thinking_model():
    """Qwen2.5's template has no enable_thinking param — passing it would TypeError."""
    tok = RecordingQwen2Tokenizer()
    result = render_chat_prompt(tok, [{"role": "user", "content": "hi"}])
    assert result == "rendered"
    assert tok.last_kwargs == {}
