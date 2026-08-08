"""Unit test for the bounded warm-prefill prompt builder — pure string logic (spec §6), no
torch/transformers/model required."""
from __future__ import annotations

from runtime.generate import WARM_PREFILL_LAST_N_TURNS, build_prefill_prompt


def test_prefill_includes_summary_and_system_prompt():
    prompt = build_prefill_prompt("Be helpful.", "User is debugging a parser.", [], [])
    assert "Be helpful." in prompt
    assert "User is debugging a parser." in prompt


def test_prefill_caps_verbatim_turns_at_last_n():
    turns = [{"role": "user", "content": f"turn{i:03d}end"} for i in range(20)]
    prompt = build_prefill_prompt("sys", "summary", turns, [])
    for i in range(20 - WARM_PREFILL_LAST_N_TURNS):
        assert f"turn{i:03d}end" not in prompt
    for i in range(20 - WARM_PREFILL_LAST_N_TURNS, 20):
        assert f"turn{i:03d}end" in prompt


def test_prefill_cost_is_bounded_regardless_of_session_length():
    """Prompt length must not grow with total transcript length once past the last-N window —
    only with the fixed-size summary + last-N turns + retrieved spans (spec §6: 'constant
    prefill cost'). Both session lengths here exceed WARM_PREFILL_LAST_N_TURNS, so both are
    already clamped to the same window."""
    turns_a = [{"role": "user", "content": "hi"} for _ in range(WARM_PREFILL_LAST_N_TURNS + 3)]
    turns_b = [{"role": "user", "content": "hi"} for _ in range(3000)]
    prompt_a = build_prefill_prompt("sys", "summary", turns_a, [])
    prompt_b = build_prefill_prompt("sys", "summary", turns_b, [])
    assert len(prompt_b) == len(prompt_a)
