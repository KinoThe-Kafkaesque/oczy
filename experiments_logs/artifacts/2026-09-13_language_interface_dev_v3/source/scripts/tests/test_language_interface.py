"""Check the prompt comparison before its first model call."""

import json
from collections import Counter

from scripts.context_preservation_contract import messages
from scripts.probe_language_interface import build_cases


def test_complete_crossed_coverage_and_fresh_words():
    rows = build_cases()
    assert len(rows) == 224
    assert set(Counter((r["split"], r["condition"], r["category"]) for r in rows).values()) == {4}
    words = {split: {r["input"] for r in rows if r["split"] == split} for split in ("calibration", "confirmation")}
    assert not words["calibration"] & words["confirmation"]
    assert not words["confirmation"] & {"oak", "pine", "elm", "pear", "plum", "kiwi", "fig"}


def test_no_final_transformed_target_supplied_in_any_prompt():
    for row in build_cases():
        if row["category"] in ("amber", "cobalt"):
            assert row["target"] not in json.dumps(row["messages"])


def test_layout_pairs_preserve_rule_and_query_text():
    rows = {r["condition"]: r for r in build_cases() if r["split"] == "confirmation" and r["input"] == "apple" and r["category"] == "amber"}
    old, merged = [rows[k]["messages"] for k in ("resolved_v2", "merged_resolved")]
    assert old[0] == merged[0] and merged[1]["content"] == old[1]["content"] + "\n\n" + old[2]["content"]
    old, moved = [rows[k]["messages"] for k in ("table_v2", "table_in_system")]
    assert old[-1] == moved[-1] and moved[0]["content"] == old[0]["content"] + "\n\n" + old[1]["content"]


def test_chat_retrieval_uses_all_nine_teaching_pairs_and_same_query():
    row = next(r for r in build_cases() if r["condition"] == "chat_examples")
    prompt = row["messages"]
    assert len(prompt) == 20 and prompt[-1] == messages(row)[-1]
    assert [m["role"] for m in prompt[1:-1]] == ["user", "assistant"] * 9
    assert len({m["content"] for m in prompt if m["role"] == "assistant"}) == 9
