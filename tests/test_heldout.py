"""The held-out split must share no tool with any calibration data, and the failure categories must be right."""
import json
import random

import pytest

from src import data as D
from src import heldout as H


def calibration_episodes(seed, n=600):
    """Agentic calibration episodes in the order src.run.build_calib draws them (Random(1000 + seed)).
    build_calib stops at a token budget (about 15 episodes, about 75 for the matched ablation), so 600 is a superset."""
    r = random.Random(1000 + seed)
    return [D.sample_episode(r) for _ in range(n)]


def test_heldout_tool_names_disjoint_from_calibration_tools():
    assert set(H.HELDOUT_TOOLS).isdisjoint(D.TOOLS)
    for a in H.HELDOUT_TOOLS:                                # no name is a substring of another set's name
        for b in D.TOOLS:
            assert a not in b and b not in a


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibration_traces_never_mention_heldout_tools(seed):
    seen = set()
    for ep in calibration_episodes(seed):
        seen |= set(ep.tools) | {t[1] for t in ep.turns}
        blob = json.dumps([D.tool_schema(t) for t in ep.tools]) + json.dumps(D.episode_messages(ep))
        assert not [n for n in H.HELDOUT_TOOLS if n in blob]
    assert seen <= set(D.TOOLS) and not seen & set(H.HELDOUT_TOOLS)


def test_heldout_episodes_use_only_heldout_tools_and_are_deterministic():
    eps = H.heldout_episodes(200)
    assert eps[:5] == H.heldout_episodes(5)
    for ep in eps:
        assert set(ep.tools) <= set(H.HELDOUT_TOOLS)
        for req, name, args, res, ans in ep.turns:
            assert name in ep.tools and set(args) == set(H.HELDOUT_TOOLS[name][2])
            blob = req + json.dumps(args) + json.dumps(res) + ans
            assert not [n for n in D.TOOLS if n in blob]
    assert {t[1] for ep in eps for t in ep.turns} == set(H.HELDOUT_TOOLS)   # every held-out tool is exercised


def test_gen_prompts_stop_at_the_user_request():
    eps = H.heldout_episodes(20)
    ps = H.gen_prompts(eps, 20, lambda msgs, tools: msgs, seed=5)
    for p in ps:
        msgs = p["prompt"]
        assert msgs[-1] == {"role": "user", "content": p["request"]}
        assert len(msgs) == 4 * p["round"] + 1
        assert p["gold"]["name"] == eps[p["episode"]].turns[p["round"]][1]


GOLD = {"name": "reserve_table", "arguments": {"restaurant": "Alo", "party_size": 4, "time": "19:30"}}
OFFERED = ["reserve_table", "split_bill"]


def call(obj):
    return "<tool_call>\n" + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n</tool_call>"


@pytest.mark.parametrize("text,cat", [
    (call(GOLD), "correct"),
    (call({"name": "reserve_table", "arguments": {"restaurant": "Alo", "party_size": 4.0, "time": "19:30"}}), "correct"),
    ("Sure, I booked Alo for 4 at 19:30.", "no_call"),
    (call('{"name": "reserve_table", "arguments": {"restaurant": "Alo",'), "malformed_json"),
    ("<tool_call>\n{\"name\": \"reserve_table\", \"arguments\": {\"restaurant\": \"Al", "malformed_json"),   # truncated
    (call({"name": "reserve_table", "args": {}}), "malformed_json"),
    (call({"name": "book_restaurant", "arguments": GOLD["arguments"]}), "hallucinated_tool"),
    (call({"name": "split_bill", "arguments": GOLD["arguments"]}), "wrong_tool"),
    (call({"name": "reserve_table", "arguments": {"restaurant": "Alo", "time": "19:30"}}), "wrong_arg_keys"),
    (call({"name": "reserve_table", "arguments": {"restaurant": "Alo", "party_size": "4", "time": "19:30"}}), "wrong_arg_type"),
    (call({"name": "reserve_table", "arguments": {"restaurant": "alo", "party_size": 4, "time": "19:30"}}), "wrong_arg_value"),
    (call({"name": "reserve_table", "arguments": {"restaurant": "Alo", "party_size": True, "time": "19:30"}}), "wrong_arg_value"),
])
def test_categorize(text, cat):
    assert H.categorize(text, GOLD, OFFERED)["category"] == cat


def test_categorize_bool_is_not_int():
    g = {"name": "play_playlist", "arguments": {"playlist": "Focus", "shuffle": True}}
    assert H.categorize(call({"name": "play_playlist", "arguments": {"playlist": "Focus", "shuffle": 1}}), g, ["play_playlist"])["category"] != "correct"
    assert H.categorize(call(g), g, ["play_playlist"])["category"] == "correct"
