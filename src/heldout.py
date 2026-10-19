"""Held-out-tools split and free-generation error analysis for one condition.

    python -m src.heldout --cond awq_long_masked --seed 0

The held-out split uses 10 tools whose names never occur in any calibration data (agentic traces
use only the 12 tools of src/data.py; the short-chat set is Alpaca). This is asserted twice: by
tests/test_heldout.py on the episode generator, and at run time here by decoding the exact
calibration batches this condition quantizes with and searching them for every held-out tool name.

For one condition, quantized exactly as src/run.py does it (same seed, so the same AWQ scales; the
chosen alphas are compared with results/raw/<cond>__s<seed>.json and the agreement is recorded):

1. teacher-forced tool-call metrics on N held-out episodes (same scorer as the main grid);
2. greedy free generation of the next assistant turn after a user request, for n_gen held-out
   prompts and n_gen in-distribution prompts (the main grid's eval episodes), each parsed and
   put into one failure category (see `categorize`).

Writes <out>/<cond>__s<seed>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time

import torch
from transformers import GenerationConfig

from . import data as D
from . import quant as Q
from .run import CONDS, ROOT, build_calib, load_model, score_text

P = D._p
STREETS = ["King St", "Queen St", "Bloor St", "Yonge St", "Rue Sainte-Catherine", "Granville St"]
RESTAURANTS = ["Pai", "Alo", "Joe Beef", "Miku", "Canoe", "Bar Isabel", "Kissa Tanto"]
MOVIES = ["Dune: Part Three", "The Long Walk", "Paddington in Peru", "Arrival", "Spirited Away"]


def _time(r):
    return f"{r.randint(11, 21):02d}:{r.choice(['00', '15', '30', '45'])}"


# name -> (description, properties, required, request templates, arg sampler); same layout as data.TOOLS.
HELDOUT_TOOLS = {
    "get_air_quality": ("Get the air quality index for a city.",
                        {"city": P("string", "City name"), "pollutant": P("string", "Pollutant", ["pm25", "ozone", "no2"])},
                        ["city", "pollutant"],
                        ["How bad is the {pollutant} level in {city} today?", "Check {pollutant} air quality for {city}.",
                         "What's the {pollutant} reading in {city}?"],
                        lambda r: {"city": r.choice(D.CITIES), "pollutant": r.choice(["pm25", "ozone", "no2"])}),
    "reserve_table": ("Reserve a table at a restaurant.",
                      {"restaurant": P("string", "Restaurant name"), "party_size": P("integer", "Number of guests"),
                       "time": P("string", "Time, HH:MM 24h")},
                      ["restaurant", "party_size", "time"],
                      ["Get me a table for {party_size} at {restaurant} at {time}.", "Book {restaurant} for {party_size} people at {time}.",
                       "Reserve {restaurant}, {party_size} guests, {time}."],
                      lambda r: {"restaurant": r.choice(RESTAURANTS), "party_size": r.randint(1, 8), "time": _time(r)}),
    "track_package": ("Track a parcel with a carrier.",
                      {"tracking_number": P("string", "Tracking number"),
                       "carrier": P("string", "Carrier", ["UPS", "FedEx", "Canada Post", "DHL"])},
                      ["tracking_number", "carrier"],
                      ["Where's my {carrier} parcel {tracking_number}?", "Track {tracking_number} with {carrier}.",
                       "Any update on {carrier} shipment {tracking_number}?"],
                      lambda r: {"tracking_number": f"TRK{r.randint(100000, 999999)}", "carrier": r.choice(["UPS", "FedEx", "Canada Post", "DHL"])}),
    "check_inventory": ("Check stock of a SKU at a warehouse.",
                        {"sku": P("string", "Stock keeping unit"), "warehouse": P("string", "Warehouse code", ["YYZ1", "YVR2", "YUL3"])},
                        ["sku", "warehouse"],
                        ["How many {sku} are left in {warehouse}?", "Check stock for {sku} at warehouse {warehouse}.",
                         "Is {sku} available in {warehouse}?"],
                        lambda r: {"sku": f"SKU-{r.randint(1000, 9999)}", "warehouse": r.choice(["YYZ1", "YVR2", "YUL3"])}),
    "create_support_ticket": ("Open a customer-support ticket.",
                              {"summary": P("string", "One-line summary"), "priority": P("string", "Priority", ["low", "medium", "high", "urgent"])},
                              ["summary", "priority"],
                              ["Open a {priority} priority ticket: {summary}.", "File a support ticket ({priority}) saying '{summary}'.",
                               "Log this as {priority}: {summary}."],
                              lambda r: {"summary": r.choice(["login page times out", "refund not received", "app crashes on start",
                                                               "wrong item delivered"]), "priority": r.choice(["low", "medium", "high", "urgent"])}),
    "get_route_eta": ("Estimate travel time between two places.",
                      {"origin": P("string", "Start address"), "destination": P("string", "End address"),
                       "mode": P("string", "Travel mode", ["driving", "transit", "cycling", "walking"])},
                      ["origin", "destination", "mode"],
                      ["How long from {origin} to {destination} by {mode}?", "ETA {origin} to {destination}, {mode}.",
                       "If I go {mode}, how long does {origin} to {destination} take?"],
                      lambda r: dict(zip(["origin", "destination"], [f"{r.randint(1, 900)} {s}" for s in r.sample(STREETS, 2)]),
                                     mode=r.choice(["driving", "transit", "cycling", "walking"]))),
    "convert_units": ("Convert a measurement between units.",
                      {"value": P("number", "Quantity"), "from_unit": P("string", "Source unit"), "to_unit": P("string", "Target unit")},
                      ["value", "from_unit", "to_unit"],
                      ["Convert {value} {from_unit} to {to_unit}.", "How many {to_unit} is {value} {from_unit}?",
                       "{value} {from_unit} in {to_unit}, please."],
                      lambda r: dict(zip(["from_unit", "to_unit"], r.choice([("km", "miles"), ("kg", "pounds"), ("liters", "gallons"),
                                                                             ("celsius", "fahrenheit"), ("meters", "feet")])),
                                     value=r.choice([3, 12, 42, 100, 250]))),
    "play_playlist": ("Start playing a playlist.",
                      {"playlist": P("string", "Playlist name"), "shuffle": P("boolean", "Shuffle order")},
                      ["playlist", "shuffle"],
                      ["Play my {playlist} playlist, shuffle {shuffle_word}.", "Put on {playlist} ({shuffle_word} shuffle).",
                       "Start {playlist} with shuffle {shuffle_word}."],
                      lambda r: {"playlist": r.choice(["Focus", "Road Trip", "Chill Jazz", "Workout", "Sunday Morning"]),
                                 "shuffle": r.choice([True, False])}),
    "get_showtimes": ("List movie showtimes in a city on a date.",
                      {"movie": P("string", "Movie title"), "city": P("string", "City"), "date": P("string", "Date, YYYY-MM-DD")},
                      ["movie", "city", "date"],
                      ["When is {movie} playing in {city} on {date}?", "Showtimes for {movie} in {city}, {date}.",
                       "Can I see {movie} in {city} on {date}?"],
                      lambda r: {"movie": r.choice(MOVIES), "city": r.choice(D.CITIES), "date": D._date(r)}),
    "split_bill": ("Split a bill between people, with tip.",
                   {"total": P("number", "Bill total"), "people": P("integer", "Number of people"), "tip_percent": P("integer", "Tip percent")},
                   ["total", "people", "tip_percent"],
                   ["Split {total} between {people} people with a {tip_percent}% tip.", "We're {people}; the bill is {total}, tip {tip_percent}%.",
                    "Divide a {total} bill {people} ways, {tip_percent} percent tip."],
                   lambda r: {"total": r.choice([48.5, 86, 120.4, 233, 310.75]), "people": r.randint(2, 6), "tip_percent": r.choice([10, 15, 18, 20])}),
}


def heldout_schema(name):
    desc, props, req, *_ = HELDOUT_TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def _result(name, args, r):
    return {"status": "ok", "ref": f"{name.split('_')[0][:3].upper()}-{r.randint(1000, 9999)}",
            "value": round(r.uniform(1, 500), 2)}


def _answer(name, args, res):
    return f"Done: {name.replace('_', ' ')} returned {res['value']} (reference {res['ref']})."


def sample_heldout_episode(r: random.Random, n_tools=(4, 8), n_rounds=(1, 3)) -> D.Episode:
    """Same structure as data.sample_episode, over held-out tools only."""
    ep = D.Episode(tools=r.sample(list(HELDOUT_TOOLS), r.randint(*n_tools)))
    for _ in range(r.randint(*n_rounds)):
        name = r.choice(ep.tools)
        _, _, _, templates, sampler = HELDOUT_TOOLS[name]
        args = sampler(r)
        fmt = dict(args, shuffle_word="on" if args.get("shuffle") else "off")
        res = _result(name, args, r)
        ep.turns.append((r.choice(templates).format(**fmt), name, args, res, _answer(name, args, res)))
    return ep


def render_heldout(tok, msgs, tools, add_generation_prompt=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   tools=[heldout_schema(t) for t in tools])


def heldout_episodes(n, seed=31337):
    r = random.Random(seed)
    return [sample_heldout_episode(r) for _ in range(n)]


def indist_episodes(n_agentic=200):
    """The main grid's eval episodes (src.run.eval_sets uses the same generator and seed)."""
    r = random.Random(424242)
    return [D.sample_episode(r) for _ in range(n_agentic)]


def gen_prompts(eps, n, render_fn, seed):
    """One prompt per episode (first n episodes): history up to the user request of a random round."""
    r = random.Random(seed)
    out = []
    for i, ep in enumerate(eps[:n]):
        k = r.randrange(len(ep.turns))
        msgs = D.episode_messages(D.Episode(tools=ep.tools, turns=ep.turns[:k + 1]))[:-3]   # drop call, result, answer
        _, name, args, _, _ = ep.turns[k]
        out.append({"episode": i, "round": k, "n_rounds": len(ep.turns), "tools": ep.tools,
                    "request": ep.turns[k][0], "gold": {"name": name, "arguments": args}, "prompt": render_fn(msgs, ep.tools)})
    return out


CATEGORIES = ["correct", "no_call", "malformed_json", "hallucinated_tool", "wrong_tool", "wrong_arg_keys",
              "wrong_arg_type", "wrong_arg_value"]
_CALL = re.compile(r"<tool_call>(.*?)(</tool_call>|$)", re.S)


def categorize(text: str, gold: dict, offered: list[str]) -> dict:
    """Parse the first <tool_call> of a generation and assign exactly one category (first that applies):
    no_call (no <tool_call> tag: prose instead of a call), malformed_json (tag present but the body is not
    a JSON object with a string "name" and an object "arguments"), hallucinated_tool (name not offered),
    wrong_tool (another offered tool), wrong_arg_keys (missing or extra argument names), wrong_arg_type
    (same keys; some value equal only after str(), e.g. "50" vs 50), wrong_arg_value, else correct."""
    m = _CALL.search(text)
    out = {"category": None, "n_calls": text.count("<tool_call>"), "detail": ""}
    if not m:
        out["category"] = "no_call"
        return out
    try:
        call = json.loads(m.group(1).strip())
        if not (isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)):
            raise ValueError("not a {name, arguments} object")
    except (json.JSONDecodeError, ValueError) as e:
        out.update(category="malformed_json", detail=str(e)[:120])
        return out
    name, args = call["name"], call["arguments"]
    out["pred"] = call
    if name != gold["name"]:
        out.update(category="wrong_tool" if name in offered else "hallucinated_tool", detail=f"{name} (gold {gold['name']})")
        return out
    gk, pk = set(gold["arguments"]), set(args)
    if gk != pk:
        out.update(category="wrong_arg_keys", detail=f"missing {sorted(gk - pk)} extra {sorted(pk - gk)}")
        return out

    def same(a, b):
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b                                    # 86 and 86.0 are the same number
        return type(a) is type(b) and a == b
    bad = [k for k in gk if not same(args[k], gold["arguments"][k])]
    if not bad:
        out["category"] = "correct"
    elif all(str(args[k]) == str(gold["arguments"][k]) for k in bad):
        out.update(category="wrong_arg_type", detail="; ".join(f"{k}: {args[k]!r} vs {gold['arguments'][k]!r}" for k in bad))
    else:
        case = all(str(args[k]).lower() == str(gold["arguments"][k]).lower() for k in bad)
        out.update(category="wrong_arg_value", case_only=case,
                   detail="; ".join(f"{k}: {args[k]!r} vs {gold['arguments'][k]!r}" for k in bad))
    return out


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=96, log_every=20, t0=None):
    cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                           eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id],
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for j, p in enumerate(prompts):
        ids = torch.tensor([tok.encode(p["prompt"], add_special_tokens=False)])
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
        p["output"] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
        p["prompt_tokens"] = ids.shape[1]
        p |= categorize(p["output"], p["gold"], p["tools"])
        if t0 is not None and j % log_every == log_every - 1:
            print(f"gen {j + 1}/{len(prompts)} t={time.time() - t0:.0f}s", flush=True)
    return prompts


def calib_overlap(tok, batches):
    """Held-out tool names found in the decoded calibration tokens (must be empty)."""
    text = "".join(tok.decode(ids[0]) for ids, _ in batches)
    return sorted(n for n in HELDOUT_TOOLS if n in text)


def quantize(cond, seed, tok, model, n_loss_tokens=1024):
    """Exactly src/run.py's quantization path (W4, fp32 group scales, no A8)."""
    meta = {}
    torch.manual_seed(seed)
    if cond == "rtn":
        Q.rtn_model(model)
    elif cond != "fp":
        src, keep, opt = CONDS[cond]
        batches, meta["calib"] = build_calib(tok, src, keep, seed, opt.get("match_effective", False))
        meta["calib_heldout_names_found"] = calib_overlap(tok, batches)
        assert not meta["calib_heldout_names_found"], meta["calib_heldout_names_found"]
        stats = Q.Collector(model, n_loss_tokens, seed=seed).run(model, batches)
        meta["awq_log"] = []
        Q.awq_model(model, stats, log=meta["awq_log"])
        ref = ROOT / f"results/raw/{cond}__s{seed}.json"
        if ref.exists():
            ra = [g["best_alpha"] for g in json.loads(ref.read_text())["meta"]["awq_log"]]
            mine = [g["best_alpha"] for g in meta["awq_log"]]
            meta["alpha_agreement_with_main_grid"] = sum(a == b for a, b in zip(ra, mine)) / len(ra)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=150, help="held-out episodes, teacher-forced")
    ap.add_argument("--n-gen", type=int, default=80, help="free-generation prompts per split (held-out, in-distribution)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/heldout")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tok, model = load_model()
    meta = {"cond": a.cond, "seed": a.seed, "threads": a.threads, "n_episodes": a.n_episodes, "n_gen": a.n_gen,
            "heldout_tools": sorted(HELDOUT_TOOLS), "calibration_tools": sorted(D.TOOLS)}
    meta |= quantize(a.cond, a.seed, tok, model)
    meta["t_quant_s"] = round(time.time() - t0, 1)
    rh = lambda msgs, tools: render_heldout(tok, msgs, tools, add_generation_prompt=True)
    ri = lambda msgs, tools: D.render(tok, msgs, tools=tools, add_generation_prompt=True)
    ho_eps = heldout_episodes(max(a.n_episodes, a.n_gen))
    texts = [render_heldout(tok, D.episode_messages(ep), ep.tools) for ep in ho_eps[:a.n_episodes]]
    meta["heldout_sha"] = hashlib.sha256("".join(texts).encode()).hexdigest()[:16]
    tf = []
    for j, t in enumerate(texts):
        tf.append(score_text(model, tok, t))
        if j % 50 == 49:
            print(f"heldout tf {j + 1}/{len(texts)} t={time.time() - t0:.0f}s", flush=True)
    gen = {"heldout": generate(model, tok, gen_prompts(ho_eps, a.n_gen, rh, 5), t0=t0),
           "indist": generate(model, tok, gen_prompts(indist_episodes(), a.n_gen, ri, 5), t0=t0)}
    for split in gen.values():
        for p in split:
            p.pop("prompt")          # reproducible from the seeds; keeps the json small
    meta["t_total_s"] = round(time.time() - t0, 1)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.cond}__s{a.seed}.json").write_text(json.dumps({"meta": meta, "teacher_forced": tf, "generation": gen}))
    calls = [s for ep in tf for s in ep if s["kind"] == "call"]
    acc = {k: round(sum(p["category"] == "correct" for p in v) / max(len(v), 1), 3) for k, v in gen.items()}
    print(a.cond, a.seed, "heldout_call_exact", round(sum(s["exact"] for s in calls) / max(len(calls), 1), 3),
          "gen_correct", acc, "alpha_agree", meta.get("alpha_agreement_with_main_grid"), "t", meta["t_total_s"], flush=True)


if __name__ == "__main__":
    main()
